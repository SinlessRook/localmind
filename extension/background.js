import {
  savePageToMemory,
  searchMemory,
  getRecentPages,
  getBlacklist,
  addToBlacklist,
  removeFromBlacklist,
  isBlacklisted,
  deleteByUrlEverywhere,
  deleteByMemoryIdEverywhere,
  clearAllStoredMemories,
} from './utils/storage.js';
import { generateEmbedding } from './utils/embeddings.js';
import { inferSemanticTagsFromText, parseNaturalLanguageQuery } from './utils/query-parser.js';

// ── Search engine URL patterns ────────────────────────────────────────────────

const SEARCH_ENGINES = [
  { hostname: 'www.google.com',     param: 'q',      path: '/search' },
  { hostname: 'www.bing.com',       param: 'q',      path: '/search' },
  { hostname: 'duckduckgo.com',     param: 'q',      path: '/'       },
  { hostname: 'search.yahoo.com',   param: 'p',      path: '/search' },
  { hostname: 'search.brave.com',   param: 'q',      path: '/search' },
];
const BACKEND_STORE_URL = 'http://127.0.0.1:8000/store';
const BOOKMARK_RESYNC_INTERVAL_MS = 24 * 60 * 60 * 1000;
const BOOKMARK_FETCH_TIMEOUT_MS = 5000;
const INSTALL_TS_KEY = 'lm_install_timestamp';
const RECORDING_ENABLED_KEY = 'lm_recording_enabled';
const SHOW_INJECTED_SUGGESTIONS_KEY = 'lm_show_injected_suggestions';
const DEBUG_QUERY_EXPANSION = true;
const LIVE_VERIFY_TOP_N = 6;
const LIVE_VERIFY_TIMEOUT_MS = 1800;
const TAB_INDEX_SNAPSHOT_KEY = 'lm_semantic_tab_index_enc';
const TAB_INDEX_CRYPTO_KEY = 'lm_semantic_tab_crypto_key';
const SEMANTIC_TAB_TOP_K = 3;
const LOCAL_LLM_VERIFY_URL = 'http://127.0.0.1:11434/api/generate';
const LOCAL_LLM_VERIFY_TIMEOUT_MS = 3500;
const TAB_DOM_REFRESH_MS = 2 * 60 * 1000;
const TAB_DOM_TEXT_MAX_CHARS = 2200;

const semanticTabIndex = new Map();
let persistTabIndexTimeout = null;
let activeTabSession = null;

// ── Open sidebar when extension icon is clicked ───────────────────────────────

chrome.action.onClicked.addListener(async (tab) => {
  await chrome.action.setBadgeText({ text: '' });
  await chrome.action.setTitle({ title: 'Open Local Mind' });
  await chrome.sidePanel.open({ windowId: tab.windowId });
});

// ── Auto-enable sidebar on search engine pages ────────────────────────────────

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (!tab.url) return;

  const query = extractSearchQuery(tab.url);

  if (query) {
    await chrome.sidePanel.setOptions({
      tabId,
      path: 'sidebar/sidebar.html',
      enabled: true,
    });

    await chrome.storage.session.set({
      pendingQuery: query,
      pendingTabId: tabId,
    });

  } else {
    await chrome.sidePanel.setOptions({
      tabId,
      path: 'sidebar/sidebar.html',
      enabled: true,
    });
    const { pendingTabId } = await chrome.storage.session.get('pendingTabId');
    if (pendingTabId === tabId) {
      await chrome.storage.session.remove(['pendingQuery', 'pendingTabId']);
    }
  }
});

// ── Extract search query from URL ─────────────────────────────────────────────

function extractSearchQuery(url) {
  try {
    const u = new URL(url);
    const engine = SEARCH_ENGINES.find(e => e.hostname === u.hostname);
    if (!engine) return null;
    if (!u.pathname.startsWith(engine.path)) return null;
    const q = u.searchParams.get(engine.param);
    return q && q.trim().length > 0 ? q.trim() : null;
  } catch {
    return null;
  }
}

// ── Message Router ────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse).catch(err => {
    console.error('[LocalMind BG]', err);
    sendResponse({ error: err.message });
  });
  return true;
});

async function handleMessage(message, sender) {
  switch (message.type) {

    case 'OPEN_SIDEBAR_WITH_QUERY': {
      await chrome.storage.session.set({ pendingQuery: message.query });
      await chrome.action.setBadgeText({ text: '●' });
      await chrome.action.setBadgeBackgroundColor({ color: '#7c6af7' });
      await chrome.action.setTitle({ title: 'Local Mind — click to see memory results' });
      return { ok: true };
    }

    case 'PAGE_CONTENT':
      await indexPage(message);
      return { ok: true };

    case 'SEARCH_QUERY':
      return { results: await handleSearch(message.query, message.sourceFilter || 'all') };

    case 'SEMANTIC_TAB_SWITCH': {
      return await switchToSemanticTab(message.query || '');
    }

    case 'SEMANTIC_TAB_INDEX_STATUS': {
      return {
        indexedTabs: semanticTabIndex.size,
      };
    }

    case 'GET_RECENT':
      return { pages: await getRecentPages(message.limit || 20) };

    case 'EXTRACT_SEARCH_QUERY_FROM_URL':
      return { query: extractSearchQuery(message.url) };

    case 'GET_PENDING_QUERY': {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const activeUrl = tabs[0]?.url;
      if (activeUrl && !extractSearchQuery(activeUrl)) {
        await chrome.storage.session.remove(['pendingQuery', 'pendingTabId']);
        return { query: null };
      }
      const data = await chrome.storage.session.get(['pendingQuery', 'pendingTabId']);
      if (data.pendingQuery) {
        await chrome.storage.session.remove(['pendingQuery', 'pendingTabId']);
        return { query: data.pendingQuery };
      }
      return { query: null };
    }

    case 'CLEAR_MEMORY': {
      const allData = await chrome.storage.local.get(null);
      const now = Date.now();

      const cutoff = {
        hour: now - 60 * 60 * 1000,
        day:  now - 24 * 60 * 60 * 1000,
        all:  0,
      }[message.range || 'all'];

      const pageKeys = Object.keys(allData).filter(k => {
        if (!k.startsWith('page_')) return false;
        if (message.range === 'all') return true;
        return (allData[k].timestamp || 0) >= cutoff;
      });

      if ((message.range || 'all') === 'all') {
        await clearAllStoredMemories();
      } else {
        await chrome.storage.local.remove(pageKeys);
      }

      if ((message.range || 'all') === 'all') {
        try {
          await fetch(`${BACKEND_STORE_URL.replace('/store', '')}/clear-memory`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
          });
        } catch (err) {
          console.warn('[LocalMind] Failed to clear backend memory:', err);
        }
      }

      if (message.range === 'all') {
        await chrome.storage.session.clear();
        await chrome.action.setBadgeText({ text: '' });
        await chrome.action.setTitle({ title: 'Open Local Mind' });
      }

      return { ok: true };
    }

    case 'GET_RECORDING_STATE': {
      const data = await chrome.storage.local.get(RECORDING_ENABLED_KEY);
      const enabled = data[RECORDING_ENABLED_KEY] !== false;
      return { enabled };
    }

    case 'SET_RECORDING_STATE': {
      const enabled = message.enabled !== false;
      await chrome.storage.local.set({ [RECORDING_ENABLED_KEY]: enabled });
      return { ok: true, enabled };
    }

    case 'GET_INJECTED_SUGGESTIONS_STATE': {
      const data = await chrome.storage.local.get(SHOW_INJECTED_SUGGESTIONS_KEY);
      const enabled = data[SHOW_INJECTED_SUGGESTIONS_KEY] !== false;
      return { enabled };
    }

    case 'SET_INJECTED_SUGGESTIONS_STATE': {
      const enabled = message.enabled !== false;
      await chrome.storage.local.set({ [SHOW_INJECTED_SUGGESTIONS_KEY]: enabled });
      return { ok: true, enabled };
    }

    case 'DELETE_PAGE': {
      if (message.memoryId) {
        await deleteByMemoryIdEverywhere(message.memoryId);
      } else {
        await deleteByUrlEverywhere(message.url);
      }
      return { ok: true };
    }

    case 'GET_BLACKLIST':
      return { list: await getBlacklist() };

    case 'ADD_TO_BLACKLIST': {
      await addToBlacklist(message.domain);
      return { ok: true };
    }

    case 'REMOVE_FROM_BLACKLIST': {
      await removeFromBlacklist(message.domain);
      return { ok: true };
    }

    default:
      return { error: 'Unknown message type' };
  }
}

// ── Semantic Tab Switcher ───────────────────────────────────────────────────

function isIndexableTabUrl(url) {
  if (!url || typeof url !== 'string') return false;
  if (url.startsWith('chrome://') || url.startsWith('chrome-extension://') || url.startsWith('edge://')) return false;
  return /^https?:\/\//i.test(url);
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const av = Number(a[i] || 0);
    const bv = Number(b[i] || 0);
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function buildTabEmbeddingText(tab, domSnapshot = null) {
  const title = tab?.title || '';
  const url = tab?.url || '';
  const host = (() => {
    try { return new URL(url).hostname; } catch { return ''; }
  })();
  const meta = domSnapshot?.metaDescription || '';
  const headings = Array.isArray(domSnapshot?.headings) ? domSnapshot.headings.join(' | ') : '';
  const body = domSnapshot?.bodyText || '';
  return `Tab title: ${title}. Tab url: ${url}. Domain: ${host}. Meta: ${meta}. Headings: ${headings}. Content: ${body}`;
}

function tokenizeSemantic(text) {
  return (text || '').toLowerCase().match(/[a-z0-9_.-]{2,}/g) || [];
}

function tokenOverlapRatio(aTokens, bTokens) {
  if (!aTokens?.length || !bTokens?.length) return 0;
  const a = new Set(aTokens);
  const b = new Set(bTokens);
  let hits = 0;
  a.forEach((t) => {
    if (b.has(t)) hits += 1;
  });
  return hits / Math.max(1, a.size);
}

function shouldRefreshDomSnapshot(existing, tab) {
  if (!existing) return true;
  if (tab?.status === 'complete') return true;
  const last = Number(existing.domIndexedAt || 0);
  return !last || (Date.now() - last) > TAB_DOM_REFRESH_MS;
}

async function fetchTabDomSnapshot(tabId) {
  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId },
      world: 'MAIN',
      func: () => {
        const pick = (sel) => Array.from(document.querySelectorAll(sel)).map((e) => (e.textContent || '').trim()).filter(Boolean);
        const desc = document.querySelector('meta[name="description"]')?.content || '';
        const h1 = pick('h1').slice(0, 3);
        const h2 = pick('h2').slice(0, 4);
        const para = pick('p').slice(0, 8).join(' ');
        const article = pick('article p').slice(0, 10).join(' ');
        const body = (article || para || (document.body?.innerText || '')).replace(/\s+/g, ' ').trim();
        return {
          metaDescription: desc,
          headings: [...h1, ...h2].slice(0, 8),
          bodyText: body.slice(0, 2200),
        };
      },
    });

    const snap = result?.result;
    if (!snap) return null;
    return {
      metaDescription: String(snap.metaDescription || '').slice(0, 300),
      headings: Array.isArray(snap.headings) ? snap.headings.slice(0, 8).map((x) => String(x || '').slice(0, 140)) : [],
      bodyText: String(snap.bodyText || '').slice(0, TAB_DOM_TEXT_MAX_CHARS),
    };
  } catch {
    return null;
  }
}

function toBase64(bytes) {
  let binary = '';
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  for (let i = 0; i < arr.length; i += 1) binary += String.fromCharCode(arr[i]);
  return btoa(binary);
}

function fromBase64(base64) {
  const binary = atob(base64 || '');
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

async function getOrCreateTabIndexCryptoKey() {
  const stored = await chrome.storage.session.get(TAB_INDEX_CRYPTO_KEY);
  const existing = stored?.[TAB_INDEX_CRYPTO_KEY];
  if (existing) {
    const raw = fromBase64(existing);
    return crypto.subtle.importKey('raw', raw, 'AES-GCM', false, ['encrypt', 'decrypt']);
  }

  const key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
  const exported = await crypto.subtle.exportKey('raw', key);
  await chrome.storage.session.set({ [TAB_INDEX_CRYPTO_KEY]: toBase64(new Uint8Array(exported)) });
  return key;
}

async function encryptTabIndexPayload(payload) {
  const key = await getOrCreateTabIndexCryptoKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plain = new TextEncoder().encode(JSON.stringify(payload));
  const cipher = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, plain);
  return { iv: toBase64(iv), cipher: toBase64(new Uint8Array(cipher)) };
}

async function decryptTabIndexPayload(blob) {
  if (!blob?.iv || !blob?.cipher) return null;
  const key = await getOrCreateTabIndexCryptoKey();
  const iv = fromBase64(blob.iv);
  const cipher = fromBase64(blob.cipher);
  const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, cipher);
  const text = new TextDecoder().decode(plain);
  return JSON.parse(text);
}

async function persistSemanticTabIndex() {
  const payload = {
    ts: Date.now(),
    tabs: Array.from(semanticTabIndex.values()),
  };
  const encrypted = await encryptTabIndexPayload(payload);
  await chrome.storage.session.set({ [TAB_INDEX_SNAPSHOT_KEY]: encrypted });
}

function schedulePersistSemanticTabIndex() {
  if (persistTabIndexTimeout) clearTimeout(persistTabIndexTimeout);
  persistTabIndexTimeout = setTimeout(() => {
    persistSemanticTabIndex().catch((err) => {
      console.warn('[LocalMind][SemanticTab] persist failed', err);
    });
  }, 220);
}

async function restoreSemanticTabIndex() {
  try {
    const stored = await chrome.storage.session.get(TAB_INDEX_SNAPSHOT_KEY);
    const encrypted = stored?.[TAB_INDEX_SNAPSHOT_KEY];
    if (!encrypted) return false;
    const payload = await decryptTabIndexPayload(encrypted);
    const tabs = payload?.tabs;
    if (!Array.isArray(tabs) || !tabs.length) return false;
    semanticTabIndex.clear();
    tabs.forEach((item) => {
      if (!item || typeof item.tabId !== 'number') return;
      semanticTabIndex.set(item.tabId, item);
    });
    return true;
  } catch (err) {
    console.warn('[LocalMind][SemanticTab] restore failed', err);
    return false;
  }
}

async function indexSingleTab(tab) {
  if (!tab || typeof tab.id !== 'number') return;
  if (!isIndexableTabUrl(tab.url)) {
    semanticTabIndex.delete(tab.id);
    schedulePersistSemanticTabIndex();
    return;
  }

  try {
    const existing = semanticTabIndex.get(tab.id);
    let domSnapshot = existing?.domSnapshot || null;
    if (shouldRefreshDomSnapshot(existing, tab)) {
      domSnapshot = await fetchTabDomSnapshot(tab.id) || domSnapshot;
    }

    const text = buildTabEmbeddingText(tab, domSnapshot);
    const embedding = await generateEmbedding(text, { taskType: 'RETRIEVAL_DOCUMENT', title: tab.title || undefined });
    semanticTabIndex.set(tab.id, {
      tabId: tab.id,
      windowId: tab.windowId,
      url: tab.url,
      title: tab.title || tab.url,
      active: !!tab.active,
      pinned: !!tab.pinned,
      audible: !!tab.audible,
      lastAccessed: Number(tab.lastAccessed || Date.now()),
      indexedAt: Date.now(),
      totalActiveMs: Number(existing?.totalActiveMs || 0),
      domIndexedAt: Date.now(),
      domSnapshot: domSnapshot || null,
      embedding,
    });
    schedulePersistSemanticTabIndex();
  } catch (err) {
    console.warn('[LocalMind][SemanticTab] index failed', err);
  }
}

function endActiveTabSession(nextTabId = null) {
  if (!activeTabSession || typeof activeTabSession.tabId !== 'number') {
    activeTabSession = nextTabId != null ? { tabId: nextTabId, startedAt: Date.now() } : null;
    return;
  }

  const prevId = activeTabSession.tabId;
  const delta = Math.max(0, Date.now() - Number(activeTabSession.startedAt || Date.now()));
  if (delta > 0) {
    const prev = semanticTabIndex.get(prevId);
    if (prev) {
      prev.totalActiveMs = Number(prev.totalActiveMs || 0) + delta;
      prev.lastAccessed = Date.now();
      semanticTabIndex.set(prevId, prev);
      schedulePersistSemanticTabIndex();
    }
  }

  activeTabSession = nextTabId != null ? { tabId: nextTabId, startedAt: Date.now() } : null;
}

async function primeActiveTabSession() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const tab = tabs?.[0];
    if (tab?.id != null) {
      activeTabSession = { tabId: tab.id, startedAt: Date.now() };
    }
  } catch {
    // ignore
  }
}

async function rebuildSemanticTabIndexFromOpenTabs() {
  const tabs = await chrome.tabs.query({});
  const indexable = tabs.filter((t) => isIndexableTabUrl(t.url));
  await Promise.all(indexable.map((t) => indexSingleTab(t)));
}

async function parseLocalLlmChoice(raw) {
  const txt = String(raw || '').trim();
  const m = txt.match(/\b([1-3])\b/);
  if (!m) return null;
  return Number(m[1]) - 1;
}

async function verifyBestSemanticTabWithLocalLlm(query, candidates) {
  if (!candidates?.length) return null;

  const prompt = [
    'You are selecting which browser tab best matches a user query.',
    'Return ONLY a single number: 1, 2, or 3.',
    `User query: ${query}`,
    'Candidates:',
    ...candidates.map((c, i) => {
      const domMeta = c?.domSnapshot?.metaDescription || '';
      const heads = (c?.domSnapshot?.headings || []).slice(0, 3).join(' | ');
      const lastSeen = Number(c.lastAccessed || 0);
      const dwellSec = Math.round(Number(c.totalActiveMs || 0) / 1000);
      return `${i + 1}. title=${c.title}; url=${c.url}; headings=${heads}; meta=${domMeta}; lastAccessed=${lastSeen}; dwellSec=${dwellSec}; score=${(c.finalScore || 0).toFixed(4)}`;
    }),
  ].join('\n');

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), LOCAL_LLM_VERIFY_TIMEOUT_MS);
    const res = await fetch(LOCAL_LLM_VERIFY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'qwen2.5:0.5b', stream: false, prompt }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const out = data?.response || '';
    const idx = await parseLocalLlmChoice(out);
    if (idx == null || idx < 0 || idx >= candidates.length) return null;
    return candidates[idx];
  } catch (err) {
    console.warn('[LocalMind][SemanticTab] local LLM verify failed, fallback to top score', err);
    return null;
  }
}

async function getSemanticTabCandidates(query, topK = SEMANTIC_TAB_TOP_K) {
  const parsed = parseNaturalLanguageQuery(query || '');
  const queryForEmbedding = parsed?.retrievalQuery || query || '';
  const qEmbedding = await generateEmbedding(queryForEmbedding, { taskType: 'RETRIEVAL_QUERY' });
  const qTerms = tokenizeSemantic(query);
  const exactPhrases = parsed?.exactPhrases || [];
  const domainHint = parsed?.domainHint || '';

  const scored = Array.from(semanticTabIndex.values())
    .map((tab) => {
      const similarity = cosineSimilarity(tab.embedding, qEmbedding);
      const tabText = `${tab.title || ''} ${tab.url || ''} ${tab?.domSnapshot?.metaDescription || ''} ${(tab?.domSnapshot?.headings || []).join(' ')} ${tab?.domSnapshot?.bodyText || ''}`.toLowerCase();
      const docTerms = tokenizeSemantic(tabText);
      const overlap = tokenOverlapRatio(qTerms, docTerms);
      const recencyBoost = Math.max(0, 1 - ((Date.now() - Number(tab.lastAccessed || 0)) / (1000 * 60 * 60 * 24))) * 0.10;
      const dwellBoost = Math.min(0.08, Math.log1p(Math.max(0, Number(tab.totalActiveMs || 0) / 1000)) / 50);
      const exactBoost = exactPhrases.length && exactPhrases.some((p) => tabText.includes(String(p || '').toLowerCase())) ? 0.16 : 0;
      const domainBoost = domainHint && String(tab.url || '').includes(domainHint) ? 0.12 : 0;
      const finalScore = (similarity * 0.72) + (overlap * 0.16) + recencyBoost + dwellBoost + exactBoost + domainBoost + (tab.active ? 0.02 : 0) + (tab.pinned ? 0.01 : 0);
      return { ...tab, similarity, overlap, recencyBoost, dwellBoost, exactBoost, domainBoost, finalScore };
    })
    .sort((a, b) => b.finalScore - a.finalScore)
    .slice(0, Math.max(1, topK));

  return scored;
}

async function switchToSemanticTab(query) {
  const q = String(query || '').trim();
  if (q.length < 2) {
    return { ok: false, reason: 'empty_query' };
  }

  if (!semanticTabIndex.size) {
    await rebuildSemanticTabIndexFromOpenTabs();
  }

  const candidates = await getSemanticTabCandidates(q, SEMANTIC_TAB_TOP_K);
  if (!candidates.length) {
    return { ok: false, reason: 'no_indexed_tabs' };
  }

  const llmChoice = await verifyBestSemanticTabWithLocalLlm(q, candidates);
  const best = llmChoice || candidates[0];
  if (!best || typeof best.tabId !== 'number') {
    return { ok: false, reason: 'no_match' };
  }

  try {
    await chrome.windows.update(best.windowId, { focused: true });
    await chrome.tabs.update(best.tabId, { active: true });
    return {
      ok: true,
      switchedTabId: best.tabId,
      switchedUrl: best.url,
      switchedTitle: best.title,
      usedLocalLlm: !!llmChoice,
      candidates: candidates.map((c) => ({ tabId: c.tabId, title: c.title, url: c.url, score: c.finalScore })),
    };
  } catch (err) {
    return { ok: false, reason: 'switch_failed', error: String(err?.message || err) };
  }
}

async function initializeSemanticTabIndexer() {
  const restored = await restoreSemanticTabIndex();
  if (!restored) {
    await rebuildSemanticTabIndexFromOpenTabs();
  }
  await primeActiveTabSession();
}

chrome.tabs.onCreated.addListener((tab) => {
  indexSingleTab(tab);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.url || changeInfo.title || changeInfo.status === 'complete') {
    indexSingleTab(tab);
  }
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  endActiveTabSession(tabId);
  try {
    const tab = await chrome.tabs.get(tabId);
    await indexSingleTab(tab);
  } catch {
    // ignore
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (activeTabSession?.tabId === tabId) {
    endActiveTabSession(null);
  }
  semanticTabIndex.delete(tabId);
  schedulePersistSemanticTabIndex();
});

chrome.tabs.onReplaced.addListener(async (addedTabId) => {
  try {
    const tab = await chrome.tabs.get(addedTabId);
    await indexSingleTab(tab);
  } catch {
    // ignore
  }
});

// ── Indexing ──────────────────────────────────────────────────────────────────

async function indexPage({ url, title, text, timestamp }) {
  if (!url) return;

  const recordingState = await chrome.storage.local.get(RECORDING_ENABLED_KEY);
  const recordingEnabled = recordingState[RECORDING_ENABLED_KEY] !== false;
  if (!recordingEnabled) {
    return;
  }

  // Skip raw search result pages to reduce retrieval noise.
  if (extractSearchQuery(url)) {
    return;
  }

  const eventTs = Number(timestamp || Date.now());
  const installMeta = await chrome.storage.local.get(INSTALL_TS_KEY);
  let installTs = installMeta[INSTALL_TS_KEY];

  // Existing installs may not have this key yet; initialize once instead of dropping captures.
  if (typeof installTs !== 'number' || Number.isNaN(installTs)) {
    installTs = eventTs;
    await chrome.storage.local.set({ [INSTALL_TS_KEY]: installTs });
  }

  if (eventTs < installTs) {
    return;
  }

  // Check blacklist first
  if (await isBlacklisted(url)) {
    console.log('[LocalMind] Skipping blacklisted page:', url);
    return;
  }

  const memoryId = `${url}::${eventTs}`;
  const trimmedText = text.slice(0, 1200);
  const tags = inferSemanticTagsFromText(`${title || ''} ${url || ''} ${trimmedText}`);

  console.log('[LocalMind] Indexing:', title);
  const embedding = await generateEmbedding(text, {
    taskType: 'RETRIEVAL_DOCUMENT',
    title: title || undefined,
  });

  await savePageToMemory({
    memoryId,
    url, title,
    text: trimmedText,
    embedding,
    tags,
    timestamp: eventTs,
    domain: new URL(url).hostname,
    sourceType: 'website',
    category: 'Visited Website',
  });

  await sendToBackendStore({
    url,
    title: title || url,
    content: text.slice(0, 4500),
    timestamp: eventTs,
  });
}

// ── Search ────────────────────────────────────────────────────────────────────

async function handleSearch(query, sourceFilter = 'all') {
  if (!query || query.trim().length < 2) return [];
  const parsed = parseNaturalLanguageQuery(query);
  const fallbackQueries = parsed.fallbackQueries?.length
    ? parsed.fallbackQueries
    : [parsed.retrievalQuery || query];

  if (DEBUG_QUERY_EXPANSION) {
    console.log('[LocalMind][QueryDebug] Parsed query', {
      original: query,
      retrievalQuery: parsed.retrievalQuery || query,
      fallbackQueries,
      focusTokens: parsed.focusTokens,
      expandedTokens: parsed.expandedTokens,
      inferredTags: parsed.inferredTags || [],
      inferredSubtags: parsed.inferredSubtags || [],
      allTagHints: parsed.allTagHints || parsed.inferredTags || [],
      exactPhrases: parsed.exactPhrases || [],
      entityPhrases: parsed.entityPhrases || [],
      mustIncludeTerms: parsed.mustIncludeTerms || [],
      domainHint: parsed.domainHint || null,
      timeHint: parsed.timeHint || null,
    });
  }

  let selectedStage = 'strict';
  let baseResults = [];

  for (const qv of fallbackQueries) {
    const queryEmbedding = await generateEmbedding(qv, { taskType: 'RETRIEVAL_QUERY' });
    baseResults = await searchMemory({
      queryEmbedding,
      queryText: qv,
      topK: 20,
      sourceFilter,
      relaxed: false,
      tagHints: parsed.allTagHints || parsed.inferredTags || [],
      exactPhrases: parsed.exactPhrases || [],
      mustIncludeTerms: parsed.mustIncludeTerms || [],
    });
    if (baseResults.length) {
      selectedStage = `strict:${qv}`;
      break;
    }
  }

  if (!baseResults.length) {
    for (const qv of fallbackQueries) {
      const queryEmbedding = await generateEmbedding(qv, { taskType: 'RETRIEVAL_QUERY' });
      baseResults = await searchMemory({
        queryEmbedding,
        queryText: qv,
        topK: 20,
        sourceFilter,
        relaxed: true,
        tagHints: parsed.allTagHints || parsed.inferredTags || [],
        exactPhrases: parsed.exactPhrases || [],
        mustIncludeTerms: parsed.mustIncludeTerms || [],
      });
      if (baseResults.length) {
        selectedStage = `relaxed:${qv}`;
        break;
      }
    }
  }

  const hinted = applyQueryHints(baseResults, parsed);
  const refined = await refineWithLiveSignals(hinted, parsed);

  if (DEBUG_QUERY_EXPANSION) {
    console.log('[LocalMind][QueryDebug] Retrieval stage selected', {
      stage: selectedStage,
      candidates: refined.length,
    });
  }

  return refined;
}

function tokenizeTerms(text) {
  return (text || '').toLowerCase().match(/[a-z0-9]{3,}/g) || [];
}

function overlapScore(aTokens, bTokens) {
  if (!aTokens.length || !bTokens.length) return 0;
  const a = new Set(aTokens);
  const b = new Set(bTokens);
  let overlap = 0;
  a.forEach((t) => {
    if (b.has(t)) overlap += 1;
  });
  return overlap / Math.max(1, a.size);
}

function applyQueryHints(results, parsed) {
  let out = results || [];
  if (!out.length) return out;

  if (parsed?.domainHint) {
    const narrowed = out.filter((r) => (r.domain || '').includes(parsed.domainHint));
    if (narrowed.length) {
      out = narrowed;
    }
  }

  if (parsed?.timeHint) {
    const now = new Date();
    const filtered = out.filter((r) => {
      const ts = Number(r.timestamp || 0);
      if (!ts) return false;
      const d = new Date(ts);
      if (parsed.timeHint === 'today') {
        return d.toDateString() === now.toDateString();
      }
      if (parsed.timeHint === 'yesterday') {
        const y = new Date(now);
        y.setDate(now.getDate() - 1);
        return d.toDateString() === y.toDateString();
      }
      if (parsed.timeHint === 'week') {
        return (now.getTime() - ts) <= (7 * 24 * 60 * 60 * 1000);
      }
      if (parsed.timeHint === 'month') {
        return (now.getTime() - ts) <= (31 * 24 * 60 * 60 * 1000);
      }
      if (parsed.timeHint === 'last_month') {
        const dMonth = d.getMonth();
        const dYear = d.getFullYear();
        const prev = new Date(now.getFullYear(), now.getMonth() - 1, 1);
        return dMonth === prev.getMonth() && dYear === prev.getFullYear();
      }
      if ((parsed.timeHint || '').startsWith('days_')) {
        const n = Number((parsed.timeHint || '').split('_')[1] || '0');
        if (n > 0) {
          return (now.getTime() - ts) <= (n * 24 * 60 * 60 * 1000);
        }
      }
      if (parsed.timeHint === 'recent') {
        return (now.getTime() - ts) <= (3 * 24 * 60 * 60 * 1000);
      }
      return true;
    });
    if (filtered.length) {
      out = filtered;
    }
  }

  if (parsed?.inferredTags?.length) {
    const narrowed = out.filter((r) => {
      const tags = (r.tags || []).map((t) => String(t || '').toLowerCase());
      const category = String(r.category || '').toLowerCase();
      return parsed.inferredTags.some((hint) => tags.includes(hint) || category.includes(hint));
    });
    if (narrowed.length) {
      out = narrowed;
    }
  }

  if (parsed?.exactPhrases?.length) {
    const narrowed = out.filter((r) => {
      const doc = `${r.title || ''} ${r.text || ''}`.toLowerCase();
      return parsed.exactPhrases.some((p) => doc.includes(String(p || '').toLowerCase()));
    });
    if (narrowed.length) {
      out = narrowed;
    }
  }

  return out;
}

function extractTitleAndText(html) {
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = (titleMatch?.[1] || '').replace(/\s+/g, ' ').trim();
  const text = stripHtmlToText(html || '').slice(0, 2200);
  return { title, text };
}

async function fetchLivePageSignal(url) {
  if (!url || extractSearchQuery(url)) {
    return { overlap: 0, fetched: false };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), LIVE_VERIFY_TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    if (!res.ok) return { overlap: 0, fetched: false };
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('text/html')) {
      return { overlap: 0, fetched: false };
    }
    const html = await res.text();
    const data = extractTitleAndText(html);
    return { ...data, fetched: true };
  } catch {
    return { overlap: 0, fetched: false };
  } finally {
    clearTimeout(timeout);
  }
}

async function refineWithLiveSignals(results, parsed) {
  if (!results?.length) return [];
  const qTokens = parsed?.focusTokens?.length
    ? parsed.focusTokens
    : tokenizeTerms(parsed?.retrievalQuery || parsed?.original || '');

  const head = results.slice(0, LIVE_VERIFY_TOP_N);
  const tail = results.slice(LIVE_VERIFY_TOP_N);

  const checked = await Promise.all(
    head.map(async (item) => {
      const live = await fetchLivePageSignal(item.url);
      if (!live.fetched) return item;

      const liveTokens = tokenizeTerms(`${live.title || ''} ${live.text || ''}`);
      const ov = overlapScore(qTokens, liveTokens);
      const boosted = { ...item };
      boosted.score = Number(item.score || 0) + (ov * 0.28) - (ov === 0 ? 0.08 : 0);
      boosted.live_overlap = Number(ov.toFixed(4));
      return boosted;
    })
  );

  const out = [...checked, ...tail].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));

  if (DEBUG_QUERY_EXPANSION) {
    console.log('[LocalMind][QueryDebug] Finalized with live verification', {
      original: parsed?.original,
      retrievalQuery: parsed?.retrievalQuery,
      returned: out.slice(0, 8).map((r) => ({
        title: r.title,
        score: r.score,
        live_overlap: r.live_overlap ?? null,
        domain: r.domain,
      })),
    });
  }

  return out;
}

function bookmarkIdToKey(bookmarkId) {
  return 'bookmark_' + bookmarkId;
}

function categorizeBookmark(bookmark, folderPath) {
  const name = `${bookmark.title || ''} ${folderPath || ''}`.toLowerCase();
  if (name.includes('work') || name.includes('project') || name.includes('docs')) return 'Work';
  if (name.includes('learn') || name.includes('course') || name.includes('tutorial')) return 'Learning';
  if (name.includes('news') || name.includes('blog')) return 'News';
  if (name.includes('shop') || name.includes('buy') || name.includes('deal')) return 'Shopping';
  return 'General';
}

function normalizeBookmarkText({ title, url, folderPath, category }) {
  return `Bookmark title: ${title || 'Untitled'}.
Bookmark url: ${url}
Bookmark folder: ${folderPath || 'Root'}
Bookmark category: ${category}`.replace(/\s+/g, ' ').trim();
}

function storageKeyForStableId(stableId) {
  return 'page_' + btoa(encodeURIComponent(stableId)).replace(/[^a-z0-9]/gi, '').slice(0, 40);
}

function stripHtmlToText(html) {
  if (!html) return '';
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/\s+/g, ' ')
    .trim();
}

async function fetchBookmarkPageText(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), BOOKMARK_FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) return '';
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('text/html')) return '';
    const html = await res.text();
    return stripHtmlToText(html).slice(0, 4000);
  } catch {
    return '';
  } finally {
    clearTimeout(timeout);
  }
}

async function sendToBackendStore({ url, title, content, timestamp }) {
  try {
    await fetch(BACKEND_STORE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, title, content, timestamp: Number(timestamp || Date.now()) }),
    });
  } catch (err) {
    console.warn('[LocalMind] Failed to sync memory to backend:', err);
  }
}

function collectBookmarksWithPath(nodes, parentPath = '') {
  const out = [];
  for (const node of nodes || []) {
    const thisPath = node.title ? `${parentPath}/${node.title}` : parentPath;
    if (node.url) out.push({ ...node, folderPath: thisPath || 'Root' });
    if (node.children?.length) out.push(...collectBookmarksWithPath(node.children, thisPath));
  }
  return out;
}

async function indexBookmark(bookmark, folderPath = 'Root') {
  if (!bookmark?.url) return;

  const recordingState = await chrome.storage.local.get(RECORDING_ENABLED_KEY);
  const recordingEnabled = recordingState[RECORDING_ENABLED_KEY] !== false;
  if (!recordingEnabled) {
    return;
  }

  // Check blacklist for bookmarks too
  if (await isBlacklisted(bookmark.url)) {
    console.log('[LocalMind] Skipping blacklisted bookmark:', bookmark.url);
    return;
  }

  let hostname = '';
  try {
    hostname = new URL(bookmark.url).hostname;
  } catch {
    return;
  }

  const category = categorizeBookmark(bookmark, folderPath);
  const memoryId = bookmarkIdToKey(bookmark.id);
  const storageKey = storageKeyForStableId(memoryId);
  const existing = await chrome.storage.local.get(storageKey);
  const existingRecord = existing[storageKey];
  const shouldRefreshRemoteContent =
    !existingRecord?.fetchedContent ||
    (Date.now() - (existingRecord?.contentFetchedAt || 0)) > BOOKMARK_RESYNC_INTERVAL_MS;

  let fetchedContent = existingRecord?.fetchedContent || '';
  if (shouldRefreshRemoteContent) {
    fetchedContent = await fetchBookmarkPageText(bookmark.url);
  }

  const baseText = normalizeBookmarkText({
    title: bookmark.title,
    url: bookmark.url,
    folderPath,
    category,
  });
  const text = fetchedContent
    ? `${baseText}\n\nPage content snapshot: ${fetchedContent}`.slice(0, 5000)
    : baseText;
  const tags = inferSemanticTagsFromText(`${bookmark.title || ''} ${bookmark.url || ''} ${folderPath || ''} ${text}`);

  const embedding = await generateEmbedding(text, {
    taskType: 'RETRIEVAL_DOCUMENT',
    title: bookmark.title || undefined,
  });

  await savePageToMemory({
    memoryId,
    url: bookmark.url,
    title: bookmark.title || bookmark.url,
    text,
    embedding,
    tags,
    timestamp: Date.now(),
    domain: hostname,
    sourceType: 'bookmark',
    category,
    bookmarkFolder: folderPath,
    fetchedContent,
    contentFetchedAt: Date.now(),
  });

  await sendToBackendStore({
    url: bookmark.url,
    title: `${bookmark.title || bookmark.url} [Bookmark]`,
    content: text,
  });
}

async function getBookmarkFolderPath(bookmark) {
  if (!bookmark?.parentId) return 'Root';
  const parts = [];
  let currentId = bookmark.parentId;
  while (currentId) {
    const nodes = await chrome.bookmarks.get(currentId);
    const node = nodes?.[0];
    if (!node) break;
    if (node.title) parts.unshift(node.title);
    if (!node.parentId || node.id === '0') break;
    currentId = node.parentId;
  }
  return parts.length ? `/${parts.join('/')}` : 'Root';
}

async function syncAllBookmarks() {
  try {
    const tree = await chrome.bookmarks.getTree();
    const bookmarks = collectBookmarksWithPath(tree);
    for (const bookmark of bookmarks) {
      await indexBookmark(bookmark, bookmark.folderPath || 'Root');
    }
  } catch (err) {
    console.warn('[LocalMind] Bookmark sync failed:', err);
  }
}

// ── Install bootstrap (post-install data only) ───────────────────────────────

chrome.runtime.onInstalled.addListener(async () => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

  const existingInstallMeta = await chrome.storage.local.get(INSTALL_TS_KEY);
  if (!existingInstallMeta[INSTALL_TS_KEY]) {
    await chrome.storage.local.set({ [INSTALL_TS_KEY]: Date.now() });
  }

  const recordingMeta = await chrome.storage.local.get(RECORDING_ENABLED_KEY);
  if (typeof recordingMeta[RECORDING_ENABLED_KEY] !== 'boolean') {
    await chrome.storage.local.set({ [RECORDING_ENABLED_KEY]: true });
  }

  const injectedSuggestionsMeta = await chrome.storage.local.get(SHOW_INJECTED_SUGGESTIONS_KEY);
  if (typeof injectedSuggestionsMeta[SHOW_INJECTED_SUGGESTIONS_KEY] !== 'boolean') {
    await chrome.storage.local.set({ [SHOW_INJECTED_SUGGESTIONS_KEY]: true });
  }

  await initializeSemanticTabIndexer();
});

chrome.bookmarks.onCreated.addListener(async (_id, bookmark) => {
  const folderPath = await getBookmarkFolderPath(bookmark);
  await indexBookmark(bookmark, folderPath);
});

chrome.bookmarks.onChanged.addListener(async (id, changeInfo) => {
  try {
    const [bookmark] = await chrome.bookmarks.get(id);
    if (!bookmark?.url) return;
    const folderPath = await getBookmarkFolderPath(bookmark);
    await indexBookmark({
      ...bookmark,
      title: changeInfo.title || bookmark.title,
      url: changeInfo.url || bookmark.url,
    }, folderPath);
  } catch (err) {
    console.warn('[LocalMind] Bookmark update sync failed:', err);
  }
});

chrome.runtime.onStartup.addListener(async () => {
  // Intentionally do not backfill historical bookmarks on startup.
  await initializeSemanticTabIndexer();
});

initializeSemanticTabIndexer().catch((err) => {
  console.warn('[LocalMind][SemanticTab] init failed', err);
});
import {
  archiveUpsert,
  archiveGetRecent,
  archiveSearchLexical,
  archiveDeleteByUrl,
  archiveDeleteById,
  archiveClearAll,
} from './archive-db.js';

const PAGE_PREFIX = 'page_';
// Hot cache for fast semantic scoring in extension; full history lives in archive DB.
const MAX_STORED_PAGES = 1200;

export async function savePageToMemory(record) {
  await archiveUpsert(record);
  const stableId = record.memoryId || record.url;
  const key = PAGE_PREFIX + btoa(encodeURIComponent(stableId)).replace(/[^a-z0-9]/gi, '').slice(0, 40);
  await chrome.storage.local.set({ [key]: record });
  await pruneIfNeeded();
}

export async function getRecentPages(limit = 20) {
  const all = await chrome.storage.local.get(null);
  const hot = Object.entries(all)
    .filter(([k]) => k.startsWith(PAGE_PREFIX))
    .map(([, v]) => v)
    .sort((a, b) => Number(b.timestamp || 0) - Number(a.timestamp || 0));

  const archived = await archiveGetRecent(Math.max(limit * 2, 60));
  const merged = [...hot, ...archived];
  const byId = new Map();
  for (const item of merged) {
    const id = item.memoryId || item.id || `${item.url || ''}::${item.timestamp || 0}`;
    if (!byId.has(id)) {
      byId.set(id, item);
    }
  }

  return Array.from(byId.values())
    .sort((a, b) => Number(b.timestamp || 0) - Number(a.timestamp || 0))
    .slice(0, limit);
}

export async function searchMemory({
  queryEmbedding,
  queryText,
  topK = 10,
  sourceFilter = 'all',
  relaxed = false,
  tagHints = [],
  exactPhrases = [],
  mustIncludeTerms = [],
}) {
  const all = await chrome.storage.local.get(null);
  const pages = Object.entries(all)
    .filter(([k]) => k.startsWith(PAGE_PREFIX))
    .map(([, v]) => v)
    .filter((p) => (p.title || p.text))
    .filter((p) => matchesSourceFilter(p, sourceFilter));

  const queryTokens = tokenizeTerms(queryText);
  const strictIntent = queryTokens.length >= 2 && !relaxed;
  const normalizedTagHints = (tagHints || []).map((t) => String(t || '').toLowerCase()).filter(Boolean);
  const normalizedExactPhrases = (exactPhrases || []).map((p) => String(p || '').toLowerCase().trim()).filter(Boolean);
  const normalizedMustTerms = (mustIncludeTerms || []).map((t) => String(t || '').toLowerCase().trim()).filter(Boolean);

  const withScores = pages.map(page => {
    let score = 0;
    let semantic = 0;

    if (page.embedding && queryEmbedding && page.embedding.length === queryEmbedding.length) {
      semantic = cosineSimilarity(page.embedding, queryEmbedding);
      score += semantic * 0.75;
    }

    const lq = queryText.toLowerCase();
    const docText = `${page.title || ''} ${page.text || ''}`.toLowerCase();
    if ((page.title || '').toLowerCase().includes(lq)) score += 0.3;
    if ((page.text || '').toLowerCase().includes(lq)) score += 0.15;

    const exactHits = normalizedExactPhrases.filter((p) => docText.includes(p)).length;
    if (exactHits > 0) {
      score += Math.min(0.5, exactHits * 0.28);
    }

    const overlap = keywordOverlap(queryTokens, tokenizeTerms(`${page.title || ''} ${page.text || ''}`));
    score += overlap * 0.2;

    const mustHits = normalizedMustTerms.filter((t) => docText.includes(t)).length;
    const mustRatio = normalizedMustTerms.length ? mustHits / normalizedMustTerms.length : 0;
    score += mustRatio * 0.16;

    const pageTags = (page.tags || []).map((t) => String(t || '').toLowerCase());
    const categoryTag = String(page.category || '').toLowerCase();
    const tagMatch = normalizedTagHints.length
      ? normalizedTagHints.some((hint) => pageTags.includes(hint) || categoryTag.includes(hint))
      : false;
    if (tagMatch) {
      score += 0.22;
    }

    const ageDays = (Date.now() - page.timestamp) / (1000 * 60 * 60 * 24);
    score += Math.max(0, (7 - ageDays) / 7) * 0.1;

    if (strictIntent && overlap <= 0 && semantic < 0.2 && !tagMatch && mustHits === 0) {
      score = -1;
    }

    if (strictIntent && normalizedExactPhrases.length && exactHits === 0) {
      score = -1;
    }

    return { ...page, score };
  });

  const minScore = strictIntent ? 0.2 : (relaxed ? 0.08 : 0.14);

  const hotResults = withScores
    .filter(p => p.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  if (hotResults.length >= topK) {
    return hotResults;
  }

  const archived = await archiveSearchLexical(
    queryText,
    topK * 2,
    sourceFilter,
    normalizedTagHints,
    normalizedExactPhrases,
    normalizedMustTerms,
  );
  const existing = new Set(hotResults.map((x) => x.memoryId || x.id || `${x.url || ''}::${x.timestamp || 0}`));
  const fill = archived.filter((x) => {
    const id = x.memoryId || x.id || `${x.url || ''}::${x.timestamp || 0}`;
    return !existing.has(id);
  });

  return [...hotResults, ...fill].slice(0, topK);
}

function tokenizeTerms(text) {
  return (text || '').toLowerCase().match(/[a-z0-9]{3,}/g) || [];
}

function keywordOverlap(a, b) {
  if (!a.length || !b.length) return 0;
  const sa = new Set(a);
  const sb = new Set(b);
  let overlap = 0;
  sa.forEach((token) => {
    if (sb.has(token)) overlap += 1;
  });
  return overlap / Math.max(sa.size, 1);
}

function matchesSourceFilter(page, sourceFilter) {
  if (!sourceFilter || sourceFilter === 'all') return true;
  if (sourceFilter === 'bookmarks') return page.sourceType === 'bookmark';
  if (sourceFilter === 'visited') return page.sourceType !== 'bookmark';
  return true;
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

async function pruneIfNeeded() {
  const all = await chrome.storage.local.get(null);
  const pageKeys = Object.keys(all)
    .filter(k => k.startsWith(PAGE_PREFIX))
    .map(k => ({ key: k, timestamp: all[k].timestamp || 0 }))
    .sort((a, b) => a.timestamp - b.timestamp);

  if (pageKeys.length > MAX_STORED_PAGES) {
    const toDelete = pageKeys.slice(0, pageKeys.length - MAX_STORED_PAGES).map(p => p.key);
    await chrome.storage.local.remove(toDelete);
  }
}


export async function deleteByUrlEverywhere(url) {
  if (!url) return;
  const all = await chrome.storage.local.get(null);
  const pageKeys = Object.keys(all).filter((k) => k.startsWith(PAGE_PREFIX) && all[k]?.url === url);
  if (pageKeys.length) {
    await chrome.storage.local.remove(pageKeys);
  }
  await archiveDeleteByUrl(url);
}


export async function deleteByMemoryIdEverywhere(memoryId) {
  if (!memoryId) return;
  const key = PAGE_PREFIX + btoa(encodeURIComponent(memoryId)).replace(/[^a-z0-9]/gi, '').slice(0, 40);
  await chrome.storage.local.remove(key);
  await archiveDeleteById(memoryId);
}


export async function clearAllStoredMemories() {
  const all = await chrome.storage.local.get(null);
  const pageKeys = Object.keys(all).filter((k) => k.startsWith(PAGE_PREFIX));
  if (pageKeys.length) {
    await chrome.storage.local.remove(pageKeys);
  }
  await archiveClearAll();
}


const BLACKLIST_KEY = 'lm_blacklist';

export async function getBlacklist() {
  const data = await chrome.storage.local.get(BLACKLIST_KEY);
  return data[BLACKLIST_KEY] || [];
}

export async function addToBlacklist(domain) {
  const list = await getBlacklist();
  if (!list.includes(domain)) {
    list.push(domain);
    await chrome.storage.local.set({ [BLACKLIST_KEY]: list });
  }
}

export async function removeFromBlacklist(domain) {
  const list = await getBlacklist();
  const updated = list.filter(d => d !== domain);
  await chrome.storage.local.set({ [BLACKLIST_KEY]: updated });
}

export async function isBlacklisted(url) {
  try {
    const domain = new URL(url).hostname.replace('www.', '');
    const list = await getBlacklist();
    return list.some(d => domain === d || domain.endsWith('.' + d));
  } catch {
    return false;
  }
}

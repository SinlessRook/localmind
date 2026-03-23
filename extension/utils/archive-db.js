const DB_NAME = 'localmind-archive';
const DB_VERSION = 1;
const STORE_VISITS = 'visits';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_VISITS)) {
        const store = db.createObjectStore(STORE_VISITS, { keyPath: 'id' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('url', 'url', { unique: false });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('Failed to open archive DB'));
  });
}

function slimRecord(record) {
  return {
    id: record.memoryId || record.id || `${record.url || 'unknown'}::${record.timestamp || Date.now()}`,
    memoryId: record.memoryId || record.id,
    url: record.url || '',
    title: record.title || record.url || 'Untitled',
    text: (record.text || '').slice(0, 1000),
    timestamp: Number(record.timestamp || Date.now()),
    sourceType: record.sourceType || 'website',
    category: record.category || '',
    tags: Array.isArray(record.tags) ? record.tags.slice(0, 8) : [],
    domain: record.domain || '',
  };
}

export async function archiveUpsert(record) {
  const db = await openDb();
  const payload = slimRecord(record);

  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_VISITS, 'readwrite');
    tx.objectStore(STORE_VISITS).put(payload);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('Archive upsert failed'));
  });
}

export async function archiveDeleteByUrl(url) {
  if (!url) return;
  const db = await openDb();

  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_VISITS, 'readwrite');
    const store = tx.objectStore(STORE_VISITS);
    const idx = store.index('url');
    const req = idx.openCursor(IDBKeyRange.only(url));

    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) return;
      store.delete(cursor.primaryKey);
      cursor.continue();
    };

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('Archive delete by url failed'));
  });
}

export async function archiveDeleteById(id) {
  if (!id) return;
  const db = await openDb();

  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_VISITS, 'readwrite');
    tx.objectStore(STORE_VISITS).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('Archive delete by id failed'));
  });
}

export async function archiveClearAll() {
  const db = await openDb();

  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_VISITS, 'readwrite');
    tx.objectStore(STORE_VISITS).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('Archive clear failed'));
  });
}

function matchesSourceFilter(page, sourceFilter) {
  if (!sourceFilter || sourceFilter === 'all') return true;
  if (sourceFilter === 'bookmarks') return page.sourceType === 'bookmark';
  if (sourceFilter === 'visited') return page.sourceType !== 'bookmark';
  return true;
}

function tokenize(text) {
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

export async function archiveGetRecent(limit = 50) {
  const db = await openDb();

  const rows = await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_VISITS, 'readonly');
    const store = tx.objectStore(STORE_VISITS);
    const idx = store.index('timestamp');
    const out = [];

    const req = idx.openCursor(null, 'prev');
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor || out.length >= limit) return;
      out.push(cursor.value);
      cursor.continue();
    };

    tx.oncomplete = () => resolve(out);
    tx.onerror = () => reject(tx.error || new Error('Archive recent read failed'));
  });

  return rows;
}

export async function archiveSearchLexical(
  queryText,
  topK = 10,
  sourceFilter = 'all',
  tagHints = [],
  exactPhrases = [],
  mustIncludeTerms = [],
) {
  const qTokens = tokenize(queryText);
  if (!qTokens.length) return [];
  const hints = (tagHints || []).map((t) => String(t || '').toLowerCase()).filter(Boolean);
  const phrases = (exactPhrases || []).map((p) => String(p || '').toLowerCase().trim()).filter(Boolean);
  const mustTerms = (mustIncludeTerms || []).map((t) => String(t || '').toLowerCase().trim()).filter(Boolean);

  const db = await openDb();

  const rows = await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_VISITS, 'readonly');
    const store = tx.objectStore(STORE_VISITS);
    const out = [];

    const req = store.openCursor();
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) return;
      const value = cursor.value;
      if (matchesSourceFilter(value, sourceFilter)) {
        out.push(value);
      }
      cursor.continue();
    };

    tx.oncomplete = () => resolve(out);
    tx.onerror = () => reject(tx.error || new Error('Archive search read failed'));
  });

  const scored = rows
    .map((item) => {
      const docText = `${item.title || ''} ${item.text || ''}`.toLowerCase();
      const docTokens = tokenize(docText);
      const overlap = overlapScore(qTokens, docTokens);
      const exact = docText.includes((queryText || '').toLowerCase()) ? 0.25 : 0;
      const phraseHits = phrases.filter((p) => docText.includes(p)).length;
      const mustHits = mustTerms.filter((t) => docText.includes(t)).length;
      const mustRatio = mustTerms.length ? mustHits / mustTerms.length : 0;
      const itemTags = (item.tags || []).map((t) => String(t || '').toLowerCase());
      const categoryTag = String(item.category || '').toLowerCase();
      const tagMatch = hints.length ? hints.some((h) => itemTags.includes(h) || categoryTag.includes(h)) : false;
      const ageDays = (Date.now() - Number(item.timestamp || 0)) / (1000 * 60 * 60 * 24);
      const recency = Math.max(0, (30 - ageDays) / 30) * 0.08;
      if (phrases.length && phraseHits === 0) {
        return { ...item, score: -1 };
      }
      return {
        ...item,
        score: overlap * 0.62 + exact + recency + (tagMatch ? 0.2 : 0) + Math.min(0.4, phraseHits * 0.25) + (mustRatio * 0.16),
      };
    })
    .filter((x) => x.score >= 0.16)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return scored;
}

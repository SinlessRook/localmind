// Injected into every page. Extracts meaningful text after user dwells.

const DWELL_TIME_MS = 1000;
const MAX_TEXT_LENGTH = 9000;
const MAX_CAPTURE_RETRIES = 3;

let dwellTimer = null;
let hasExtracted = false;
let captureAttempts = 0;

function extractPageContent() {
  if (hasExtracted) return;

  if (!document.body) {
    scheduleRetry();
    return;
  }

  const noiseTags = ['nav', 'footer', 'header', 'script', 'style', 'noscript', 'aside'];
  const clone = document.body.cloneNode(true);
  noiseTags.forEach(tag => clone.querySelectorAll(tag).forEach(el => el.remove()));

  const mainEl = clone.querySelector('article, main, [role="main"]') || clone;
  const headingNodes = Array.from(mainEl.querySelectorAll('h1, h2, h3')).slice(0, 18);
  const headings = headingNodes
    .map((node) => (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join('\n');

  const rawText = mainEl.innerText || mainEl.textContent || '';
  const bodyText = rawText.replace(/\s+/g, ' ').trim();
  const cleanText = bodyText.slice(0, MAX_TEXT_LENGTH);

  if (cleanText.length < 100) {
    scheduleRetry();
    return;
  }

  const structuredText = [`Title: ${document.title || ''}`, headings ? `Headings:\n${headings}` : '', `Body:\n${cleanText}`]
    .filter(Boolean)
    .join('\n\n');

  const payload = {
    type: 'PAGE_CONTENT',
    url: location.href,
    title: document.title,
    text: structuredText,
    timestamp: Date.now(),
  };

  sendToBackground(payload)
    .then(() => {
      hasExtracted = true;
    })
    .catch(() => {
      scheduleRetry();
    });
}

function scheduleRetry() {
  if (hasExtracted) return;
  if (captureAttempts >= MAX_CAPTURE_RETRIES) return;
  captureAttempts += 1;
  if (dwellTimer) clearTimeout(dwellTimer);
  dwellTimer = setTimeout(extractPageContent, DWELL_TIME_MS * 2);
}

function sendToBackground(payload) {
  return new Promise((resolve, reject) => {
    try {
      chrome.runtime.sendMessage(payload, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (response?.error) {
          reject(new Error(response.error));
          return;
        }
        resolve(response || {});
      });
    } catch (err) {
      reject(err);
    }
  });
}

function startDwellTracking() {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && dwellTimer) {
      clearTimeout(dwellTimer);
      dwellTimer = null;
    } else if (!document.hidden && !hasExtracted) {
      dwellTimer = setTimeout(extractPageContent, DWELL_TIME_MS);
    }
  });

  if (!document.hidden) {
    dwellTimer = setTimeout(extractPageContent, DWELL_TIME_MS);
  }
}

startDwellTracking();

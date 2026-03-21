// search-trigger.js — injected into search engine pages

(function () {
  if (document.getElementById('lm-iframe-wrap')) return;

  const query = extractQuery();
  if (!query) return;

  // ── Inject iframe ───────────────────────────────────────────────────────────

  const wrap = document.createElement('div');
  wrap.id = 'lm-iframe-wrap';

  const iframe = document.createElement('iframe');
  iframe.src = chrome.runtime.getURL(`injected-panel.html#${encodeURIComponent(query)}`);
  iframe.id = 'lm-iframe';
  iframe.allow = '';
  iframe.setAttribute('scrolling', 'no');

  const style = document.createElement('style');
  style.textContent = `
    #lm-iframe-wrap {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 999999;
      animation: lm-in 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    }

    #lm-iframe {
      width: 280px;
      height: 320px;
      border: none;
      border-radius: 14px;
      box-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(124,106,247,0.2);
      display: block;
    }

    @keyframes lm-in {
      from { opacity: 0; transform: translateY(20px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
  `;

  document.head.appendChild(style);
  wrap.appendChild(iframe);
  document.body.appendChild(wrap);

  // ── Listen for close / open-full messages from iframe ────────────────────

  window.addEventListener('message', (e) => {
    if (e.data?.type === 'LM_CLOSE') {
      wrap.style.animation = 'none';
      wrap.style.opacity = '0';
      wrap.style.transform = 'translateY(10px)';
      wrap.style.transition = 'all 0.2s ease';
      setTimeout(() => wrap.remove(), 200);
    }
  });

  function extractQuery() {
    try {
      const u = new URL(location.href);
      return u.searchParams.get('q') || u.searchParams.get('p') || null;
    } catch { return null; }
  }

})();
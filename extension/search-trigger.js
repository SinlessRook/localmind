// search-trigger.js — injected into search engine pages

(function () {
  const SHOW_INJECTED_SUGGESTIONS_KEY = 'lm_show_injected_suggestions';
  const PANEL_ID = 'lm-iframe-wrap';
  const IFRAME_ID = 'lm-iframe';

  let visibleEnabled = true;
  let currentQuery = '';
  let queryDebounce = null;

  ensurePanelStyle();

  chrome.storage.local.get(SHOW_INJECTED_SUGGESTIONS_KEY, (data) => {
    visibleEnabled = data?.[SHOW_INJECTED_SUGGESTIONS_KEY] !== false;
    refreshPanelVisibilityAndQuery();
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || !changes?.[SHOW_INJECTED_SUGGESTIONS_KEY]) return;
    visibleEnabled = changes[SHOW_INJECTED_SUGGESTIONS_KEY].newValue !== false;
    refreshPanelVisibilityAndQuery();
  });

  window.addEventListener('message', (e) => {
    if (e.data?.type === 'LM_CLOSE') {
      removePanel();
    }
  });

  bindSearchInputs();
  refreshPanelVisibilityAndQuery();

  function bindSearchInputs() {
    const selectors = [
      'input[name="q"]',
      'textarea[name="q"]',
      'input[name="p"]',
      'textarea[name="p"]',
      'input[type="search"]',
    ];
    const fields = Array.from(new Set(selectors.flatMap((s) => Array.from(document.querySelectorAll(s)))));

    fields.forEach((field) => {
      field.addEventListener('input', () => {
        if (queryDebounce) clearTimeout(queryDebounce);
        queryDebounce = setTimeout(() => {
          refreshPanelVisibilityAndQuery();
        }, 130);
      });
      field.addEventListener('change', () => refreshPanelVisibilityAndQuery());
      field.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          setTimeout(() => refreshPanelVisibilityAndQuery(), 120);
        }
      });
    });
  }

  function ensurePanelStyle() {
    if (document.getElementById('lm-iframe-style')) return;
    const style = document.createElement('style');
    style.id = 'lm-iframe-style';
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
  }

  function readLiveQueryFromInputs() {
    const byNameQ = document.querySelector('input[name="q"], textarea[name="q"]');
    const byNameP = document.querySelector('input[name="p"], textarea[name="p"]');
    const firstSearch = document.querySelector('input[type="search"]');
    const fromField = (byNameQ?.value || byNameP?.value || firstSearch?.value || '').trim();
    if (fromField) return fromField;
    return extractQueryFromUrl();
  }

  function refreshPanelVisibilityAndQuery() {
    const q = readLiveQueryFromInputs();
    if (!visibleEnabled || !q) {
      removePanel();
      return;
    }

    const created = ensurePanel(q);
    if (!created && q !== currentQuery) {
      postQueryToPanel(q);
    }
    currentQuery = q;
  }

  function ensurePanel(initialQuery) {
    let wrap = document.getElementById(PANEL_ID);
    if (wrap) return false;

    wrap = document.createElement('div');
    wrap.id = PANEL_ID;

    const iframe = document.createElement('iframe');
    iframe.src = chrome.runtime.getURL(`injected-panel.html#${encodeURIComponent(initialQuery)}`);
    iframe.id = IFRAME_ID;
    iframe.allow = '';
    iframe.setAttribute('scrolling', 'no');

    wrap.appendChild(iframe);
    document.body.appendChild(wrap);
    currentQuery = initialQuery;

    iframe.addEventListener('load', () => {
      postQueryToPanel(currentQuery);
    });

    return true;
  }

  function postQueryToPanel(query) {
    const iframe = document.getElementById(IFRAME_ID);
    if (!iframe?.contentWindow) return;
    iframe.contentWindow.postMessage({ type: 'LM_QUERY_UPDATE', query }, '*');
  }

  function removePanel() {
    const wrap = document.getElementById(PANEL_ID);
    if (!wrap) return;
    wrap.style.animation = 'none';
    wrap.style.opacity = '0';
    wrap.style.transform = 'translateY(10px)';
    wrap.style.transition = 'all 0.2s ease';
    setTimeout(() => {
      const live = document.getElementById(PANEL_ID);
      if (live) live.remove();
    }, 200);
  }

  function extractQueryFromUrl() {
    try {
      const u = new URL(location.href);
      return u.searchParams.get('q') || u.searchParams.get('p') || null;
    } catch { return null; }
  }

})();
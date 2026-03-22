async function readSyncedTheme() {
  try {
    if (typeof chrome !== 'undefined' && chrome.storage?.local) {
      const { lmTheme } = await chrome.storage.local.get('lmTheme');
      if (lmTheme === 'light' || lmTheme === 'dark') return lmTheme;
    }
  } catch (_) { /* ignore */ }
  const legacy =
    localStorage.getItem('lm-theme') ||
    localStorage.getItem('localmind-theme');
  if (legacy === 'light' || legacy === 'dark') return legacy;
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.textContent = theme === 'dark' ? 'Light' : 'Dark';
  }
}

async function persistTheme(next) {
  localStorage.setItem('lm-theme', next);
  localStorage.setItem('localmind-theme', next);
  try {
    if (typeof chrome !== 'undefined' && chrome.storage?.local) {
      await chrome.storage.local.set({ lmTheme: next });
      await chrome.storage.session.set({ lmTheme: next });
    }
  } catch (_) { /* ignore */ }
}

async function bootstrap() {
  const theme = await readSyncedTheme();
  applyTheme(theme);
  try {
    if (typeof chrome !== 'undefined' && chrome.storage?.local) {
      await chrome.storage.local.set({ lmTheme: theme });
    }
  } catch (_) { /* ignore */ }

  if (typeof chrome !== 'undefined' && chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area !== 'local' || !changes.lmTheme) return;
      const v = changes.lmTheme.newValue;
      if (v === 'light' || v === 'dark') {
        applyTheme(v);
        localStorage.setItem('lm-theme', v);
        localStorage.setItem('localmind-theme', v);
      }
    });
  }

  const nav = document.getElementById('nav');
  if (nav) {
    window.addEventListener('scroll', () => {
      nav.classList.toggle('scrolled', window.scrollY > 40);
    });
  }

  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', async () => {
      const current =
        document.documentElement.getAttribute('data-theme') === 'light'
          ? 'light'
          : 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      await persistTheme(next);
      applyTheme(next);
    });
  }

  const reveals = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
          setTimeout(() => entry.target.classList.add('visible'), i * 80);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 },
  );
  reveals.forEach((el) => observer.observe(el));

  document.querySelectorAll('.feature-item').forEach((item) => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.feature-item').forEach((i) => i.classList.remove('active'));
      item.classList.add('active');
    });
  });
}

bootstrap();

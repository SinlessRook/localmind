const FILLER_TERMS = new Set([
  'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'our', 'ours',
  'the', 'a', 'an', 'to', 'for', 'of', 'on', 'in', 'at', 'from', 'with',
  'about', 'that', 'this', 'these', 'those', 'it', 'is', 'are', 'was', 'were',
  'do', 'did', 'does', 'can', 'could', 'would', 'should', 'please', 'show',
  'find', 'search', 'look', 'tell', 'give', 'what', 'where', 'when', 'which',
  'read', 'saw', 'seen', 'page', 'pages', 'site', 'sites', 'something', 'anything',
]);

const DOMAIN_ALIASES = {
  youtube: 'youtube.com',
  wikipedia: 'wikipedia.org',
  reddit: 'reddit.com',
  github: 'github.com',
  stackoverflow: 'stackoverflow.com',
  stack: 'stackoverflow.com',
  medium: 'medium.com',
};

const MISSPELLINGS = {
  explotion: 'explosion',
  nulear: 'nuclear',
  chernobly: 'chernobyl',
  radation: 'radiation',
};

const SYNONYM_MAP = {
  nuclear: ['atomic', 'radiation', 'reactor', 'fallout'],
  explosion: ['blast', 'detonation', 'disaster', 'incident'],
  chernobyl: ['ukraine', 'reactor', 'radiation'],
  where: ['location', 'place', 'site'],
  place: ['location', 'site', 'area'],
};

const TAG_KEYWORDS = {
  entertainment: ['movie', 'film', 'music', 'song', 'netflix', 'youtube', 'series', 'anime', 'game', 'gaming', 'stream'],
  productivity: ['calendar', 'notion', 'docs', 'document', 'spreadsheet', 'meeting', 'todo', 'task', 'workflow', 'schedule'],
  learning: ['tutorial', 'course', 'learn', 'lesson', 'guide', 'documentation', 'study', 'concept'],
  coding: ['code', 'coding', 'programming', 'python', 'javascript', 'react', 'debug', 'github', 'stackoverflow', 'api'],
  shopping: ['buy', 'price', 'deal', 'cart', 'checkout', 'amazon', 'flipkart', 'product', 'review'],
  finance: ['bank', 'finance', 'stock', 'trading', 'investment', 'crypto', 'budget', 'loan', 'tax'],
  health: ['health', 'fitness', 'diet', 'workout', 'medicine', 'symptom', 'doctor', 'nutrition'],
  news: ['news', 'headline', 'breaking', 'update', 'report', 'politics', 'world'],
  research: ['paper', 'journal', 'arxiv', 'reference', 'citation', 'study', 'analysis'],
  social: ['reddit', 'twitter', 'x.com', 'instagram', 'facebook', 'post', 'thread', 'community'],
  travel: ['flight', 'hotel', 'trip', 'travel', 'booking', 'itinerary', 'destination'],
  security: ['security', 'vulnerability', 'exploit', 'bypass', 'auth', 'authentication', 'login', 'username', 'password', 'token', 'csrf', 'xss', 'sqli'],
};

const SUBTAG_RULES = [
  { terms: ['chess', 'lichess', 'chess.com'], tags: ['entertainment', 'entertainment:games', 'entertainment:games:chess'] },
  { terms: ['microservice', 'microservices', 'step up rule'], tags: ['coding', 'coding:architecture', 'coding:architecture:microservices'] },
  { terms: ['login bypass', 'auth bypass', 'username', 'authentication'], tags: ['security', 'security:auth', 'security:auth:login-bypass'] },
  { terms: ['palindrome', 'edge case', 'two pointer'], tags: ['coding', 'coding:algorithms', 'coding:algorithms:string'] },
];

const TAG_PHRASES = {
  entertainment: ['watch later', 'music video', 'movie review'],
  productivity: ['to do', 'project plan', 'meeting notes', 'daily plan'],
  learning: ['how to', 'step by step', 'getting started'],
  coding: ['code example', 'stack overflow', 'api docs'],
  shopping: ['best price', 'buy now', 'product comparison'],
  finance: ['stock market', 'mutual fund', 'credit card'],
  health: ['weight loss', 'mental health', 'home workout'],
  news: ['latest news', 'breaking news'],
  research: ['research paper', 'literature review'],
  social: ['social media', 'reddit thread'],
  travel: ['travel plan', 'hotel booking'],
};

const CORRECTION_VOCAB = Array.from(new Set([
  ...Object.keys(SYNONYM_MAP),
  ...Object.values(SYNONYM_MAP).flat(),
  ...Object.keys(DOMAIN_ALIASES),
  ...Object.values(DOMAIN_ALIASES).map((d) => d.split('.')[0]),
]));

function tokenize(text) {
  return (text || '').toLowerCase().match(/[a-z0-9.-]{3,}/g) || [];
}

function normalizeToken(token) {
  if (!token) return token;
  let t = token.toLowerCase();
  if (t.length > 5 && t.endsWith('ing')) t = t.slice(0, -3);
  else if (t.length > 4 && t.endsWith('ed')) t = t.slice(0, -2);
  else if (t.length > 4 && t.endsWith('es')) t = t.slice(0, -2);
  else if (t.length > 3 && t.endsWith('s')) t = t.slice(0, -1);
  return t;
}

function editDistance(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i += 1) dp[i][0] = i;
  for (let j = 0; j <= n; j += 1) dp[0][j] = j;
  for (let i = 1; i <= m; i += 1) {
    for (let j = 1; j <= n; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost,
      );
    }
  }
  return dp[m][n];
}

function correctToken(token) {
  if (!token || token.length < 5) return token;
  if (MISSPELLINGS[token]) return MISSPELLINGS[token];

  let best = token;
  let bestDist = 99;
  for (const candidate of CORRECTION_VOCAB) {
    const dist = editDistance(token, candidate);
    if (dist < bestDist) {
      best = candidate;
      bestDist = dist;
    }
  }

  return bestDist <= 2 ? best : token;
}

function dedupeStable(tokens) {
  const out = [];
  const seen = new Set();
  for (const t of tokens) {
    if (!t || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
}

function expandTokens(tokens) {
  const out = [...tokens];
  for (const t of tokens) {
    const syns = SYNONYM_MAP[t] || [];
    out.push(...syns);
  }
  return dedupeStable(out);
}

function inferTagsFromSignals(original, semanticTokens) {
  const lower = (original || '').toLowerCase();
  const normTokens = semanticTokens.map(normalizeToken);
  const tokenSet = new Set(normTokens);
  const tagScores = {};

  for (const [tag, words] of Object.entries(TAG_KEYWORDS)) {
    let score = 0;
    for (const w of words) {
      const nw = normalizeToken(w);
      if (tokenSet.has(nw)) score += 1;
      if (lower.includes(w)) score += 0.35;
    }
    for (const phrase of (TAG_PHRASES[tag] || [])) {
      if (lower.includes(phrase)) score += 1.4;
    }
    if (score > 0) tagScores[tag] = score;
  }

  if (/\b(youtube|netflix|spotify)\b/.test(lower)) {
    tagScores.entertainment = (tagScores.entertainment || 0) + 1.1;
  }
  if (/\b(github|stackoverflow|leetcode)\b/.test(lower)) {
    tagScores.coding = (tagScores.coding || 0) + 1.1;
  }
  if (/\b(amazon|flipkart|myntra)\b/.test(lower)) {
    tagScores.shopping = (tagScores.shopping || 0) + 1.1;
  }

  const top = Object.entries(tagScores)
    .sort((a, b) => b[1] - a[1])
    .filter(([, score]) => score >= 1.0)
    .slice(0, 4)
    .map(([tag]) => tag);

  const hierarchical = [];
  for (const rule of SUBTAG_RULES) {
    const hit = rule.terms.some((term) => lower.includes(term) || tokenSet.has(normalizeToken(term)));
    if (hit) {
      hierarchical.push(...rule.tags);
    }
  }

  return dedupeStable([...top, ...hierarchical]).slice(0, 10);
}

export function inferSemanticTagsFromText(text) {
  const tokens = tokenize(text).map(correctToken);
  const expanded = expandTokens(tokens);
  return inferTagsFromSignals(text, expanded);
}

function extractDomainHint(text) {
  const lower = (text || '').toLowerCase();

  const domainMatch = lower.match(/\b([a-z0-9-]+\.(com|org|net|io|dev|ai|in))\b/);
  if (domainMatch?.[1]) {
    return domainMatch[1];
  }

  for (const alias of Object.keys(DOMAIN_ALIASES)) {
    if (new RegExp(`\\b${alias}\\b`, 'i').test(lower)) {
      return DOMAIN_ALIASES[alias];
    }
  }

  return '';
}

function extractTimeHint(text) {
  const lower = (text || '').toLowerCase();
  if (lower.includes('today')) return 'today';
  if (lower.includes('yesterday')) return 'yesterday';
  if (lower.includes('this week') || lower.includes('last 7 days')) return 'week';
  if (lower.includes('this month')) return 'month';
  if (lower.includes('last month')) return 'last_month';
  if (lower.includes('recent') || lower.includes('lately')) return 'recent';
  const dayWindow = lower.match(/\blast\s+(\d{1,2})\s+days\b/);
  if (dayWindow?.[1]) {
    const n = Number(dayWindow[1]);
    if (n >= 1 && n <= 60) return `days_${n}`;
  }
  return '';
}

function extractQuotedPhrases(text) {
  const out = [];
  const re = /['"`]([^'"`]{2,64})['"`]/g;
  let m;
  while ((m = re.exec(text || '')) !== null) {
    const phrase = (m[1] || '').trim().toLowerCase();
    if (phrase) out.push(phrase);
  }
  return dedupeStable(out);
}

function extractEntityPhrases(text) {
  const lower = (text || '').toLowerCase();
  const entities = [];

  const usernameMatch = lower.match(/(?:username|user\s*name|handle|user\s*id)\s*(?:is|was|=|:)?\s*['"`]?([a-z0-9_.@-]{3,})['"`]?/i);
  if (usernameMatch?.[1]) {
    entities.push(usernameMatch[1]);
  }

  const titleCase = (text || '').match(/\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){1,4})\b/g) || [];
  for (const p of titleCase) {
    entities.push((p || '').trim().toLowerCase());
  }

  return dedupeStable(entities.filter((x) => x && x.length >= 3));
}

export function parseNaturalLanguageQuery(rawQuery) {
  const original = (rawQuery || '').trim();
  const tokens = tokenize(original);
  const exactPhrases = extractQuotedPhrases(original);
  const entityPhrases = extractEntityPhrases(original);

  const focusTokens = tokens.filter((t) => {
    if (FILLER_TERMS.has(t)) return false;
    if (/^[0-9]+$/.test(t)) return false;
    return true;
  });

  const uniqueFocus = dedupeStable(focusTokens);
  const correctedTokens = uniqueFocus.map(correctToken);
  const expandedTokens = expandTokens(correctedTokens);
  const inferredTags = inferTagsFromSignals(original, expandedTokens);

  const strictQuery = correctedTokens.length
    ? correctedTokens.slice(0, 10).join(' ')
    : tokens.slice(0, 10).join(' ');
  const expandedQuery = expandedTokens.length
    ? expandedTokens.slice(0, 16).join(' ')
    : strictQuery;
  const relaxedQuery = correctedTokens.length
    ? correctedTokens.slice(0, 3).join(' ')
    : strictQuery;

  const fallbackQueries = dedupeStable([expandedQuery, strictQuery, relaxedQuery, original].filter(Boolean));
  const mustIncludeTerms = dedupeStable([
    ...exactPhrases,
    ...entityPhrases,
    ...correctedTokens.filter((t) => t.length >= 4).slice(0, 3),
  ]);

  return {
    original,
    retrievalQuery: expandedQuery || strictQuery || original,
    focusTokens: correctedTokens,
    expandedTokens,
    inferredTags,
    inferredSubtags: inferredTags.filter((t) => t.includes(':')),
    allTagHints: inferredTags,
    exactPhrases,
    entityPhrases,
    mustIncludeTerms,
    fallbackQueries,
    domainHint: extractDomainHint(original),
    timeHint: extractTimeHint(original),
  };
}

from fastapi import FastAPI
from pydantic import BaseModel
from embed import get_embedding, get_embeddings
from memory import add_memories, search_memory, get_all_memories, get_cluster_count, clear_all_memories
import requests
import json
import re
import random
import time
import os
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse, quote
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
POLLINATIONS_TEXT_URL = "https://text.pollinations.ai"
POLLINATIONS_COOLDOWN_SECONDS = 300
POLLINATIONS_COOLDOWN_UNTIL = 0.0
MATCH_LIMIT = 3
CHUNK_MIN_TOKENS = 300
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_RATIO = 0.15
MAX_BODY_FOR_INDEXING_CHARS = 9000
MAX_SECTION_TEXT = 1400

QUERY_EXPANSIONS = {
    "nuclear": ["radiation", "atomic", "reactor", "fission", "chernobyl", "disaster"],
    "ai": ["artificial intelligence", "machine learning", "llm", "neural network"],
    "python": ["programming", "code", "library", "package", "debug"],
    "cars": ["engine", "automobile", "vehicle", "torque", "fuel"],
}
DEBUG_QUERY_EXPANSION = True

NL_FILLER_TERMS = {
    "i", "me", "my", "mine", "you", "your", "yours", "we", "our", "ours",
    "the", "a", "an", "to", "for", "of", "on", "in", "at", "from", "with",
    "about", "that", "this", "these", "those", "it", "is", "are", "was", "were",
    "do", "did", "does", "can", "could", "would", "should", "please", "show",
    "find", "search", "look", "tell", "give", "what", "where", "when", "which",
    "read", "saw", "seen", "page", "pages", "site", "sites", "something", "anything",
}

DOMAIN_ALIASES = {
    "youtube": "youtube.com",
    "wikipedia": "wikipedia.org",
    "reddit": "reddit.com",
    "github": "github.com",
    "stackoverflow": "stackoverflow.com",
    "stack": "stackoverflow.com",
    "medium": "medium.com",
}

MISSPELLINGS = {
    "explotion": "explosion",
    "nulear": "nuclear",
    "chernobly": "chernobyl",
    "radation": "radiation",
}

SYNONYM_MAP = {
    "nuclear": ["atomic", "radiation", "reactor", "fallout"],
    "explosion": ["blast", "detonation", "disaster", "incident"],
    "chernobyl": ["ukraine", "reactor", "radiation"],
    "where": ["location", "place", "site"],
    "place": ["location", "site", "area"],
}

TAG_KEYWORDS = {
    "entertainment": ["movie", "film", "music", "song", "netflix", "youtube", "series", "anime", "game", "gaming", "stream"],
    "productivity": ["calendar", "notion", "docs", "document", "spreadsheet", "meeting", "todo", "task", "workflow", "schedule"],
    "learning": ["tutorial", "course", "learn", "lesson", "guide", "documentation", "study", "concept"],
    "coding": ["code", "coding", "programming", "python", "javascript", "react", "debug", "github", "stackoverflow", "api"],
    "shopping": ["buy", "price", "deal", "cart", "checkout", "amazon", "flipkart", "product", "review"],
    "finance": ["bank", "finance", "stock", "trading", "investment", "crypto", "budget", "loan", "tax"],
    "health": ["health", "fitness", "diet", "workout", "medicine", "symptom", "doctor", "nutrition"],
    "news": ["news", "headline", "breaking", "update", "report", "politics", "world"],
    "research": ["paper", "journal", "arxiv", "reference", "citation", "study", "analysis"],
    "social": ["reddit", "twitter", "x.com", "instagram", "facebook", "post", "thread", "community"],
    "travel": ["flight", "hotel", "trip", "travel", "booking", "itinerary", "destination"],
    "security": ["security", "vulnerability", "exploit", "bypass", "auth", "authentication", "login", "username", "password", "token", "csrf", "xss", "sqli"],
}

SUBTAG_RULES = [
    {
        "terms": ["chess", "lichess", "chess.com"],
        "tags": ["entertainment", "entertainment:games", "entertainment:games:chess"],
    },
    {
        "terms": ["microservice", "microservices", "step up rule"],
        "tags": ["coding", "coding:architecture", "coding:architecture:microservices"],
    },
    {
        "terms": ["login bypass", "auth bypass", "username", "authentication"],
        "tags": ["security", "security:auth", "security:auth:login-bypass"],
    },
    {
        "terms": ["palindrome", "edge case", "two pointer"],
        "tags": ["coding", "coding:algorithms", "coding:algorithms:string"],
    },
]

QUERY_KNOWLEDGE_FILE = os.getenv(
    "LOCALMIND_QUERY_KNOWLEDGE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "query_knowledge.json"),
)


def _build_correction_vocab() -> list:
    return sorted(set(
        list(SYNONYM_MAP.keys())
        + [v for values in SYNONYM_MAP.values() for v in values]
        + list(DOMAIN_ALIASES.keys())
        + [v.split(".")[0] for v in DOMAIN_ALIASES.values()]
    ))


CORRECTION_VOCAB = _build_correction_vocab()


def _merge_map_list(base: dict, extra: dict) -> dict:
    if not isinstance(extra, dict):
        return base
    for k, values in extra.items():
        key = str(k or "").strip().lower()
        if not key:
            continue
        incoming = []
        if isinstance(values, list):
            incoming = [str(v).strip().lower() for v in values if str(v).strip()]
        elif isinstance(values, str):
            incoming = [values.strip().lower()] if values.strip() else []
        existing = [str(v).strip().lower() for v in base.get(key, []) if str(v).strip()]
        seen = set(existing)
        for item in incoming:
            if item not in seen:
                existing.append(item)
                seen.add(item)
        if existing:
            base[key] = existing
    return base


def _merge_map_scalar(base: dict, extra: dict) -> dict:
    if not isinstance(extra, dict):
        return base
    for k, v in extra.items():
        key = str(k or "").strip().lower()
        value = str(v or "").strip().lower()
        if key and value:
            base[key] = value
    return base


def _merge_subtag_rules(base: list, extra: list) -> list:
    if not isinstance(extra, list):
        return base
    out = list(base)
    for rule in extra:
        if not isinstance(rule, dict):
            continue
        terms = [str(t).strip().lower() for t in (rule.get("terms") or []) if str(t).strip()]
        tags = [str(t).strip().lower() for t in (rule.get("tags") or []) if str(t).strip()]
        if not terms or not tags:
            continue
        out.append({"terms": terms, "tags": tags})
    return out


def _load_query_knowledge_pack() -> dict:
    if not os.path.exists(QUERY_KNOWLEDGE_FILE):
        return {}
    try:
        with open(QUERY_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print("[QueryKnowledge] load failed:", e)
        return {}


def _apply_query_knowledge_pack() -> None:
    global QUERY_EXPANSIONS, DOMAIN_ALIASES, MISSPELLINGS, SYNONYM_MAP, TAG_KEYWORDS, SUBTAG_RULES, NL_FILLER_TERMS, CORRECTION_VOCAB
    data = _load_query_knowledge_pack()
    if not data:
        return

    QUERY_EXPANSIONS = _merge_map_list(QUERY_EXPANSIONS, data.get("query_expansions") or {})
    DOMAIN_ALIASES = _merge_map_scalar(DOMAIN_ALIASES, data.get("domain_aliases") or {})
    MISSPELLINGS = _merge_map_scalar(MISSPELLINGS, data.get("misspellings") or {})
    SYNONYM_MAP = _merge_map_list(SYNONYM_MAP, data.get("synonym_map") or {})
    TAG_KEYWORDS = _merge_map_list(TAG_KEYWORDS, data.get("tag_keywords") or {})
    SUBTAG_RULES = _merge_subtag_rules(SUBTAG_RULES, data.get("subtag_rules") or [])

    filler = data.get("filler_terms") or []
    if isinstance(filler, list):
        NL_FILLER_TERMS = set(list(NL_FILLER_TERMS) + [str(x).strip().lower() for x in filler if str(x).strip()])

    CORRECTION_VOCAB = _build_correction_vocab()


_apply_query_knowledge_pack()

IRRELEVANT_PATTERNS = [
    r"\bcookie(s)?\b",
    r"\bprivacy policy\b",
    r"\bterms of service\b",
    r"\ball rights reserved\b",
    r"\bsubscribe\b",
    r"\bsign\s?in\b",
    r"\blog\s?in\b",
    r"\bsign\s?up\b",
    r"\bcreate account\b",
    r"\bmenu\b",
    r"\bnavigation\b",
    r"\bskip to content\b",
]

# -------- Request Models --------
class Page(BaseModel):
    url: str
    title: str
    content: str
    timestamp: Optional[float] = None

class Query(BaseModel):
    query: str

class EmbedRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    message: str
    provider: str = "ollama"


STOP_WORDS = {
    "this", "that", "with", "from", "have", "were", "will", "your", "about", "which",
    "when", "where", "what", "into", "than", "then", "them", "they", "been", "also",
    "more", "some", "such", "only", "over", "very", "into", "onto", "just", "page",
    "home", "news", "today", "latest", "search", "google", "wikipedia", "chess", "world",
}


MIKE_OPENERS = [
    "Quick read from your history:",
    "Here is the cleanest play:",
    "Straight answer, no fluff:",
    "You are looking at this pattern:",
]

MIKE_MEMORY_FLEX = [
    "Photographic memory check complete.",
    "Memory scan locked in.",
    "Pattern match confirmed from your browsing trail.",
    "Timeline and content alignment confirmed.",
]

MIKE_FAST_LINES = [
    "Fast lane answer:",
    "No need to overthink this:",
    "Quick pull from your trail:",
    "Clean read, right away:",
]


def _parse_event_datetime(item: dict):
    visited_at = item.get("visited_at")
    if visited_at:
        try:
            return datetime.fromisoformat(str(visited_at).replace("Z", "+00:00"))
        except Exception:
            pass

    ts = item.get("timestamp")
    if ts is not None:
        try:
            tsf = float(ts)
            if tsf > 1e12:
                tsf /= 1000.0
            return datetime.fromtimestamp(tsf)
        except Exception:
            pass

    visited_date = item.get("visited_date")
    if visited_date:
        try:
            return datetime.fromisoformat(str(visited_date))
        except Exception:
            pass

    return None


def _event_sort_ts(item: dict) -> float:
    dt = _parse_event_datetime(item)
    if dt is not None:
        try:
            return float(dt.timestamp())
        except Exception:
            pass
    ts = item.get("timestamp")
    try:
        tsf = float(ts)
        if tsf > 1e12:
            tsf /= 1000.0
        return tsf
    except Exception:
        return 0.0


def _extract_domain(url: str) -> str:
    if not url:
        return "unknown"
    try:
        parsed = urlparse(url)
        return parsed.netloc or "unknown"
    except Exception:
        return "unknown"


def _extract_keywords(text: str) -> list:
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    return [w for w in words if w not in STOP_WORDS]


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _safe_timestamp(value) -> float:
    try:
        tsf = float(value)
        if tsf > 1e12:
            tsf /= 1000.0
        return tsf
    except Exception:
        return float(datetime.now().timestamp())


def _split_body_sections(text: str):
    cleaned = _clean_text(text)[:MAX_BODY_FOR_INDEXING_CHARS]
    if not cleaned:
        return [], ""

    lines = [ln.strip() for ln in re.split(r"\n+", text or "") if ln.strip()]
    headings = []
    body_parts = []
    for line in lines:
        compact = _clean_text(line)
        if not compact:
            continue
        is_heading = (
            len(compact) <= 110
            and (_token_count(compact) <= 14)
            and (compact.endswith(":") or compact.istitle() or compact.isupper())
        )
        if is_heading:
            headings.append(compact)
        else:
            body_parts.append(compact)

    body_text = _clean_text(" ".join(body_parts))
    if not body_text:
        body_text = cleaned
    return headings[:20], body_text


def _chunk_by_tokens(text: str, min_tokens: int = CHUNK_MIN_TOKENS, max_tokens: int = CHUNK_MAX_TOKENS, overlap_ratio: float = CHUNK_OVERLAP_RATIO) -> list:
    words = re.findall(r"\S+", text or "")
    if not words:
        return []

    if len(words) <= max_tokens:
        return [" ".join(words)]

    overlap = max(1, int(max_tokens * overlap_ratio))
    step = max(min_tokens, max_tokens - overlap)

    out = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_tokens]
        if not chunk_words:
            break
        out.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break
        i += step
    return out


def _expand_query(query: str) -> str:
    cleaned = _clean_text(query).lower()
    if not cleaned:
        return ""
    terms = re.findall(r"[a-zA-Z]{2,}", cleaned)
    expanded = list(terms)
    for t in terms:
        expanded.extend(QUERY_EXPANSIONS.get(t, []))
    seen = set()
    dedup = []
    for t in expanded:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return " ".join(dedup[:24])


def _extract_domain_hint(text: str) -> str:
    lower = (text or "").lower()
    direct = re.search(r"\b([a-z0-9-]+\.(com|org|net|io|dev|ai|in))\b", lower)
    if direct:
        return direct.group(1)
    for alias, domain in DOMAIN_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return domain
    return ""


def _edit_distance(a: str, b: str) -> int:
    m = len(a)
    n = len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _correct_token(token: str) -> str:
    if not token or len(token) < 5:
        return token
    if token in MISSPELLINGS:
        return MISSPELLINGS[token]

    best = token
    best_dist = 99
    for candidate in CORRECTION_VOCAB:
        dist = _edit_distance(token, candidate)
        if dist < best_dist:
            best = candidate
            best_dist = dist

    return best if best_dist <= 2 else token


def _dedupe_stable(tokens: list) -> list:
    out = []
    seen = set()
    for t in tokens:
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _expand_tokens(tokens: list) -> list:
    out = list(tokens)
    for t in tokens:
        out.extend(SYNONYM_MAP.get(t, []))
    return _dedupe_stable(out)


def _normalize_token(token: str) -> str:
    t = (token or "").lower()
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t


def _infer_tags_from_signals(original_text: str, semantic_tokens: list) -> list:
    lower = (original_text or "").lower()
    token_set = {_normalize_token(t) for t in (semantic_tokens or []) if t}
    scored = {}

    for tag, words in TAG_KEYWORDS.items():
        score = 0.0
        for w in words:
            nw = _normalize_token(w)
            if nw in token_set:
                score += 1.0
            if w in lower:
                score += 0.35
        if score > 0:
            scored[tag] = score

    if re.search(r"\b(youtube|netflix|spotify)\b", lower):
        scored["entertainment"] = scored.get("entertainment", 0.0) + 1.1
    if re.search(r"\b(github|stackoverflow|leetcode)\b", lower):
        scored["coding"] = scored.get("coding", 0.0) + 1.1
    if re.search(r"\b(amazon|flipkart|myntra)\b", lower):
        scored["shopping"] = scored.get("shopping", 0.0) + 1.1

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    top = [tag for tag, score in ranked if score >= 1.0][:4]

    hierarchical = []
    for rule in SUBTAG_RULES:
        terms = rule.get("terms") or []
        tags = rule.get("tags") or []
        hit = any((term in lower) or (_normalize_token(term) in token_set) for term in terms)
        if hit:
            hierarchical.extend(tags)

    return _dedupe_stable(top + hierarchical)[:10]


def _extract_quoted_phrases(text: str) -> list:
    out = []
    for m in re.finditer(r"['\"`]([^'\"`]{2,64})['\"`]", text or ""):
        phrase = (m.group(1) or "").strip().lower()
        if phrase:
            out.append(phrase)
    return _dedupe_stable(out)


def _extract_entity_phrases(text: str) -> list:
    lower = (text or "").lower()
    out = []

    username_match = re.search(
        r"(?:username|user\s*name|handle|user\s*id)\s*(?:is|was|=|:)?\s*['\"`]?([a-z0-9_.@-]{3,})['\"`]?",
        lower,
        flags=re.IGNORECASE,
    )
    if username_match and username_match.group(1):
        out.append(username_match.group(1))

    for phrase in re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){1,4})\b", text or ""):
        s = (phrase or "").strip().lower()
        if s:
            out.append(s)

    return _dedupe_stable([x for x in out if len(x) >= 3])


def _extract_time_intent_from_text(text: str) -> str:
    m = (text or "").lower()
    if "today" in m:
        return "today"
    if "yesterday" in m:
        return "yesterday"
    if "this week" in m or "last 7 days" in m:
        return "week"
    if "this month" in m:
        return "month"
    if "last month" in m:
        return "last_month"
    if "recent" in m or "lately" in m:
        return "recent"

    day_window = re.search(r"\blast\s+(\d{1,2})\s+days\b", m)
    if day_window:
        try:
            n = int(day_window.group(1))
            if 1 <= n <= 60:
                return f"days_{n}"
        except Exception:
            pass
    return ""


def _parse_natural_language_query(raw_query: str) -> dict:
    cleaned = _clean_text(raw_query)
    tokens = re.findall(r"[a-z0-9.-]{3,}", cleaned.lower())
    exact_phrases = _extract_quoted_phrases(raw_query or cleaned)
    entity_phrases = _extract_entity_phrases(raw_query or cleaned)
    focus = [t for t in tokens if t not in NL_FILLER_TERMS and not t.isdigit()]
    dedup_focus = _dedupe_stable(focus)
    corrected = [_correct_token(t) for t in dedup_focus]
    expanded_tokens = _expand_tokens(corrected)
    inferred_tags = _infer_tags_from_signals(cleaned, expanded_tokens)

    compact = " ".join(corrected[:10]) if corrected else " ".join(tokens[:10])
    expanded = " ".join(expanded_tokens[:16]) if expanded_tokens else compact
    expanded = _expand_query(expanded or compact or cleaned)
    relaxed = " ".join(corrected[:3]) if corrected else compact
    fallback_queries = _dedupe_stable([expanded, compact, relaxed, cleaned])
    must_include_terms = _dedupe_stable(exact_phrases + entity_phrases + [t for t in corrected[:5] if len(t) >= 4])

    return {
        "original": cleaned,
        "focus_terms": corrected,
        "expanded_terms": expanded_tokens,
        "inferred_tags": inferred_tags,
        "inferred_subtags": [t for t in inferred_tags if ":" in t],
        "all_tag_hints": inferred_tags,
        "exact_phrases": exact_phrases,
        "entity_phrases": entity_phrases,
        "must_include_terms": must_include_terms,
        "retrieval_query": expanded or compact or cleaned,
        "fallback_queries": fallback_queries,
        "domain_hint": _extract_domain_hint(cleaned),
        "time_hint": _extract_time_intent_from_text(cleaned),
    }


def _apply_domain_hint(items: list, domain_hint: str) -> list:
    if not domain_hint:
        return items
    narrowed = [i for i in items if domain_hint in (_extract_domain(i.get("url", "")) or "")]
    return narrowed if narrowed else items


def _build_structured_chunks(url: str, title: str, content: str, timestamp=None) -> list:
    title_clean = _clean_text(title)[:220] or "Untitled"
    domain = _extract_domain(url)
    ts = _safe_timestamp(timestamp)
    seed_tokens = re.findall(r"[a-z0-9.-]{3,}", f"{title_clean} {content[:1200]}".lower())
    tags = _infer_tags_from_signals(f"{title_clean} {url} {content[:2000]}", _expand_tokens(seed_tokens))

    headings, body = _split_body_sections(content)
    condensed = condense_content(title_clean, body, max_chars=MAX_BODY_FOR_INDEXING_CHARS)
    if not condensed:
        return []

    chunks = []
    chunks.append({
        "id": f"{int(ts)}-title-{abs(hash(url + title_clean)) % 1000000}",
        "content": f"Title: {title_clean}",
        "title": title_clean,
        "url": url,
        "keywords": _extract_keywords(title_clean),
        "timestamp": ts,
        "domain": domain,
        "section": "title",
        "tags": tags,
    })

    if headings:
        heading_text = _clean_text(" | ".join(headings))[:MAX_SECTION_TEXT]
        if heading_text:
            chunks.append({
                "id": f"{int(ts)}-head-{abs(hash(url + heading_text)) % 1000000}",
                "content": heading_text,
                "title": title_clean,
                "url": url,
                "keywords": _extract_keywords(heading_text),
                "timestamp": ts,
                "domain": domain,
                "section": "headings",
                "tags": tags,
            })

    body_chunks = _chunk_by_tokens(condensed)
    for idx, chunk in enumerate(body_chunks):
        chunk_text = _clean_text(chunk)
        if len(chunk_text) < 120:
            continue
        chunks.append({
            "id": f"{int(ts)}-body-{idx}-{abs(hash(url + str(idx))) % 1000000}",
            "content": chunk_text,
            "title": title_clean,
            "url": url,
            "keywords": _extract_keywords(f"{title_clean} {chunk_text}"),
            "timestamp": ts,
            "domain": domain,
            "section": "body",
            "tags": tags,
        })

    return chunks[:18]


def _clean_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _is_irrelevant_sentence(sentence: str) -> bool:
    s = sentence.lower().strip()
    if len(s) < 35:
        return True
    if sum(ch.isdigit() for ch in s) > len(s) * 0.35:
        return True
    for pattern in IRRELEVANT_PATTERNS:
        if re.search(pattern, s):
            return True
    return False


def _keyword_set(text: str):
    words = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "you", "your",
        "are", "was", "were", "have", "has", "had", "into", "about", "they",
        "them", "their", "will", "would", "there", "what", "when", "where",
    }
    return {w for w in words if w not in stop}


def condense_content(title: str, content: str, max_chars: int = 1400) -> str:
    cleaned = _clean_text(content)
    if not cleaned:
        return ""

    sentences = _split_sentences(cleaned)
    if not sentences:
        return cleaned[:max_chars]

    title_terms = _keyword_set(title)
    doc_terms = _keyword_set(cleaned)

    scored = []
    for i, sentence in enumerate(sentences):
        if _is_irrelevant_sentence(sentence):
            continue

        sent_terms = _keyword_set(sentence)
        if not sent_terms:
            continue

        title_overlap = len(sent_terms & title_terms)
        doc_overlap = len(sent_terms & doc_terms)
        length_bonus = min(len(sentence), 220) / 220.0
        score = (title_overlap * 2.0) + (doc_overlap * 0.25) + length_bonus
        scored.append((score, i, sentence))

    if not scored:
        fallback = [s for s in sentences if len(s) >= 35][:8]
        text = " ".join(fallback) if fallback else cleaned
        return text[:max_chars]

    # Keep highest-signal sentences, then restore original order for readability.
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:10]
    top_sorted = sorted(top, key=lambda x: x[1])

    out = []
    total = 0
    for _, _, sentence in top_sorted:
        if total + len(sentence) + 1 > max_chars:
            break
        out.append(sentence)
        total += len(sentence) + 1

    if not out:
        return top_sorted[0][2][:max_chars]
    return " ".join(out)


def rerank_results(query: str, candidates: list, top_k: int = 3, min_score: float = 0.10) -> list:
    query_terms = _keyword_set(query)
    if not candidates:
        return []

    scored = []
    for item in candidates:
        title = item.get("title", "")
        content = item.get("content", "")
        semantic = float(item.get("semantic_score", 0.0))

        title_terms = _keyword_set(title)
        content_terms = _keyword_set(content)

        # Prefer items whose title/content share intent words with the query.
        title_overlap = len(query_terms & title_terms)
        content_overlap = len(query_terms & content_terms)
        has_exact_phrase = 1.0 if query.lower() in f"{title} {content}".lower() else 0.0

        lexical_score = (title_overlap * 0.5) + (content_overlap * 0.2) + has_exact_phrase
        final_score = (semantic * 0.7) + (lexical_score * 0.3)

        enriched = dict(item)
        enriched["score"] = round(final_score, 4)
        scored.append(enriched)

    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return [r for r in scored[:top_k] if r.get("score", 0.0) >= min_score]


def _intent_strength(query: str, item: dict) -> float:
    q_terms = _keyword_set(query)
    if not q_terms:
        return 0.0
    doc_terms = _keyword_set(f"{item.get('title', '')} {item.get('content', '')}")
    if not doc_terms:
        return 0.0
    return float(len(q_terms & doc_terms)) / max(1.0, float(len(q_terms)))


def _apply_intent_filter(query: str, items: list) -> list:
    q_terms = _keyword_set(query)
    if len(q_terms) < 2:
        return items

    filtered = []
    for item in items:
        strength = _intent_strength(query, item)
        semantic = float(item.get("semantic_score", item.get("score", 0.0)))
        if strength >= 0.22 or semantic >= 0.28:
            enriched = dict(item)
            enriched["intent_strength"] = round(strength, 4)
            filtered.append(enriched)
    return filtered

# -------- Health Check --------
@app.get("/test")
def test():
    return {"message": "Backend connected!"}


@app.post("/clear-memory")
def clear_memory():
    try:
        clear_all_memories()
        return {"status": "cleared"}
    except Exception as e:
        print("Clear Memory Error:", e)
        return {"status": "error"}


@app.get("/analytics")
def analytics(limit: int = 1000):
    records = get_all_memories()
    if limit > 0:
        records = records[-limit:]

    domain_counter = Counter()
    date_counter = Counter()
    hour_counter = Counter()
    keyword_counter = Counter()
    url_counter = Counter()
    total_content_chars = 0
    total_time_spent_ms = 0
    records_with_time_spent = 0
    enriched = []

    for i, item in enumerate(records):
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        content = item.get("content", "")
        domain = _extract_domain(url)
        dt = _parse_event_datetime(item)
        visited_at = dt.isoformat() if dt else None
        visited_date = dt.date().isoformat() if dt else item.get("visited_date")

        time_spent_ms = item.get("time_spent_ms")
        try:
            time_spent_ms = int(time_spent_ms) if time_spent_ms is not None else None
        except Exception:
            time_spent_ms = None

        domain_counter[domain] += 1
        url_counter[url] += 1
        if visited_date:
            date_counter[str(visited_date)] += 1
        if dt:
            hour_counter[dt.hour] += 1

        total_content_chars += len(content or "")
        if time_spent_ms and time_spent_ms > 0:
            total_time_spent_ms += time_spent_ms
            records_with_time_spent += 1

        for kw in _extract_keywords(f"{title} {content[:500]}"):
            keyword_counter[kw] += 1

        enriched.append({
            "id": i + 1,
            "url": url,
            "title": title,
            "domain": domain,
            "visited_at": visited_at,
            "visited_date": visited_date,
            "hour": dt.hour if dt else None,
            "time_spent_ms": time_spent_ms,
            "content_length": len(content or ""),
            "snippet": (content or "")[:240],
        })

    unique_urls = len({r.get("url") for r in records if r.get("url")})
    unique_domains = len(domain_counter)
    duplicate_urls = sum(1 for _, c in url_counter.items() if c > 1)

    top_domains = [
        {"domain": d, "count": c}
        for d, c in domain_counter.most_common(12)
    ]
    top_keywords = [
        {"keyword": k, "count": c}
        for k, c in keyword_counter.most_common(20)
    ]
    visits_by_date = [
        {"date": d, "count": date_counter[d]}
        for d in sorted(date_counter.keys())
    ]
    visits_by_hour = [
        {"hour": h, "count": hour_counter.get(h, 0)}
        for h in range(24)
    ]

    summary = {
        "total_records": len(records),
        "unique_urls": unique_urls,
        "unique_domains": unique_domains,
        "clusters": get_cluster_count(),
        "duplicate_urls": duplicate_urls,
        "avg_content_length": round((total_content_chars / len(records)), 1) if records else 0,
        "total_time_spent_ms": total_time_spent_ms,
        "records_with_time_spent": records_with_time_spent,
    }

    return {
        "summary": summary,
        "top_domains": top_domains,
        "top_keywords": top_keywords,
        "visits_by_date": visits_by_date,
        "visits_by_hour": visits_by_hour,
        "records": list(reversed(enriched)),
    }

# -------- Embed Endpoint --------
@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        vector = get_embedding(req.text)
        return {"embedding": vector}
    except Exception as e:
        print("Embed Error:", e)
        return {"embedding": []}

# -------- Store Endpoint --------
@app.post("/store")
def store(page: Page):
    try:
        if not page.content.strip():
            return {"status": "empty content skipped"}

        chunks = _build_structured_chunks(
            url=page.url,
            title=page.title,
            content=page.content,
            timestamp=page.timestamp if page.timestamp is not None else time.time(),
        )
        if not chunks:
            return {"status": "empty content skipped"}

        texts = [f"{c.get('title', '')}\n\n{c.get('content', '')}".strip() for c in chunks]
        vectors = get_embeddings(texts, is_query=False)
        to_store = []
        for c, vec in zip(chunks, vectors):
            item = dict(c)
            item["embedding"] = vec
            to_store.append(item)

        add_memories(to_store)
        return {"status": "stored", "chunks": len(to_store)}

    except Exception as e:
        print("Store Error:", e)
        return {"status": "error"}

# -------- LLM Answer --------
def generate_answer(query, context):
    print("Generating answer with context length:", len(context))
    print("Query:", query)
    try:
        prompt = f"""
You are a smart personal memory assistant.

Use ONLY the context below to answer the question.

- Be concise (2-3 sentences)
- use contextual meaning of question and compare with context to find relevant info
- If answer is not found, say "I couldn't find anything relevant."

Context:
{context}

Question: {query}

Answer:
"""

        response = requests.post(
            OLLAMA_GEN_URL,
            json={
                "model": "qwen2.5:0.5b",
                "stream": False,
                "prompt": prompt
            },
            timeout=60
        )

        return response.json().get("response", "No response from model.")

    except Exception as e:
        print("LLM Error:", e)
        return "Error generating answer."


def _format_chat_context_item(item: dict) -> str:
    title = item.get("title", "Untitled")
    url = item.get("url", "")
    domain = _extract_domain(url)
    dt = _parse_event_datetime(item)
    visited_at = dt.isoformat() if dt else (item.get("visited_at") or item.get("visited_date") or "unknown")
    time_spent_ms = item.get("time_spent_ms")
    try:
        time_spent_label = f"{int(time_spent_ms) // 1000}s" if time_spent_ms is not None else "unknown"
    except Exception:
        time_spent_label = "unknown"
    content = _clean_text(item.get("content", ""))[:700]
    return (
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Domain: {domain}\n"
        f"Visited: {visited_at}\n"
        f"Time Spent: {time_spent_label}\n"
        f"Content: {content}"
    )


def _chat_sources_from_results(results: list) -> list:
    sources = []
    seen = set()
    for r in results:
        key = (r.get("url", ""), r.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": r.get("title", "Untitled"),
            "url": r.get("url", ""),
            "domain": _extract_domain(r.get("url", "")),
            "score": r.get("score", r.get("semantic_score", 0.0)),
        })
    return sources


def _query_variants_for_chat(message: str) -> list:
    cleaned = _clean_text(message)
    if not cleaned:
        return []

    variants = [cleaned]
    terms = re.findall(r"[a-zA-Z]{3,}", cleaned.lower())
    if len(terms) >= 2:
        variants.append(" ".join(terms[: min(6, len(terms))]))
    if len(terms) >= 3:
        variants.append(" ".join(terms[-min(5, len(terms)) :]))

    deduped = []
    seen = set()
    for v in variants:
        key = v.strip().lower()
        if key and key not in seen:
            deduped.append(v)
            seen.add(key)
    return deduped[:3]


def _extract_time_intent(message: str) -> str:
    return _extract_time_intent_from_text(message)


def _is_metadata_query(message: str) -> bool:
    m = (message or "").lower()
    probes = [
        "when", "date", "visited", "visit", "time", "spent", "how long",
        "last", "first", "timeline", "history", "day", "today", "yesterday",
        "where did", "where can", "list link", "links to", "show link",
    ]
    return any(p in m for p in probes)


def _apply_time_intent_filter(items: list, intent: str) -> list:
    if not intent:
        return items

    now = datetime.now()
    if intent == "today":
        target = now.date()
        return [i for i in items if (_parse_event_datetime(i) and _parse_event_datetime(i).date() == target)]
    if intent == "yesterday":
        target = now.date().fromordinal(now.date().toordinal() - 1)
        return [i for i in items if (_parse_event_datetime(i) and _parse_event_datetime(i).date() == target)]
    if intent == "week":
        min_date = now.date().fromordinal(now.date().toordinal() - 6)
        return [i for i in items if (_parse_event_datetime(i) and _parse_event_datetime(i).date() >= min_date)]
    if intent == "month":
        min_date = now.date().replace(day=1)
        return [i for i in items if (_parse_event_datetime(i) and _parse_event_datetime(i).date() >= min_date)]
    if intent == "last_month":
        first_this = now.date().replace(day=1)
        last_prev = first_this.fromordinal(first_this.toordinal() - 1)
        first_prev = last_prev.replace(day=1)
        return [
            i
            for i in items
            if (_parse_event_datetime(i) and first_prev <= _parse_event_datetime(i).date() <= last_prev)
        ]
    if intent == "recent":
        min_date = now.date().fromordinal(now.date().toordinal() - 2)
        return [i for i in items if (_parse_event_datetime(i) and _parse_event_datetime(i).date() >= min_date)]
    if (intent or "").startswith("days_"):
        try:
            n = int((intent or "").split("_")[1])
            if n > 0:
                min_date = now.date().fromordinal(now.date().toordinal() - (n - 1))
                return [i for i in items if (_parse_event_datetime(i) and _parse_event_datetime(i).date() >= min_date)]
        except Exception:
            pass
    return items


def _apply_exact_phrase_filter(items: list, phrases: list) -> list:
    if not phrases:
        return items
    normalized = [str(p or "").strip().lower() for p in phrases if str(p or "").strip()]
    if not normalized:
        return items
    narrowed = []
    for i in items:
        hay = f"{i.get('title', '')} {i.get('content', '')}".lower()
        if any(p in hay for p in normalized):
            narrowed.append(i)
    return narrowed if narrowed else items


def _apply_must_terms_filter(items: list, terms: list) -> list:
    if not terms:
        return items
    normalized = [str(t or "").strip().lower() for t in terms if str(t or "").strip()]
    if not normalized:
        return items
    scored = []
    for i in items:
        hay = f"{i.get('title', '')} {i.get('content', '')}".lower()
        hits = sum(1 for t in normalized if t in hay)
        enriched = dict(i)
        enriched["must_term_hits"] = hits
        if hits > 0:
            enriched["score"] = float(enriched.get("score", 0.0)) + (0.03 * min(4, hits))
        scored.append(enriched)
    strong = [i for i in scored if i.get("must_term_hits", 0) > 0]
    return strong if strong else scored


def _is_simple_chat_query(message: str) -> bool:
    m = (message or "").strip().lower()
    if not m:
        return True

    greeting = {"hi", "hello", "hey", "yo", "sup", "hii", "heyy"}
    if m in greeting:
        return True

    quick_patterns = [
        r"\bhelp\b",
        r"\bwhat can you do\b",
        r"\bhow many\b",
        r"\btotal records\b",
        r"\btotal domains\b",
        r"\btoday\b",
        r"\byesterday\b",
        r"\blast 7 days\b",
        r"\bthis week\b",
        r"\brecent\b",
        r"\blatest\b",
    ]
    if any(re.search(p, m) for p in quick_patterns):
        return True

    return len(m.split()) <= 4


def _format_brief_items(items: list, limit: int = 3) -> str:
    lines = []
    for i, item in enumerate(items[:limit], start=1):
        title = item.get("title", "Untitled")
        domain = _extract_domain(item.get("url", ""))
        dt = _parse_event_datetime(item)
        when = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown time"
        lines.append(f"{i}. {title} ({domain}) on {when}")
    return "\n".join(lines)


def _simple_chat_reply(message: str) -> tuple[str, list]:
    m = (message or "").strip().lower()
    all_records = get_all_memories()
    recent = sorted(all_records, key=_event_sort_ts, reverse=True)

    opener = random.choice(MIKE_FAST_LINES)

    if m in {"", "hi", "hello", "hey", "yo", "sup", "hii", "heyy"}:
        reply = (
            f"{opener} You can ask about what you read, when you visited it, and how long you stayed. "
            "Try: 'what did you visit today' or 'show recent chess pages'."
        )
        return reply, []

    if "help" in m or "what can you do" in m:
        reply = (
            f"{opener} You can get source-backed answers for content, timeline, domains, and time spent. "
            "Ask things like: 'what did you read yesterday', 'which domain did you visit most', or 'how long on chess.com'."
        )
        return reply, []

    if "how many" in m or "total records" in m or "total domains" in m:
        unique_domains = len({_extract_domain(i.get("url", "")) for i in all_records if i.get("url")})
        reply = (
            f"{opener} You currently have {len(all_records)} records across {unique_domains} domains. "
            f"{random.choice(MIKE_MEMORY_FLEX)}"
        )
        return reply, []

    time_intent = _extract_time_intent(m)
    if time_intent:
        scoped = _apply_time_intent_filter(recent, time_intent)
        if not scoped:
            return f"{opener} No matching visits found for that time window yet.", []
        brief = _format_brief_items(scoped, limit=4)
        reply = f"{opener} Here are your top matches for {time_intent}:\n{brief}\n{random.choice(MIKE_MEMORY_FLEX)}"
        return reply, scoped[:5]

    if "recent" in m or "latest" in m:
        if not recent:
            return f"{opener} No recent records found yet.", []
        brief = _format_brief_items(recent, limit=4)
        reply = f"{opener} Most recent history hits:\n{brief}"
        return reply, recent[:5]

    return "", []


def retrieve_chat_candidates(user_message: str, per_query_k: int = 14) -> list:
    parsed_main = _parse_natural_language_query(user_message)
    fallback_queries = parsed_main.get("fallback_queries") or [parsed_main.get("retrieval_query") or user_message]
    if DEBUG_QUERY_EXPANSION:
        print("[QueryDebug][chat]", {
            "original": user_message,
            "retrieval_query": parsed_main.get("retrieval_query"),
            "fallback_queries": fallback_queries,
            "focus_terms": parsed_main.get("focus_terms"),
            "inferred_tags": parsed_main.get("inferred_tags"),
            "inferred_subtags": parsed_main.get("inferred_subtags"),
            "exact_phrases": parsed_main.get("exact_phrases"),
            "must_include_terms": parsed_main.get("must_include_terms"),
            "domain_hint": parsed_main.get("domain_hint"),
            "time_hint": parsed_main.get("time_hint"),
        })

    pooled = []
    for rq in fallback_queries:
        qv = get_embedding(rq, is_query=True)
        pooled.extend(search_memory(
            qv,
            query_text=rq,
            k=per_query_k,
            clusters_to_search=2,
            query_tags=parsed_main.get("all_tag_hints") or parsed_main.get("inferred_tags") or [],
        ))

    if not pooled:
        variants = _query_variants_for_chat(user_message)
        for v in variants:
            parsed_v = _parse_natural_language_query(v)
            query_for_embed = parsed_v.get("retrieval_query") or v
            qv = get_embedding(query_for_embed, is_query=True)
            pooled.extend(search_memory(
                qv,
                query_text=query_for_embed,
                k=per_query_k,
                clusters_to_search=2,
                query_tags=parsed_main.get("all_tag_hints") or parsed_main.get("inferred_tags") or [],
            ))

    if not pooled:
        for rq in fallback_queries:
            qv = get_embedding(rq, is_query=True)
            pooled.extend(
                search_memory(
                    qv,
                    query_text=rq,
                    k=max(per_query_k + 4, 18),
                    clusters_to_search=3,
                    min_similarity=0.04,
                    query_tags=parsed_main.get("all_tag_hints") or parsed_main.get("inferred_tags") or [],
                )
            )

    # Keep best item per URL+title using semantic score.
    best = {}
    for item in pooled:
        key = (item.get("url", ""), item.get("title", ""))
        prev = best.get(key)
        if prev is None or float(item.get("semantic_score", 0.0)) > float(prev.get("semantic_score", 0.0)):
            best[key] = item

    candidates = list(best.values())

    # For timeline/visit-intent questions, blend in recent records so date answers stay reliable.
    if _is_metadata_query(user_message):
        all_records = get_all_memories()
        recent = sorted(
            all_records,
            key=_event_sort_ts,
            reverse=True,
        )[:25]
        for item in recent:
            key = (item.get("url", ""), item.get("title", ""))
            if key not in best:
                boosted = dict(item)
                boosted["semantic_score"] = max(float(boosted.get("semantic_score", 0.0)), 0.08)
                best[key] = boosted
        candidates = list(best.values())

    candidates = rerank_results(user_message, candidates, top_k=10)
    candidates = _apply_exact_phrase_filter(candidates, parsed_main.get("exact_phrases") or [])
    candidates = _apply_must_terms_filter(candidates, parsed_main.get("must_include_terms") or [])

    candidates = _apply_domain_hint(candidates, parsed_main.get("domain_hint", ""))

    intent = parsed_main.get("time_hint") or _extract_time_intent(user_message)
    if intent:
        filtered = _apply_time_intent_filter(candidates, intent)
        if filtered:
            return filtered[:8]

    return candidates[:8]


def _generate_with_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_GEN_URL,
        json={
            "model": "qwen2.5:0.5b",
            "stream": False,
            "prompt": prompt,
        },
        timeout=60,
    )
    return response.json().get("response", "").strip()


def _generate_with_pollinations(prompt: str) -> str:
    # Try OpenAI-compatible chat endpoint first.
    try:
        response = requests.post(
            f"{POLLINATIONS_TEXT_URL}/openai",
            json={
                "model": "openai",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=20,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Pollinations /openai HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        if isinstance(data, dict):
            if data.get("error") or int(data.get("status") or 0) >= 400:
                raise RuntimeError(f"Pollinations /openai error payload: {str(data)[:300]}")
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = (msg.get("content") or "").strip()
                if content:
                    return content
            content = (data.get("response") or data.get("text") or "").strip()
            if content:
                return content
    except Exception:
        pass

    # Fallback to raw text endpoint.
    encoded = quote(prompt[:1800])
    response = requests.get(
        f"{POLLINATIONS_TEXT_URL}/{encoded}?model=openai",
        timeout=20,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Pollinations text HTTP {response.status_code}: {response.text[:200]}")

    text = (response.text or "").strip()
    if text.startswith("{"):
      try:
          payload = json.loads(text)
          if isinstance(payload, dict) and (payload.get("error") or int(payload.get("status") or 0) >= 400):
              raise RuntimeError(f"Pollinations text error payload: {str(payload)[:300]}")
      except json.JSONDecodeError:
          pass
    return text


def _pollinations_available() -> bool:
    return time.time() >= POLLINATIONS_COOLDOWN_UNTIL


def _mark_pollinations_down(reason: Exception) -> None:
    global POLLINATIONS_COOLDOWN_UNTIL
    POLLINATIONS_COOLDOWN_UNTIL = time.time() + POLLINATIONS_COOLDOWN_SECONDS
    print(f"Pollinations temporarily disabled for {POLLINATIONS_COOLDOWN_SECONDS}s: {reason}")


def generate_history_chat_answer(user_message: str, context_items: list, provider: str = "ollama") -> str:
    if not context_items:
        return "I checked your history and found no strong matches yet. Give me a more specific clue like site name, topic, or time window."

    context_block = "\n\n".join(_format_chat_context_item(item) for item in context_items[:8])

    prompt = f"""
You are Mike Ross, a sharp personal history assistant with fast legal-room confidence.
Style:
- concise, confident, practical
- natural conversation tone (not robotic)
- witty and human, but not theatrical
- 4-8 short sentences
- explicitly reference where information came from (site/title)
- do not invent facts beyond context
- synthesize across multiple records when useful
- write in second-person guidance using "you" and "your"
- avoid first-person voice (do not use: I, me, my, mine, I've, I'll)
- do not quote or reproduce copyrighted TV dialogue

Task:
Answer the user based only on the browsing history context below.
If the answer is weak/partial, say what is missing and suggest a better follow-up query.
When possible, include timeline cues (today/yesterday/date) from the provided context.
If asked about date visited, when, or time spent, include explicit fields like Visited and Time Spent from matching records.
If the user asks where they read something, which page or site it was on, or to list, show, or give links or URLs, include the URL (and title/domain) from the context for each top matching record — do not omit links for those questions.
If multiple records match, rank the top 2-4 and state the most likely match first.

Browsing History Context:
{context_block}

User Message:
{user_message}

Mike Ross:
"""

    provider_norm = (provider or "ollama").strip().lower()
    try:
        raw = ""

        if provider_norm == "pollinations":
            if _pollinations_available():
                try:
                    raw = _generate_with_pollinations(prompt)
                except Exception as pollinations_err:
                    _mark_pollinations_down(pollinations_err)
            if not raw:
                try:
                    raw = _generate_with_ollama(prompt)
                except Exception as ollama_err:
                    print("Ollama Fallback Error:", ollama_err)
        else:
            try:
                raw = _generate_with_ollama(prompt)
            except Exception as ollama_err:
                print("Ollama Error:", ollama_err)
            if not raw:
                if _pollinations_available():
                    try:
                        raw = _generate_with_pollinations(prompt)
                    except Exception as pollinations_err:
                        _mark_pollinations_down(pollinations_err)

        if not raw:
            raw = "History was retrieved, but a response could not be generated right now."
        voiced = _normalize_mike_voice(raw)
        return _add_mike_flair(voiced, user_message)
    except Exception as e:
        print("Chat LLM Error:", e)
        top = context_items[:3]
        bullets = []
        for i, item in enumerate(top, start=1):
            bullets.append(f"{i}. {item.get('title', 'Untitled')} ({_extract_domain(item.get('url', ''))})")
        if not bullets:
            return "I found related history, but the response model is unavailable right now."
        return (
            "I found relevant items in your history. Here are the strongest matches right now:\n"
            + "\n".join(bullets)
            + "\nAsk me to narrow this by date, domain, or a specific detail."
        )


@app.post("/chat-history")
def chat_history(req: ChatRequest):
    try:
        message = (req.message or "").strip()
        provider = (req.provider or "ollama").strip().lower()
        if len(message) < 2:
            return {
                "bot": "Mike Ross",
                "reply": "Ask me anything about what you've read. I can trace pages, topics, and patterns from your history.",
                "sources": [],
            }

        # Bypass LLM for simple asks to keep chat fast and stable.
        if _is_simple_chat_query(message):
            fast_reply, fast_items = _simple_chat_reply(message)
            if fast_reply:
                return {
                    "bot": "Mike Ross",
                    "reply": fast_reply,
                    "sources": _chat_sources_from_results(fast_items),
                }

        ranked = retrieve_chat_candidates(message, per_query_k=14)

        reply = generate_history_chat_answer(message, ranked, provider=provider)
        sources = _chat_sources_from_results(ranked)

        return {
            "bot": "Mike Ross",
            "reply": reply,
            "sources": sources,
        }
    except Exception as e:
        print("Chat History Error:", e)
        return {
            "bot": "Mike Ross",
            "reply": "Something went wrong while reading your history. Try again in a moment.",
            "sources": [],
        }


def _normalize_mike_voice(text: str) -> str:
    if not text:
        return text
    out = text
    replacements = [
        (r"\bI'm\b", "You are"),
        (r"\bI am\b", "You are"),
        (r"\bI've\b", "You have"),
        (r"\bI'll\b", "You will"),
        (r"\bI'd\b", "You would"),
        (r"\bmy\b", "your"),
        (r"\bmine\b", "yours"),
        (r"\bme\b", "you"),
        (r"\bI\b", "You"),
    ]
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)

    # Keep responses in the assistant persona even if model prepends labels.
    out = re.sub(r"^\s*(assistant|mike ross)\s*:\s*", "", out, flags=re.IGNORECASE)
    return out


def _add_mike_flair(text: str, user_message: str) -> str:
    if not text:
        return text

    trimmed = text.strip()
    lower = trimmed.lower()

    # Add occasional opening cadence so responses feel less templated.
    if random.random() < 0.45 and not lower.startswith(("quick read", "here is", "straight answer", "you are looking")):
        trimmed = f"{random.choice(MIKE_OPENERS)} {trimmed}"

    # Add occasional memory-flex line, especially for timeline/time/date questions.
    needs_memory_flex = _is_metadata_query(user_message) or any(k in lower for k in ["visited", "date", "time", "timeline"])
    if needs_memory_flex and random.random() < 0.55:
        if trimmed.endswith((".", "!", "?")):
            trimmed = f"{trimmed} {random.choice(MIKE_MEMORY_FLEX)}"
        else:
            trimmed = f"{trimmed}. {random.choice(MIKE_MEMORY_FLEX)}"

    return trimmed


@app.post("/reload-query-knowledge")
def reload_query_knowledge():
    try:
        _apply_query_knowledge_pack()
        return {
            "status": "reloaded",
            "knowledge_file": QUERY_KNOWLEDGE_FILE,
            "query_expansions": len(QUERY_EXPANSIONS),
            "synonym_roots": len(SYNONYM_MAP),
            "tag_roots": len(TAG_KEYWORDS),
            "subtag_rules": len(SUBTAG_RULES),
            "filler_terms": len(NL_FILLER_TERMS),
        }
    except Exception as e:
        print("Reload Query Knowledge Error:", e)
        return {"status": "error", "error": str(e)}

# -------- Query Endpoint --------
@app.post("/query")
def query(q: Query):
    try:
        parsed_query = _parse_natural_language_query(q.query)
        fallback_queries = parsed_query.get("fallback_queries") or [parsed_query.get("retrieval_query") or q.query]
        retrieval_query = parsed_query.get("retrieval_query") or q.query
        if DEBUG_QUERY_EXPANSION:
            print("[QueryDebug][/query]", {
                "original": q.query,
                "retrieval_query": retrieval_query,
                "fallback_queries": fallback_queries,
                "focus_terms": parsed_query.get("focus_terms"),
                "inferred_tags": parsed_query.get("inferred_tags"),
                "inferred_subtags": parsed_query.get("inferred_subtags"),
                "exact_phrases": parsed_query.get("exact_phrases"),
                "must_include_terms": parsed_query.get("must_include_terms"),
                "domain_hint": parsed_query.get("domain_hint"),
                "time_hint": parsed_query.get("time_hint"),
            })

        candidates = []
        chosen_stage = "strict"
        for rq in fallback_queries:
            query_vector = get_embedding(rq, is_query=True)
            stage_candidates = search_memory(
                query_vector,
                query_text=rq,
                k=max(MATCH_LIMIT * 4, 12),
                clusters_to_search=2,
                query_tags=parsed_query.get("all_tag_hints") or parsed_query.get("inferred_tags") or [],
            )
            stage_candidates = _apply_domain_hint(stage_candidates, parsed_query.get("domain_hint", ""))
            if parsed_query.get("time_hint"):
                stage_candidates = _apply_time_intent_filter(stage_candidates, parsed_query.get("time_hint"))
            stage_candidates = _apply_exact_phrase_filter(stage_candidates, parsed_query.get("exact_phrases") or [])
            stage_candidates = _apply_must_terms_filter(stage_candidates, parsed_query.get("must_include_terms") or [])
            if stage_candidates:
                candidates = stage_candidates
                chosen_stage = f"strict:{rq}"
                break

        results = rerank_results(q.query, candidates, top_k=MATCH_LIMIT, min_score=0.12)
        results = _apply_intent_filter(q.query, results)

        if not results:
            for rq in fallback_queries:
                query_vector = get_embedding(rq, is_query=True)
                stage_candidates = search_memory(
                    query_vector,
                    query_text=rq,
                    k=max(MATCH_LIMIT * 5, 16),
                    clusters_to_search=3,
                    min_similarity=0.04,
                    query_tags=parsed_query.get("all_tag_hints") or parsed_query.get("inferred_tags") or [],
                )
                stage_candidates = _apply_domain_hint(stage_candidates, parsed_query.get("domain_hint", ""))
                if parsed_query.get("time_hint"):
                    stage_candidates = _apply_time_intent_filter(stage_candidates, parsed_query.get("time_hint"))
                stage_candidates = _apply_exact_phrase_filter(stage_candidates, parsed_query.get("exact_phrases") or [])
                stage_candidates = _apply_must_terms_filter(stage_candidates, parsed_query.get("must_include_terms") or [])
                relaxed_results = rerank_results(q.query, stage_candidates, top_k=MATCH_LIMIT, min_score=0.06)
                relaxed_results = _apply_intent_filter(q.query, relaxed_results)
                if relaxed_results:
                    results = relaxed_results
                    chosen_stage = f"relaxed:{rq}"
                    break

        if DEBUG_QUERY_EXPANSION:
            print("[QueryDebug][/query-stage]", {"stage": chosen_stage, "result_count": len(results)})

        if not results:
            return {
                "answer": "No memory found yet. Browse something first!",
                "sources": []
            }

        context = "\n\n".join([r["content"] for r in results])
        print("Context for LLM:", context[:500])  # print first 500 chars of context

        answer = generate_answer(q.query, context)

        # Clean sources (only send useful fields)
        sources = [
            {
                "url": r["url"],
                "title": r.get("title", "Untitled"),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]

        return {
            "answer": answer.strip(),
            "sources": sources
        }

    except Exception as e:
        print("Query Error:", e)
        return {
            "answer": "Something went wrong.",
            "sources": []
        }
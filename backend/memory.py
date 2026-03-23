import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Optional

import faiss
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
META_FILE = os.path.join(DATA_DIR, "metadata.json")

DEFAULT_DIM = 768
DEFAULT_CLUSTERS = 24
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 120
HNSW_EF_SEARCH = 64

_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")

metadata = []
embeddings = []
idf = {}
cluster_centroids = {}
cluster_doc_ids = {}
cluster_sizes = {}
dimension = DEFAULT_DIM
num_clusters = DEFAULT_CLUSTERS


def _normalize_rows(vec: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec
    return vec / norm


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _make_index(dim: int) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(dim, HNSW_M)
    idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    idx.hnsw.efSearch = HNSW_EF_SEARCH
    return idx


def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _parse_iso_timestamp(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return 0.0


def _tokenize(text: str) -> list:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _normalize_tags(tags_value) -> list:
    if not isinstance(tags_value, list):
        return []
    out = []
    seen = set()
    for t in tags_value:
        s = str(t or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[:8]


def _compute_idf() -> None:
    global idf
    n_docs = max(1, len(metadata))
    doc_freq = {}
    for rec in metadata:
        terms = set(rec.get("keywords") or _tokenize(f"{rec.get('title', '')} {rec.get('content', '')}"))
        for t in terms:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    idf = {t: math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5))) for t, df in doc_freq.items()}


def _bm25_score(query_terms: list, rec: dict, avg_len: float) -> float:
    if not query_terms:
        return 0.0
    terms = rec.get("keywords") or _tokenize(f"{rec.get('title', '')} {rec.get('content', '')}")
    if not terms:
        return 0.0

    tf = {}
    for t in terms:
        tf[t] = tf.get(t, 0) + 1

    k1 = 1.5
    b = 0.75
    doc_len = len(terms)
    score = 0.0
    for t in query_terms:
        if t not in tf:
            continue
        f = tf[t]
        t_idf = idf.get(t, 0.0)
        denom = f + k1 * (1.0 - b + b * (doc_len / max(avg_len, 1.0)))
        score += t_idf * ((f * (k1 + 1.0)) / max(denom, 1e-8))
    return float(score)


def _title_match_score(query_terms: list, title: str) -> float:
    if not query_terms:
        return 0.0
    title_terms = set(_tokenize(title or ""))
    if not title_terms:
        return 0.0
    overlap = len(set(query_terms) & title_terms)
    return float(overlap) / max(1.0, float(len(set(query_terms))))


def _keyword_overlap_score(query_terms: list, rec_terms: list) -> float:
    if not query_terms or not rec_terms:
        return 0.0
    q = set(query_terms)
    r = set(rec_terms)
    return float(len(q & r)) / max(1.0, float(len(q)))


def _timestamp_recency_score(timestamp_value) -> float:
    tsf = _safe_float(timestamp_value, 0.0)
    if tsf <= 0:
        return 0.0
    now = datetime.now(tz=timezone.utc).timestamp()
    age_days = max(0.0, (now - tsf) / 86400.0)
    if age_days <= 1.0:
        return 1.0
    if age_days >= 60.0:
        return 0.0
    return max(0.0, 1.0 - (age_days / 60.0))


def _build_clusters() -> None:
    global cluster_centroids, cluster_doc_ids, cluster_sizes, num_clusters
    cluster_centroids = {}
    cluster_doc_ids = {}
    cluster_sizes = {}

    if not embeddings:
        num_clusters = 1
        return

    x = np.array(embeddings, dtype="float32")
    n = x.shape[0]
    k = min(max(2, int(round(math.sqrt(n)))), DEFAULT_CLUSTERS)
    if n < 6:
        k = 1
    num_clusters = max(1, k)

    if num_clusters == 1:
        centroid = _normalize_vec(x.mean(axis=0))
        cluster_centroids[0] = centroid
        ids = list(range(n))
        cluster_doc_ids[0] = ids
        cluster_sizes[0] = len(ids)
        for i in range(len(metadata)):
            metadata[i]["cluster_id"] = 0
        return

    kmeans = faiss.Kmeans(d=x.shape[1], k=num_clusters, niter=22, verbose=False, gpu=False)
    kmeans.train(x)
    _, assign = kmeans.index.search(x, 1)

    for i, cluster_value in enumerate(assign.reshape(-1).tolist()):
        cid = int(cluster_value)
        metadata[i]["cluster_id"] = cid
        cluster_doc_ids.setdefault(cid, []).append(i)

    for cid, ids in cluster_doc_ids.items():
        subset = x[ids]
        centroid = _normalize_vec(subset.mean(axis=0))
        cluster_centroids[cid] = centroid
        cluster_sizes[cid] = len(ids)


def _select_clusters(query_vector: np.ndarray, top_n: int = 2) -> list:
    if not cluster_centroids:
        return []
    if len(cluster_centroids) <= top_n:
        return sorted(cluster_centroids.keys())

    q = _normalize_vec(query_vector.astype("float32"))
    scored = []
    for cid, centroid in cluster_centroids.items():
        sim = float(np.dot(q, centroid))
        scored.append((sim, cid))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [cid for _, cid in scored[:top_n]]


def _rebuild_index() -> None:
    global index
    index = _make_index(dimension)
    if embeddings:
        arr = np.array(embeddings, dtype="float32")
        arr = _normalize_rows(arr)
        index.add(arr)


def _migrate_legacy_records(raw_records: list, old_index: Optional[faiss.Index]) -> list:
    out = []
    for i, item in enumerate(raw_records):
        rec = dict(item)
        emb = rec.get("embedding")

        if (not isinstance(emb, list) or not emb) and old_index is not None and i < old_index.ntotal:
            try:
                emb = old_index.reconstruct(i).astype("float32").tolist()
            except Exception:
                emb = None

        if not isinstance(emb, list) or not emb:
            continue

        title = rec.get("title") or rec.get("url") or "Untitled"
        content = rec.get("content") or ""
        url = rec.get("url") or ""
        domain = rec.get("domain") or "unknown"
        timestamp = rec.get("timestamp")
        if timestamp is None:
            timestamp = _parse_iso_timestamp(rec.get("visited_at")) or _parse_iso_timestamp(rec.get("visited_date"))

        out.append({
            "id": rec.get("id") or f"legacy-{i}",
            "content": content,
            "title": title,
            "url": url,
            "embedding": emb,
            "cluster_id": int(_safe_float(rec.get("cluster_id"), 0.0)),
            "keywords": rec.get("keywords") or _tokenize(f"{title} {content}"),
            "timestamp": _safe_float(timestamp, 0.0),
            "domain": domain,
            "section": rec.get("section") or "body",
            "tags": _normalize_tags(rec.get("tags")),
        })
    return out


def _load_state() -> None:
    global metadata, embeddings, dimension, index
    _ensure_data_dir()

    old_index = None
    if os.path.exists(INDEX_FILE):
        try:
            old_index = faiss.read_index(INDEX_FILE)
        except Exception:
            old_index = None

    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    else:
        payload = {}

    if isinstance(payload, dict):
        records = payload.get("records") or []
        loaded_dim = int(payload.get("dimension") or 0)
        dimension = loaded_dim if loaded_dim > 0 else DEFAULT_DIM
    elif isinstance(payload, list):
        records = _migrate_legacy_records(payload, old_index)
        dimension = DEFAULT_DIM
    else:
        records = []
        dimension = DEFAULT_DIM

    cleaned = []
    for i, rec in enumerate(records):
        item = dict(rec)
        emb = item.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        item["id"] = item.get("id") or f"mem-{i}"
        item["content"] = item.get("content") or ""
        item["title"] = item.get("title") or "Untitled"
        item["url"] = item.get("url") or ""
        item["domain"] = item.get("domain") or "unknown"
        item["keywords"] = item.get("keywords") or _tokenize(f"{item['title']} {item['content']}")
        item["timestamp"] = _safe_float(item.get("timestamp"), 0.0)
        item["cluster_id"] = int(_safe_float(item.get("cluster_id"), 0.0))
        item["section"] = item.get("section") or "body"
        item["tags"] = _normalize_tags(item.get("tags"))
        cleaned.append(item)

    metadata = cleaned
    if metadata:
        first = metadata[0].get("embedding")
        if isinstance(first, list) and first:
            dimension = len(first)

    embeddings = [list(map(float, item["embedding"])) for item in metadata]
    _rebuild_index()
    _build_clusters()
    _compute_idf()


def save() -> None:
    _ensure_data_dir()
    faiss.write_index(index, INDEX_FILE)
    payload = {
        "dimension": dimension,
        "num_clusters": num_clusters,
        "records": metadata,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def add_memories(records: list) -> None:
    global metadata, embeddings, dimension
    if not records:
        return

    for rec in records:
        emb = rec.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue

        if not embeddings:
            dimension = len(emb)
        if len(emb) != dimension:
            continue

        item = {
            "id": rec.get("id") or f"mem-{len(metadata) + 1}",
            "content": rec.get("content") or "",
            "title": rec.get("title") or "Untitled",
            "url": rec.get("url") or "",
            "embedding": [float(x) for x in emb],
            "cluster_id": int(_safe_float(rec.get("cluster_id"), 0.0)),
            "keywords": rec.get("keywords") or _tokenize(f"{rec.get('title', '')} {rec.get('content', '')}"),
            "timestamp": _safe_float(rec.get("timestamp"), 0.0),
            "domain": rec.get("domain") or "unknown",
            "section": rec.get("section") or "body",
            "tags": _normalize_tags(rec.get("tags")),
        }
        metadata.append(item)
        embeddings.append(item["embedding"])

    _rebuild_index()
    _build_clusters()
    _compute_idf()
    save()


def add_memory(vector, data) -> None:
    if vector is None:
        return
    rec = dict(data or {})
    rec["embedding"] = vector
    add_memories([rec])


def search_memory(
    query_vector,
    query_text: str = "",
    k: int = 8,
    min_similarity: float = 0.10,
    clusters_to_search: int = 2,
    query_tags: Optional[list] = None,
):
    if not metadata:
        return []

    q = np.array(query_vector, dtype="float32")
    if q.ndim != 1 or q.shape[0] != dimension:
        return []
    q = _normalize_vec(q)

    selected = _select_clusters(q, top_n=max(1, clusters_to_search))
    if not selected:
        selected = sorted(cluster_doc_ids.keys()) or [0]

    candidate_ids = []
    for cid in selected:
        candidate_ids.extend(cluster_doc_ids.get(cid, []))

    if not candidate_ids:
        candidate_ids = list(range(len(metadata)))

    candidate_matrix = np.array([embeddings[i] for i in candidate_ids], dtype="float32")
    semantic = np.dot(candidate_matrix, q)

    query_terms = _tokenize(query_text)
    query_tag_set = set(_normalize_tags(query_tags or []))
    avg_len = float(np.mean([len((metadata[i].get("keywords") or [])) for i in candidate_ids])) if candidate_ids else 1.0

    bm25_scores = []
    title_scores = []
    for idx in candidate_ids:
        rec = metadata[idx]
        bm25_scores.append(_bm25_score(query_terms, rec, avg_len=avg_len))
        title_scores.append(_title_match_score(query_terms, rec.get("title", "")))

    bm25_np = np.array(bm25_scores, dtype="float32") if bm25_scores else np.array([], dtype="float32")
    if bm25_np.size > 0 and float(np.max(bm25_np)) > float(np.min(bm25_np)):
        bm25_norm = (bm25_np - float(np.min(bm25_np))) / (float(np.max(bm25_np)) - float(np.min(bm25_np)) + 1e-8)
    else:
        bm25_norm = np.zeros_like(bm25_np)

    title_np = np.array(title_scores, dtype="float32") if title_scores else np.array([], dtype="float32")

    semantic_norm = (semantic + 1.0) / 2.0
    hybrid = (0.6 * semantic_norm) + (0.3 * bm25_norm) + (0.1 * title_np)

    initial_order = np.argsort(-hybrid)[: max(20, k * 4)]

    rescored = []
    for pos in initial_order.tolist():
        rec_idx = candidate_ids[pos]
        rec = metadata[rec_idx]

        rec_terms = rec.get("keywords") or []
        overlap = _keyword_overlap_score(query_terms, rec_terms)
        recency = _timestamp_recency_score(rec.get("timestamp"))
        rec_tag_set = set(_normalize_tags(rec.get("tags")))
        tag_overlap = float(len(query_tag_set & rec_tag_set)) / max(1.0, float(len(query_tag_set))) if query_tag_set else 0.0

        final_score = (0.68 * float(hybrid[pos])) + (0.16 * overlap) + (0.10 * recency) + (0.06 * tag_overlap)
        semantic_score = float(semantic[pos])
        if semantic_score < min_similarity:
            continue

        item = dict(rec)
        item["semantic_score"] = round(semantic_score, 4)
        item["vector_score"] = round(float(semantic_norm[pos]), 4)
        item["bm25_score"] = round(float(bm25_norm[pos]) if bm25_norm.size else 0.0, 4)
        item["title_score"] = round(float(title_np[pos]) if title_np.size else 0.0, 4)
        item["tag_score"] = round(float(tag_overlap), 4)
        item["score"] = round(float(final_score), 4)
        rescored.append(item)

    if not rescored:
        return []

    allowed_clusters = set(selected)
    filtered = [r for r in rescored if int(r.get("cluster_id", -1)) in allowed_clusters]
    filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return filtered[:k]


def get_all_memories():
    return [dict(item) for item in metadata]


def get_cluster_count() -> int:
    return len(cluster_doc_ids)


def clear_all_memories() -> None:
    global metadata, embeddings, idf, cluster_centroids, cluster_doc_ids, cluster_sizes, num_clusters
    metadata = []
    embeddings = []
    idf = {}
    cluster_centroids = {}
    cluster_doc_ids = {}
    cluster_sizes = {}
    num_clusters = 1
    _rebuild_index()
    save()


_load_state()
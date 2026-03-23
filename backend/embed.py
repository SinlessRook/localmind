import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Better semantic balance for retrieval than MiniLM in this use case.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def _clean_text(text: str) -> str:
    # Normalize whitespace so semantically similar pages map more consistently.
    return re.sub(r"\s+", " ", (text or "").strip())


def _chunk_words(text: str, chunk_size: int = 180, overlap: int = 40):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def _with_prefix(text: str, is_query: bool) -> str:
    cleaned = _clean_text(text)
    if is_query:
        return f"query: {cleaned}"
    return f"passage: {cleaned}"


def get_embeddings(texts, is_query: bool = False):
    prepared = []
    dim = model.get_sentence_embedding_dimension()

    for t in texts:
        cleaned = _clean_text(t)
        if cleaned:
            prepared.append(_with_prefix(cleaned, is_query=is_query))
        else:
            prepared.append(None)

    non_empty = [p for p in prepared if p is not None]
    if not non_empty:
        return [[0.0] * dim for _ in texts]

    encoded = model.encode(
        non_empty,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=16,
    )

    out = []
    cursor = 0
    for p in prepared:
        if p is None:
            out.append([0.0] * dim)
            continue
        vec = _l2_normalize(encoded[cursor].astype("float32"))
        out.append(vec.tolist())
        cursor += 1
    return out


def get_embedding(text, is_query: bool = False):
    cleaned = _clean_text(text)
    if not cleaned:
        return [0.0] * model.get_sentence_embedding_dimension()

    words = cleaned.split()

    # For long documents, embed overlapping chunks and mean-pool.
    if len(words) > 220:
        chunks = _chunk_words(cleaned)
        prefixed = [_with_prefix(chunk, is_query=is_query) for chunk in chunks]
        chunk_vectors = model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=16,
        )
        pooled = chunk_vectors.mean(axis=0)
        pooled = _l2_normalize(pooled.astype("float32"))
        return pooled.tolist()

    vector = model.encode(
        _with_prefix(cleaned, is_query=is_query),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vector = _l2_normalize(vector.astype("float32"))
    return vector.tolist()
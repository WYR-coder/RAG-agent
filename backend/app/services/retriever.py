"""Hybrid retrieval: vector (BGE-M3) + BM25 (jieba) -> RRF fusion + LLM rerank."""

import json
import os
import pickle
import re
from pathlib import Path

import jieba

from ..core.db import get_db, COLLECTION
from ..core.embedding import embed_query

ROOT = Path(__file__).resolve().parent.parent.parent.parent

BM25_FILE = ROOT / "data" / "chunks" / "bm25_index.pkl"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"

RRF_K = 60
VECTOR_WEIGHT = 0.3
BM25_WEIGHT = 0.7

# ── Query synonym expansion ────────────────────────────────────────────
# Loaded from data/query_synonyms.txt at module init; updated via API.
QUERY_SYNONYMS: dict[str, str] = {}

RERANK_CANDIDATES = 15
CONTENT_TYPE_BOOST = {}
IMAGE_CHUNK_BOOST = 1.08
FILE_BOOST_MAX = 20.0
SECTION_BOOST_MAX = 5.0

_QUERY_STOP_WORDS = {
    "怎么", "如何", "什么", "哪些", "为什么", "这个", "那个", "一下", "吗", "呢", "吧",
    "系统", "设计", "要求", "标准", "规范", "规定", "做法",
    "配置", "考虑", "应", "宜", "可", "不应", "不宜", "不得",
    "是否", "怎样", "什么样", "如何设计",
}

_cached_filenames: list[str] | None = None
_bm25 = None
_tokenized_corpus = None
_chunks = None
_reranker = None


def reload_config():
    """Reload synonyms and dictionary from disk (called after user updates)."""
    global QUERY_SYNONYMS
    QUERY_SYNONYMS = {}
    _dict_path = ROOT / "data" / "terms_dict.txt"
    if _dict_path.exists():
        jieba.load_userdict(str(_dict_path))
    _syn_path = ROOT / "data" / "query_synonyms.txt"
    if _syn_path.exists():
        for line in _syn_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                QUERY_SYNONYMS[k.strip()] = v.strip()


# Initialize on module load
reload_config()


def _expand_query(query: str) -> str:
    added = []
    for alias, canonical in QUERY_SYNONYMS.items():
        if alias in query and canonical not in query:
            added.append(canonical)
    if added:
        return query + " " + " ".join(added)
    return query


def _get_bm25():
    global _bm25, _tokenized_corpus
    if _bm25 is None:
        with open(BM25_FILE, "rb") as f:
            _bm25, _tokenized_corpus = pickle.load(f)
    return _bm25, _tokenized_corpus


def _get_chunks():
    global _chunks
    if _chunks is None:
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            _chunks = [json.loads(line) for line in f]
    return _chunks


def invalidate_cache():
    global _bm25, _tokenized_corpus, _chunks, _reranker, _cached_filenames
    _bm25 = None
    _tokenized_corpus = None
    _chunks = None
    _reranker = None
    _cached_filenames = None


def _build_expr(category_filter: list[str] | None, content_type_filter: list[str] | None) -> str | None:
    parts = []
    if category_filter:
        quoted = ", ".join(f'"{s}"' for s in category_filter)
        parts.append(f"category in [{quoted}]")
    if content_type_filter:
        quoted = ", ".join(f'"{c}"' for c in content_type_filter)
        parts.append(f"content_type in [{quoted}]")
    return " && ".join(parts) if parts else None


def _filter_bm25_indices(indices: list[int], category_filter: list[str] | None, content_type_filter: list[str] | None) -> list[int]:
    if not category_filter and not content_type_filter:
        return indices
    chunks = _get_chunks()
    return [
        idx for idx in indices
        if (not category_filter or chunks[idx].get("category") in category_filter)
        and (not content_type_filter or chunks[idx].get("content_type") in content_type_filter)
    ]


def _compute_file_boosts(query: str) -> dict[str, float]:
    global _cached_filenames
    query_words = [w for w in jieba.cut(query) if len(w.strip()) >= 2 and w.strip() not in _QUERY_STOP_WORDS]
    if not query_words:
        return {}
    if _cached_filenames is None:
        chunks = _get_chunks()
        _cached_filenames = list({c.get("source_file", "") for c in chunks if c.get("source_file")})
    boosts: dict[str, float] = {}
    for fname in _cached_filenames:
        if not fname:
            continue
        fname_clean = fname.replace(".md", "")
        matched = sum(1 for w in query_words if w in fname_clean)
        if matched > 0:
            ratio = matched / len(query_words)
            boosts[fname] = 1.0 + ratio * (FILE_BOOST_MAX - 1.0)
    return boosts


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        model_path = str(ROOT / "models" / "BAAI" / "bge-reranker-v2-m3")
        _reranker = CrossEncoder(model_path, local_files_only=True)
    return _reranker


def _bge_rerank(query: str, candidates: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
    if len(candidates) <= 3:
        return candidates[:top_k]
    chunks = _get_chunks()
    to_rerank = candidates[:RERANK_CANDIDATES]
    reranker = _get_reranker()
    pairs = [(query, chunks[ci]["content"][:500]) for ci, score in to_rerank]
    scores = reranker.predict(pairs, show_progress_bar=False)
    reranked = sorted(zip(to_rerank, scores), key=lambda x: x[1], reverse=True)
    return [item for item, score in reranked[:top_k]]


def _classify_relevance(chunks: list[dict], query: str = "") -> list[dict]:
    if not chunks:
        return chunks
    chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
    max_score = chunks[0]["relevance_score"]
    threshold = max_score * 0.9
    for c in chunks:
        c["relevance"] = "direct" if c["relevance_score"] >= threshold else "indirect"
    return chunks


def retrieve(
    query: str,
    top_k: int = 10,
    category_filter: list[str] | None = None,
    content_type_filter: list[str] | None = None,
) -> list[dict]:
    db = get_db()
    col = db.get_collection(COLLECTION)
    col.load()
    expr = _build_expr(category_filter, content_type_filter)

    query = _expand_query(query)

    # Vector search
    query_vec = embed_query(query)
    vector_hits = col.search(
        query_vectors=[query_vec],
        top_k=top_k * 5,
        anns_field="embedding",
        output_fields=["chunk_index", "content", "category", "source_file", "section_path", "content_type", "has_image", "image_ids"],
        expr=expr,
    )[0]

    # BM25 search
    bm25, _tokenized_corpus = _get_bm25()
    tokenized_query = [w for w in jieba.cut(query) if w.strip() and w not in _QUERY_STOP_WORDS]
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranked_indices = _filter_bm25_indices(bm25_ranked_indices, category_filter, content_type_filter)
    bm25_ranked_indices = bm25_ranked_indices[:top_k * 5]

    # RRF fusion
    rrf: dict[int, float] = {}
    for rank, hit in enumerate(vector_hits):
        ci = hit["entity"]["chunk_index"]
        rrf[ci] = rrf.get(ci, 0) + VECTOR_WEIGHT / (RRF_K + rank + 1)
    for rank, ci in enumerate(bm25_ranked_indices):
        rrf[ci] = rrf.get(ci, 0) + BM25_WEIGHT / (RRF_K + rank + 1)

    chunks = _get_chunks()

    # Content-type boost
    for ci in list(rrf.keys()):
        ct = chunks[ci].get("content_type", "")
        boost = CONTENT_TYPE_BOOST.get(ct, 1.0)
        if boost != 1.0:
            rrf[ci] *= boost

    # Image boost
    for ci in list(rrf.keys()):
        if chunks[ci].get("has_image", False):
            rrf[ci] *= IMAGE_CHUNK_BOOST

    # File-title boost
    file_boosts = _compute_file_boosts(query)
    if file_boosts:
        for ci in list(rrf.keys()):
            fname = chunks[ci].get("source_file", "")
            fb = file_boosts.get(fname, 1.0)
            if fb != 1.0:
                rrf[ci] *= fb

    # Section-path boost
    query_words = [w for w in jieba.cut(query) if len(w.strip()) >= 2 and w.strip() not in _QUERY_STOP_WORDS]
    if query_words:
        for ci in list(rrf.keys()):
            section = chunks[ci].get("section_path", "")
            if not section:
                continue
            matched = sum(1 for w in query_words if w in section)
            if matched > 0:
                ratio = matched / len(query_words)
                rrf[ci] *= 1.0 + ratio * (SECTION_BOOST_MAX - 1.0)

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k * 4]

    # File diversity: top-5 files, max 2 per file
    file_scores: dict[str, float] = {}
    for ci, score in fused:
        fname = chunks[ci].get("source_file", "")
        if fname not in file_scores or score > file_scores[fname]:
            file_scores[fname] = score
    top_files = sorted(file_scores, key=file_scores.get, reverse=True)[:5]
    file_counts: dict[str, int] = {}
    filtered: list[tuple[int, float]] = []
    for ci, score in fused:
        fname = chunks[ci].get("source_file", "")
        if fname not in top_files:
            continue
        if file_counts.get(fname, 0) >= 2:
            continue
        file_counts[fname] = file_counts.get(fname, 0) + 1
        filtered.append((ci, score))
    fused = filtered[:top_k]

    # LLM Rerank
    if len(fused) > 3:
        fused = _bge_rerank(query, fused, top_k)

    chunks = _get_chunks()
    results = []
    for ci, score in fused:
        c = chunks[ci]
        results.append({
            "chunk_index": ci,
            "content": c["content"],
            "category": c.get("category", ""),
            "source_file": c["source_file"],
            "section_path": c.get("section_path", ""),
            "content_type": c["content_type"],
            "relevance_score": round(score, 6),
            "has_image": c.get("has_image", False),
            "image_ids": c.get("image_ids", ""),
            "relevance": "",
        })

    return _classify_relevance(results, query)

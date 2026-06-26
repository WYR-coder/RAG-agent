"""Hybrid retrieval: vector (BGE-M3) + BM25 (jieba) → RRF fusion + LLM rerank."""

import json
import os
import pickle
import re
from pathlib import Path

import jieba

# Load custom MEP terminology dictionary so jieba preserves compound terms
_dict_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "mep_terms_dict.txt"  # noqa: F811
if _dict_path.exists():
    jieba.load_userdict(str(_dict_path))

from ..core.db import get_db, COLLECTION
from ..core.embedding import embed_query

ROOT = Path(__file__).resolve().parent.parent.parent.parent

_TEST_MODE = os.environ.get("MECH_KB_TEST_MODE", "").lower() in ("1", "true", "yes")
if _TEST_MODE:
    BM25_FILE = ROOT / "data" / "chunks" / "bm25_index_test.pkl"
    CHUNKS_FILE = ROOT / "data" / "chunks" / "test_bracket_chunks.jsonl"
else:
    BM25_FILE = ROOT / "data" / "chunks" / "bm25_index_v3.pkl"
    CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks_v3.jsonl"

RRF_K = 20
VECTOR_WEIGHT = 0.3
BM25_WEIGHT = 0.7

# ── Query synonym expansion ────────────────────────────────────────────────
# When a query contains a key, the value is appended to improve recall.
# Format: { "口语/别名": "标准术语" }
#
# Building-type synonyms are critical: BGE-M3 embeddings cannot distinguish
# "一类高层" from "二类高层" (vectors are too close). Expanding to canonical
# forms like "一类高层住宅" gives BM25 a surface-level exact-match signal
# that overrides the embedding ambiguity for fine-grained classification.
QUERY_SYNONYMS: dict[str, str] = {
    "户标段需要系数": "住宅户内需要系数",
    # Building-type canonical expansions
    "一类高层": "一类高层住宅",
    "二类高层": "二类高层住宅",
    "超高层": "超高层住宅",
    "多层住宅": "多层住宅",
    "多层": "多层住宅",
    # ── Electrical terminology aliases ──────────────────────────────────
    # Doc frequencies justifying each pair (key → value):
    #   浪涌2→电涌18  消控室2→消防控制室35  需用系数0→需要系数15
    #   不间断电源1→UPS30  电缆桥架2→桥架41  户箱1→配电箱48
    #   竖井4→电井108  避雷6→防雷接地25  SPD9→电涌保护器(in电涌18)
    #   消防栓0→消火栓4
    "浪涌保护器": "电涌保护器",
    "浪涌": "电涌",
    "避雷": "防雷接地",
    "需用系数": "需要系数",
    "消控室": "消防控制室",
    "户箱": "配电箱",
    "竖井": "电井",
    "不间断电源": "UPS",
    "电缆桥架": "桥架",
    "SPD": "电涌保护器",
    "消防栓": "消火栓",
    # ── Internal terminology mappings ─────────────────────────────────────
    # Knowledge base uses "综合仪表" for power monitoring (metering points
    # table in 强电设计任务书模板), not "电力监控系统".
    "电力监控": "综合仪表",
    # ── Distribution data aliases ───────────────────────────────────────
    # Knowledge base uses "配电数据" and "管线" in the charging-pile
    # distribution table (chunk 59: 表10 充电桩常用配电数据).
    "配电回路": "配电数据",
    "配电规格": "配电数据",
    "线缆": "管线",
    "线缆选择": "管线选择",
}

# Load additional synonyms from file if present
_syn_path = ROOT / "data" / "query_synonyms.txt"
if _syn_path.exists():
    for line in _syn_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            QUERY_SYNONYMS[k.strip()] = v.strip()


def _expand_query(query: str) -> str:
    """Append synonym terms to the query for better recall."""
    added = []
    for alias, canonical in QUERY_SYNONYMS.items():
        if alias in query and canonical not in query:
            added.append(canonical)
    if added:
        return query + " " + " ".join(added)
    return query

# LLM rerank: how many candidates to rerank
RERANK_CANDIDATES = 15

# Boost relevant content types, penalize generic/template ones.
# 参数表(68 chunks) get a moderate lift to ensure tables enter top-k context.
CONTENT_TYPE_BOOST = {}

# Lightweight image boost: chunks with images get a slight RRF advantage.
# Compensates for the missing CN-CLIP vector search (Phase 2 was cancelled).
IMAGE_CHUNK_BOOST = 1.08

# File-title boosting: when query keywords match source filenames, boost those chunks
FILE_BOOST_MAX = 20.0  # Maximum boost multiplier for strong filename match

# Section-path boosting: when query keywords match section_path, boost that chunk.
# This is critical for distinguishing "一类高层" from "二类高层" etc. — BGE-M3
# embeddings are too close for these, so surface-level keyword match is the tiebreaker.
SECTION_BOOST_MAX = 5.0
_QUERY_STOP_WORDS = {
    "怎么", "如何", "什么", "哪些", "为什么", "这个", "那个", "一下", "吗", "呢", "吧",
    # High-DF technical noise — hit nearly all 387 chunks, zero discriminative power
    "系统", "设计", "要求", "标准", "规范", "规定", "做法",
    "配置", "考虑", "应", "宜", "可", "不应", "不宜", "不得",
    "是否", "怎样", "什么样", "如何设计",
}

# Cache unique filenames for file-title boost (filenames don't change at runtime)
_cached_filenames: list[str] | None = None

_bm25 = None
_tokenized_corpus = None
_chunks = None
_reranker = None


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


def _build_expr(specialty_filter: list[str] | None, content_type_filter: list[str] | None) -> str | None:
    parts = []
    if specialty_filter:
        quoted = ", ".join(f'"{s}"' for s in specialty_filter)
        parts.append(f"specialty in [{quoted}]")
    if content_type_filter:
        quoted = ", ".join(f'"{c}"' for c in content_type_filter)
        parts.append(f"content_type in [{quoted}]")
    return " && ".join(parts) if parts else None


def _filter_bm25_indices(
    indices: list[int],
    specialty_filter: list[str] | None,
    content_type_filter: list[str] | None,
) -> list[int]:
    if not specialty_filter and not content_type_filter:
        return indices
    chunks = _get_chunks()
    return [
        idx for idx in indices
        if (not specialty_filter or chunks[idx].get("specialty") in specialty_filter)
        and (not content_type_filter or chunks[idx].get("content_type") in content_type_filter)
    ]


def _compute_file_boosts(query: str) -> dict[str, float]:
    """Boost source files whose names overlap with query keywords.

    Returns {filename: boost_multiplier}. Range [1.0, FILE_BOOST_MAX].
    Files with no keyword overlap get 1.0 (no boost).
    """
    global _cached_filenames

    # Extract meaningful query keywords (2+ char, filter stop words)
    query_words = [
        w for w in jieba.cut(query)
        if len(w.strip()) >= 2 and w.strip() not in _QUERY_STOP_WORDS
    ]
    if not query_words:
        return {}

    # Cache unique filenames (they don't change at runtime)
    if _cached_filenames is None:
        chunks = _get_chunks()
        _cached_filenames = list({c.get("source_file", "") for c in chunks if c.get("source_file")})

    boosts: dict[str, float] = {}
    for fname in _cached_filenames:
        if not fname:
            continue
        fname_clean = fname.replace(".md", "")
        # Count query keywords that appear in the filename
        matched = sum(1 for w in query_words if w in fname_clean)
        if matched > 0:
            ratio = matched / len(query_words)
            boosts[fname] = 1.0 + ratio * (FILE_BOOST_MAX - 1.0)

    return boosts


def _get_reranker():
    """Lazy-load BGE-Reranker (1.1GB, loaded once)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        import os
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        model_path = str(ROOT / "models" / "BAAI" / "bge-reranker-v2-m3")
        _reranker = CrossEncoder(model_path, local_files_only=True)
    return _reranker


def _bge_rerank(query: str, candidates: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
    """Use BGE-Reranker to re-score top candidates with cross-attention."""
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
    """Label each chunk as 'direct' or 'indirect'.

    Top-ranked is always 'direct'. Others are 'direct' only if their score
    is within 10% of the max — this keeps closely-related chunks (e.g. intro
    + specific section, or 2~3户 + 4~6户 variants) in 'direct' while clearly
    different chunks (different building types, different files) stay indirect.
    """
    if not chunks:
        return chunks

    chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
    max_score = chunks[0]["relevance_score"]
    threshold = max_score * 0.9

    for c in chunks:
        if c["relevance_score"] >= threshold:
            c["relevance"] = "direct"
        else:
            c["relevance"] = "indirect"

    return chunks


def retrieve(
    query: str,
    top_k: int = 10,
    specialty_filter: list[str] | None = None,
    content_type_filter: list[str] | None = None,
) -> list[dict]:
    """Hybrid search: vector + BM25 → RRF fusion."""
    db = get_db()
    col = db.get_collection(COLLECTION)
    col.load()
    expr = _build_expr(specialty_filter, content_type_filter)

    # Expand query with synonyms for better recall
    query = _expand_query(query)

    # ── Vector search ──────────────────────────────
    query_vec = embed_query(query)
    vector_hits = col.search(
        query_vectors=[query_vec],
        top_k=top_k * 5,  # wider pool → better recall for distinguishable chunks
        anns_field="embedding",
        output_fields=["chunk_index", "content", "specialty", "source_file", "section_path", "content_type", "has_image", "image_ids"],
        expr=expr,
    )[0]

    # ── BM25 search ────────────────────────────────
    bm25, _tokenized_corpus = _get_bm25()
    tokenized_query = [w for w in jieba.cut(query) if w.strip() and w not in _QUERY_STOP_WORDS]
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranked_indices = _filter_bm25_indices(bm25_ranked_indices, specialty_filter, content_type_filter)
    bm25_ranked_indices = bm25_ranked_indices[:top_k * 5]

    # ── RRF fusion ─────────────────────────────────
    # Key: chunk_index → accumulated RRF score
    rrf: dict[int, float] = {}

    for rank, hit in enumerate(vector_hits):
        ci = hit["entity"]["chunk_index"]
        rrf[ci] = rrf.get(ci, 0) + VECTOR_WEIGHT / (RRF_K + rank + 1)

    for rank, ci in enumerate(bm25_ranked_indices):
        rrf[ci] = rrf.get(ci, 0) + BM25_WEIGHT / (RRF_K + rank + 1)

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k * 4]

    # ── File-level diversity: top-5 files, max 2 chunks per file ──
    chunks = _get_chunks()
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

    # ── LLM Rerank: re-score top candidates for fine-grained relevance ──
    if len(fused) > 3:
        fused = _bge_rerank(query, fused, top_k)

    # ── Build result dicts ─────────────────────────
    chunks = _get_chunks()
    results = []
    for ci, score in fused:
        c = chunks[ci]
        results.append({
            "chunk_index": ci,
            "content": c["content"],
            "specialty": c["specialty"],
            "source_file": c["source_file"],
            "section_path": c.get("section_path", ""),
            "content_type": c["content_type"],
            "split_strategy": c.get("split_strategy", ""),
            "relevance_score": round(score, 6),
            "has_image": c.get("has_image", False),
            "image_ids": c.get("image_ids", ""),
            "relevance": "",  # placeholder, filled by _classify_relevance
        })

    return _classify_relevance(results, query)

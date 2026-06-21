"""Stage 3: Embed chunks with BGE-M3 and index into Milvus Lite + BM25.
Output: data/milvus_lite_v3/ + data/chunks/bm25_index.pkl + data/chunks/vectors.pkl
"""

import json
import os
import pickle
from pathlib import Path

import jieba

from milvus_lite import MilvusLite, CollectionSchema, FieldSchema, DataType
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
MILVUS_DIR = str(ROOT / "data" / "milvus_lite_v3")
BM25_FILE = ROOT / "data" / "chunks" / "bm25_index.pkl"
VECTORS_FILE = ROOT / "data" / "chunks" / "vectors.pkl"

COLLECTION = "rag_agent_chunks"
DIM = 1024
INSERT_BATCH = 200


def tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def _load_model():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(ROOT / "models" / "BAAI" / "bge-m3"), local_files_only=True)


def embed_batch(texts: list[str]) -> list[list[float]]:
    # Try cached vectors
    if VECTORS_FILE.exists():
        with open(VECTORS_FILE, "rb") as f:
            cached_texts, cached_vectors = pickle.load(f)
        if len(cached_texts) == len(texts) and all(a == b for a, b in zip(cached_texts, texts)):
            print(f"  Loaded {len(cached_vectors)} cached vectors")
            return cached_vectors
        print(f"  Cache mismatch, re-embedding")

    model = _load_model()
    SUB_BATCH = 50
    all_vectors: list[list[float]] = []
    print(f"  Encoding {len(texts)} texts in sub-batches of {SUB_BATCH} ...")
    for start in range(0, len(texts), SUB_BATCH):
        end = min(start + SUB_BATCH, len(texts))
        chunk = texts[start:end]
        vecs = model.encode(chunk, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        all_vectors.extend(vecs.tolist())
        print(f"    [{start:4d}-{end:4d}] done")

    with open(VECTORS_FILE, "wb") as f:
        pickle.dump((texts, all_vectors), f)
    print(f"  Vectors cached to {VECTORS_FILE}")
    return all_vectors


def build_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]
    return CollectionSchema(fields=fields)


def main():
    # 1. Load chunks
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # 2. Embed
    texts = []
    for c in chunks:
        ct = c.get("content_type", "")
        sp = c.get("section_path", "")
        body = c.get("enriched_content", c.get("content", ""))
        prefix = f"{ct}：" if ct else ""
        sf = c.get("source_file", "").replace(".md", "")
        src = f" | 来源: {sf}" if sf else ""
        texts.append(f"{prefix}{sp}{src}\n\n{body}" if sp else f"{prefix}{src}\n\n{body}")
    print(f"Embedding {len(texts)} texts with BGE-M3 (1024d) ...")
    vectors = embed_batch(texts)
    print(f"  {len(vectors)} vectors ({len(vectors[0])}d)")

    # 3. BM25 index
    print("Building BM25 index (jieba) ...")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    BM25_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_FILE, "wb") as f:
        pickle.dump((bm25, tokenized), f)
    print(f"  BM25 index saved to {BM25_FILE}")

    # 4. Milvus
    db = MilvusLite(data_dir=MILVUS_DIR)
    if db.has_collection(COLLECTION):
        db.drop_collection(COLLECTION)
        print(f"Dropped existing collection '{COLLECTION}'")

    col = db.create_collection(name=COLLECTION, schema=build_schema())
    print(f"Created collection '{COLLECTION}' ({DIM}d, COSINE)")

    # 5. Insert
    print(f"Inserting {len(chunks)} chunks ...")
    data = []
    for i, c in enumerate(chunks):
        data.append({
            "content": c["content"],
            "embedding": vectors[i],
            "category": c.get("category", "default"),
            "source_file": c["source_file"],
            "section_path": c.get("section_path", "")[:8000],
            "content_type": c["content_type"],
            "chunk_index": i,
        })

    total = 0
    for i in range(0, len(data), INSERT_BATCH):
        batch = data[i:i + INSERT_BATCH]
        col.insert(batch)
        total += len(batch)
        print(f"  {total}/{len(data)}")

    # 6. Index
    print("Creating vector index ...")
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    col.load()
    print("Collection loaded and ready.")

    stats = db.get_collection_stats(COLLECTION)
    print(f"\nCollection stats: {stats}")
    print(f"Milvus ready at: {MILVUS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

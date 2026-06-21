"""Shared Milvus Lite connection (singleton)."""

from pathlib import Path
from milvus_lite import MilvusLite

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MILVUS_DIR = str(ROOT / "data" / "milvus_lite_v3")
COLLECTION = "rag_agent_chunks"

_db: MilvusLite | None = None


def get_db() -> MilvusLite:
    global _db
    if _db is None:
        _db = MilvusLite(data_dir=MILVUS_DIR)
    return _db


def close_db():
    global _db
    if _db is not None:
        _db.close()
        _db = None

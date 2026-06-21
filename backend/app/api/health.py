from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health")
async def health():
    try:
        from ..core.db import get_db, COLLECTION
        db = get_db()
        stats = db.get_collection_stats(COLLECTION) if db.has_collection(COLLECTION) else {}
        return {"status": "ok", "milvus": {"collection": COLLECTION, "rows": stats.get("row_count", 0)}}
    except Exception:
        return {"status": "ok", "milvus": {"collection": COLLECTION, "rows": 0}}

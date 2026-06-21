"""Document upload and management."""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".docx", ".pdf", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".doc"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}。支持: {', '.join(ALLOWED_EXTENSIONS)}")

    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "status": "ok",
        "filename": file.filename,
        "stored_as": safe_name,
        "size": len(content),
        "category": _detect_category(file.filename or ""),
    }


@router.get("")
async def list_documents():
    if not UPLOAD_DIR.exists():
        return {"documents": []}
    docs = []
    for fp in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if fp.suffix.lower() in ALLOWED_EXTENSIONS:
            original = fp.name.split("_", 1)[1] if "_" in fp.name else fp.name
            docs.append({
                "filename": original,
                "stored_as": fp.name,
                "size": fp.stat().st_size,
                "uploaded_at": fp.stat().st_mtime,
            })
    return {"documents": docs}


@router.delete("/{stored_as}")
async def delete_document(stored_as: str):
    fp = UPLOAD_DIR / stored_as
    if not fp.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    fp.unlink()
    return {"status": "deleted"}


def _detect_category(filename: str) -> str:
    """Guess a category from filename. User can override later."""
    return "default"

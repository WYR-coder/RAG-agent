"""Pipeline trigger — parse → chunk → index → audit."""

import json
import subprocess
import sys
import time
import threading
from pathlib import Path

from fastapi import APIRouter

from ..models.schemas import PipelineStatus

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PIPELINES_DIR = Path(__file__).resolve().parent.parent.parent / "pipelines"

_pipeline_status = PipelineStatus(stage="idle")

# Audit runs inline after indexing completes
_AUDIT_AVAILABLE = True


def _run_audit() -> dict:
    """Run pipeline audit and return report."""
    try:
        from ..services.auditor import PipelineAuditor
        auditor = PipelineAuditor(
            chunks_file=ROOT / "data" / "chunks" / "chunks.jsonl",
            milvus_dir=str(ROOT / "data" / "milvus_lite_v3"),
            bm25_file=ROOT / "data" / "chunks" / "bm25_index.pkl",
            collection="rag_agent_chunks",
        )
        report = auditor.run_all_checks()
        return report.model_dump()
    except Exception as e:
        return {"passed": False, "checks": [], "error": str(e)}


def _run_pipeline():
    global _pipeline_status

    try:
        _pipeline_status = PipelineStatus(stage="parsing", progress=0.0, message="开始解析文档...")
        result = subprocess.run(
            [sys.executable, str(PIPELINES_DIR / "parse_docs.py")],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT),
        )
        if result.returncode != 0:
            _pipeline_status = PipelineStatus(stage="failed", progress=0.3, message=f"解析失败: {result.stderr[:200]}")
            return

        _pipeline_status = PipelineStatus(stage="chunking", progress=0.3, message="开始文档分块...")
        result = subprocess.run(
            [sys.executable, str(PIPELINES_DIR / "chunk_all.py")],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
        if result.returncode != 0:
            _pipeline_status = PipelineStatus(stage="failed", progress=0.5, message=f"分块失败: {result.stderr[:200]}")
            return

        _pipeline_status = PipelineStatus(stage="indexing", progress=0.6, message="开始向量化与索引...")
        result = subprocess.run(
            [sys.executable, str(PIPELINES_DIR / "index_chunks.py")],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT),
        )
        if result.returncode != 0:
            _pipeline_status = PipelineStatus(stage="failed", progress=0.8, message=f"索引失败: {result.stderr[:200]}")
            return

        # Invalidate retriever caches after re-indexing
        from ..services.retriever import invalidate_cache
        invalidate_cache()

        # ── Fixed: run audit after every pipeline completion ──
        _pipeline_status = PipelineStatus(stage="auditing", progress=0.9, message="全链路审计中...")
        audit_report = _run_audit()

        _pipeline_status = PipelineStatus(
            stage="completed", progress=1.0,
            message="知识库构建完成",
            audit=audit_report,
        )
    except Exception as e:
        _pipeline_status = PipelineStatus(stage="failed", progress=0.0, message=str(e))


@router.post("/run")
async def run_pipeline():
    global _pipeline_status
    if _pipeline_status.stage not in ("idle", "completed", "failed"):
        return {"status": "busy", "current": _pipeline_status.stage}

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    return {"status": "started"}


@router.get("/status", response_model=PipelineStatus)
async def pipeline_status():
    return _pipeline_status

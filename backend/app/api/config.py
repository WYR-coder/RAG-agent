"""API configuration and key management."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models.schemas import KeyConfig

router = APIRouter(prefix="/api/config", tags=["config"])

_ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"


@router.post("/keys")
async def save_keys(config: KeyConfig):
    """Save API keys to .env file."""
    try:
        lines = _ENV_PATH.read_text(encoding="utf-8") if _ENV_PATH.exists() else ""
        updated: dict[str, str] = {}
        for line in lines.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _ = line.split("=", 1)
                updated[k.strip()] = line
        # Update keys
        if config.llm_api_key:
            updated["LLM_API_KEY"] = f"LLM_API_KEY={config.llm_api_key}"
            settings.llm_api_key = config.llm_api_key
        if config.zhipu_api_key:
            updated["ZHIPU_API_KEY"] = f"ZHIPU_API_KEY={config.zhipu_api_key}"
            settings.zhipu_api_key = config.zhipu_api_key
        if config.volc_api_key:
            updated["VOLC_API_KEY"] = f"VOLC_API_KEY={config.volc_api_key}"
            settings.volc_api_key = config.volc_api_key
        _ENV_PATH.write_text("\n".join(updated.values()) + "\n", encoding="utf-8")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-llm")
async def test_llm():
    """Test LLM connection with a simple ping."""
    try:
        from ..core.llm import chat
        resp = chat([{"role": "user", "content": "回复 OK"}], temperature=0.1)
        return {"status": "ok", "response": resp[:50]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/status")
async def config_status():
    """Check which models and keys are configured."""
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "BAAI"
    return {
        "llm_configured": bool(settings.llm_api_key),
        "zhipu_configured": bool(settings.zhipu_api_key),
        "volc_configured": bool(settings.volc_api_key),
        "bge_m3_downloaded": (models_dir / "bge-m3" / "pytorch_model.bin").exists(),
        "reranker_downloaded": (models_dir / "bge-reranker-v2-m3" / "model.safetensors").exists(),
    }

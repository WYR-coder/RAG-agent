"""Dictionary management — user-defined terms, synonyms, desensitize rules."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..models.schemas import DictionaryUpdate
from ..services.retriever import reload_config
from ..services.desensitizer import reload_rules

router = APIRouter(prefix="/api/dictionary", tags=["dictionary"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data"


@router.get("")
async def get_dictionaries():
    """Get current dictionary contents."""
    result = {}
    for name in ["terms_dict.txt", "query_synonyms.txt", "desensitize_rules.json"]:
        fp = DATA / name
        result[name] = fp.read_text(encoding="utf-8") if fp.exists() else ""
    return result


@router.post("")
async def update_dictionary(req: DictionaryUpdate):
    """Update one or more dictionary files."""
    try:
        if req.terms is not None:
            (DATA / "terms_dict.txt").write_text(req.terms, encoding="utf-8")
        if req.synonyms is not None:
            (DATA / "query_synonyms.txt").write_text(req.synonyms, encoding="utf-8")
        if req.desensitize is not None:
            (DATA / "desensitize_rules.json").write_text(req.desensitize, encoding="utf-8")

        # Reload in running services
        reload_config()
        reload_rules()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

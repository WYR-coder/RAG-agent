"""Desensitize sensitive terms using configurable rules loaded from disk."""

import json
import re
from pathlib import Path


_RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "desensitize_rules.json"


def _load_rules() -> list[tuple[str, str]]:
    if not _RULES_PATH.exists():
        return []
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # Sort by length descending so longer terms match first
            return sorted(
                [(r["pattern"], r["replacement"]) for r in data if "pattern" in r and "replacement" in r],
                key=lambda x: -len(x[0]),
            )
        return []
    except Exception:
        return []


_SENSITIVE_TERMS = _load_rules()


def reload_rules():
    global _SENSITIVE_TERMS
    _SENSITIVE_TERMS = _load_rules()


def desensitize_text(text: str) -> str:
    for old, new in _SENSITIVE_TERMS:
        text = text.replace(old, new)
    return text


def normalize_dirname(name: str) -> str:
    for old, new in _SENSITIVE_TERMS:
        name = name.replace(old, new)
    name = re.sub(r"[★☆（）【】《》]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_.")
    if not name:
        name = "document"
    return name


def desensitize_and_normalize(text: str) -> str:
    return desensitize_text(text)

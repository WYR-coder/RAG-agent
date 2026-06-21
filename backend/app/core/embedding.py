from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "BAAI" / "bge-m3"

_model = None
_ready = False


def get_model():
    global _model, _ready
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(str(_MODEL_DIR))
        _ready = True
    return _model


def is_ready() -> bool:
    return _ready


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    results = model.encode(texts, normalize_embeddings=True)
    return results.tolist()


def embed_query(query: str) -> list[float]:
    return embed([query])[0]

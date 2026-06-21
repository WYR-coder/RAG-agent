"""RAG Agent — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting RAG Agent on {settings.backend_host}:{settings.backend_port}")
    # Ensure data directories exist
    for subdir in ["parsed", "chunks", "uploads"]:
        (Path(__file__).resolve().parent.parent.parent / "data" / subdir).mkdir(parents=True, exist_ok=True)
    yield
    from .core.db import close_db
    close_db()
    logger.info("RAG Agent shut down")


app = FastAPI(title="RAG Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────
from .api.health import router as health_router
from .api.chat import router as chat_router
from .api.config import router as config_router
from .api.dictionary import router as dictionary_router
from .api.documents import router as documents_router
from .api.pipeline import router as pipeline_router

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(config_router)
app.include_router(dictionary_router)
app.include_router(documents_router)
app.include_router(pipeline_router)

# ── Static files (parsed markdown images) ────────────────────────────────
_IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "parsed"
if _IMAGES_DIR.exists():
    app.mount("/api/images", StaticFiles(directory=str(_IMAGES_DIR)), name="images")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)

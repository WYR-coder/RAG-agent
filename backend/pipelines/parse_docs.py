"""Stage 1: Parse uploaded documents to Markdown.

Walks data/uploads/ and routes each file to the appropriate parser.
Output: data/parsed/{category}/*.md + images
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "parsed"

# Import parser from the app package
sys.path.insert(0, str(ROOT / "backend"))
from app.services.parser import DocumentParser


def main():
    if not UPLOAD_DIR.exists() or not list(UPLOAD_DIR.glob("*")):
        logger.error("No files in data/uploads/. Please upload documents first.")
        sys.exit(1)

    parser = DocumentParser(use_kreuzberg=True)
    manifest = []
    files = sorted(UPLOAD_DIR.iterdir())

    for fp in files:
        ext = fp.suffix.lower()
        if ext not in {".docx", ".pdf", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".doc"}:
            continue

        # Determine category from filename prefix or default
        category = "default"
        stem = fp.stem

        out_dir = OUTPUT_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)
        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)

        logger.info(f"Parsing: {fp.name} ({category})")
        parser.set_image_context(img_dir, stem)

        result = parser.parse(fp)
        result["category"] = category

        # Write markdown output
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(result["content"], encoding="utf-8")
        result["output_path"] = str(md_path)
        manifest.append(result)
        logger.info(f"  -> {md_path} ({result['char_count']} chars, parser={result['parser']})")

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Done. {len(manifest)} files parsed -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

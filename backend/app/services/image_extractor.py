"""Extract images from DOCX/PPTX/PDF and describe with vision model."""

import base64
import io
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import fitz
from zhipuai import ZhipuAI

from ..config import settings

logger = logging.getLogger(__name__)

MIN_IMAGE_SIZE = 8192
MAX_IMAGES_PER_FILE = 5

_volc_client = None


def _get_volc_client():
    global _volc_client
    if _volc_client is None:
        from openai import OpenAI
        _volc_client = OpenAI(api_key=settings.volc_api_key, base_url=settings.volc_base_url)
    return _volc_client


class ImageExtractor:
    def __init__(self, model: str = "doubao-seed-2-0-pro-260215", skip_vision: bool = False):
        self.model = model
        self.skip_vision = skip_vision
        self.client = ZhipuAI(api_key=settings.zhipu_api_key) if not skip_vision else None

    def process(self, file_path: Path, output_dir: Path) -> dict:
        ext = file_path.suffix.lower()
        if ext == ".docx":
            images = self._extract_docx(file_path, output_dir)
        elif ext == ".pptx":
            images = self._extract_pptx(file_path, output_dir)
        elif ext == ".pdf":
            images = self._extract_pdf(file_path, output_dir)
        else:
            return {"images": [], "descriptions": [], "is_image_only": False}

        images.sort(key=lambda x: x["size"], reverse=True)
        is_image_only = ext == ".pdf" and images and images[0].get("is_page_render")

        described = []
        if not self.skip_vision:
            to_describe = [
                img for img in images
                if img["size"] >= MIN_IMAGE_SIZE
                and not img.get("is_page_render")
                and img["ext"] not in self.UNSUPPORTED_IMAGE_EXTS
            ]
            to_describe = to_describe[:MAX_IMAGES_PER_FILE]
            for img in to_describe:
                desc = self._describe_image(img["path"], img["ext"])
                if desc:
                    img["description"] = desc
                    described.append(img)

        page_descriptions = []
        if is_image_only:
            page_images = [img for img in images if img.get("is_page_render")]
            for img in page_images:
                desc = self._describe_page(img["path"])
                if desc:
                    page_descriptions.append({
                        "page": img["page"], "filename": img["filename"], "description": desc,
                    })

        return {
            "images": images, "described": described, "page_descriptions": page_descriptions,
            "total_count": len(images), "described_count": len(described), "is_image_only": is_image_only,
        }

    def _extract_docx(self, docx_path: Path, output_dir: Path) -> list[dict]:
        images = []
        try:
            with zipfile.ZipFile(docx_path, "r") as zf:
                media_names = [n for n in zf.namelist() if n.startswith("word/media/") and not n.endswith("/")]
                for name in sorted(media_names):
                    data = zf.read(name)
                    fname = Path(name).name
                    out_path = output_dir / fname
                    counter = 1
                    stem, ext = os.path.splitext(fname)
                    while out_path.exists():
                        out_path = output_dir / f"{stem}_{counter}{ext}"
                        counter += 1
                    out_path.write_bytes(data)
                    images.append({
                        "filename": out_path.name, "path": str(out_path),
                        "size": len(data), "ext": Path(fname).suffix.lower(), "source": name,
                    })
        except Exception as e:
            logger.warning(f"DOCX image extract failed for {docx_path.name}: {e}")
        return images

    def _extract_pptx(self, pptx_path: Path, output_dir: Path) -> list[dict]:
        images = []
        try:
            with zipfile.ZipFile(pptx_path, "r") as zf:
                media_names = [n for n in zf.namelist() if n.startswith("ppt/media/") and not n.endswith("/")]
                for name in sorted(media_names):
                    data = zf.read(name)
                    fname = Path(name).name
                    out_path = output_dir / fname
                    stem, ext = os.path.splitext(fname)
                    counter = 1
                    while out_path.exists():
                        out_path = output_dir / f"{stem}_{counter}{ext}"
                        counter += 1
                    out_path.write_bytes(data)
                    images.append({
                        "filename": out_path.name, "path": str(out_path),
                        "size": len(data), "ext": Path(fname).suffix.lower(), "source": name,
                    })
        except Exception as e:
            logger.warning(f"PPTX image extract failed for {pptx_path.name}: {e}")
        return images

    def _extract_pdf(self, pdf_path: Path, output_dir: Path) -> list[dict]:
        images = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_idx, page in enumerate(doc):
                embedded = page.get_images(full=True)
                for img_idx, img_info in enumerate(embedded):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                    except Exception:
                        continue
                    data = base_image["image"]
                    ext = base_image["ext"]
                    w, h = base_image.get("width", 0), base_image.get("height", 0)
                    if len(data) < 256:
                        continue
                    fname = f"page{page_idx+1}_img{img_idx+1}.{ext}"
                    out_path = output_dir / fname
                    stem = f"page{page_idx+1}_img{img_idx+1}"
                    counter = 1
                    while out_path.exists():
                        out_path = output_dir / f"{stem}_{counter}.{ext}"
                        counter += 1
                    out_path.write_bytes(data)
                    images.append({
                        "filename": out_path.name, "path": str(out_path),
                        "size": len(data), "ext": f".{ext}",
                        "width": w, "height": h, "page": page_idx + 1,
                        "is_page_render": False, "source": f"pdf page {page_idx+1}",
                    })

            total_text = sum(len(page.get_text().strip()) for page in doc)
            if total_text < 100 and len(images) > 0 and len(doc) > 0:
                page_renders = self._render_pdf_pages(doc, pdf_path, output_dir)
                images.extend(page_renders)
            doc.close()
        except Exception as e:
            logger.warning(f"PDF image extract failed for {pdf_path.name}: {e}")
        return images

    def _render_pdf_pages(self, doc, pdf_path: Path, output_dir: Path) -> list[dict]:
        renders = []
        try:
            for page_idx, page in enumerate(doc):
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat)
                fname = f"page{page_idx+1}_render.png"
                out_path = output_dir / fname
                pix.save(str(out_path))
                renders.append({
                    "filename": out_path.name, "path": str(out_path),
                    "size": os.path.getsize(str(out_path)), "ext": ".png",
                    "width": pix.width, "height": pix.height, "page": page_idx + 1,
                    "is_page_render": True, "source": f"pdf page {page_idx+1} (rendered)",
                })
        except Exception as e:
            logger.warning(f"PDF page render failed for {pdf_path.name}: {e}")
        return renders

    def _describe_image(self, image_path: str, ext: str) -> str:
        b64 = self._encode_image(image_path)
        mime = self._mime_type(ext)
        client = _get_volc_client()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=settings.volc_vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            {"type": "text", "text": (
                                "请详细描述这张图片（150-200字）：\n"
                                "1. 图片类型（表格/示意图/照片/图纸）\n"
                                "2. 涉及的系统或设备名称\n"
                                "3. 关键技术参数或尺寸标注\n"
                                "4. 适用场景或引用规范"
                            )},
                        ],
                    }],
                    max_tokens=500, temperature=0.1,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    return content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                logger.warning(f"Vision API failed for {Path(image_path).name}: {e}")
        return ""

    def _describe_page(self, image_path: str) -> str:
        b64 = self._encode_image(image_path)
        client = _get_volc_client()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=settings.volc_vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": (
                                "请详细描述这张PDF页面（技术设计文档）的内容：\n"
                                "1. 页面主题和用途\n"
                                "2. 如果有表格，列出表格的关键列名和数值\n"
                                "3. 如果有图纸/示意图，说明图示内容、标注、尺寸\n"
                                "4. 所有关键技术参数和规范编号\n"
                                "请用中文回答，控制在200字以内。"
                            )},
                        ],
                    }],
                    max_tokens=800, temperature=0.1,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    return content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                logger.warning(f"Page describe failed for {Path(image_path).name}: {e}")
        return ""

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    UNSUPPORTED_IMAGE_EXTS = {".wmf", ".emf", ".svg", ".svgz"}

    @staticmethod
    def _mime_type(ext: str) -> str:
        ext = ext.lower().lstrip(".")
        mime_map = {
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp",
            "tiff": "image/tiff", "tif": "image/tiff",
        }
        return mime_map.get(ext, f"image/{ext}")

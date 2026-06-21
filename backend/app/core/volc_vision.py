"""Volcano Engine Ark vision client for on-the-fly image analysis."""
import base64
import logging
from pathlib import Path

from openai import OpenAI
from ..config import settings

logger = logging.getLogger(__name__)

_client = None
_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "parsed"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.volc_api_key, base_url=settings.volc_base_url)
    return _client


def analyze_image(image_path: str, query: str, category: str = "") -> str:
    full_path = _DATA_ROOT / category / "images" / image_path.replace("/", "\\") if category else Path(image_path)
    if not full_path.is_file():
        logger.warning(f"Image not found: {full_path}")
        return ""

    try:
        b64 = base64.b64encode(full_path.read_bytes()).decode()
        ext = full_path.suffix.lower().lstrip(".")
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else 'png'}"

        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.volc_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": (
                        f"用户正在查询：「{query}」\n"
                        "请描述这张图片中与查询相关的内容，包括：\n"
                        "1. 图片类型和主题\n"
                        "2. 关键参数、尺寸标注或规范编号\n"
                        "3. 与用户查询直接相关的信息\n"
                        "用中文回答，控制在100字以内。"
                    )},
                ],
            }],
            max_tokens=300,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Volcano vision failed for {image_path}: {e}")
        return ""


def analyze_top_images(image_paths: list[str], query: str, category: str = "", max_images: int = 2) -> list[str]:
    results: list[str] = []
    for img_path in image_paths[:max_images]:
        analysis = analyze_image(img_path, query, category)
        if analysis:
            results.append(f"[图片分析] {Path(img_path).stem}: {analysis}")
    return results

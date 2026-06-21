#!/usr/bin/env python3
"""模型下载脚本 — ModelScope → HF 镜像 → HF 官方 三级回退。

下载内容:
  - BAAI/bge-m3 (嵌入模型, ~2.2 GB)
  - BAAI/bge-reranker-v2-m3 (重排序模型, ~2.1 GB)

使用:
  python scripts/download_models.py              # 自动选择最优源
  HF_ENDPOINT=https://hf-mirror.com python ...   # 手动指定镜像
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
]

MIRRORS = [
    ("ModelScope", "https://www.modelscope.cn"),
    ("HF Mirror", "https://hf-mirror.com"),
    ("HuggingFace", "https://huggingface.co"),
]


def _test_url(url: str, timeout: float = 3.0) -> float:
    """Test URL reachability, return latency in seconds or inf."""
    try:
        t0 = time.time()
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return time.time() - t0
    except Exception:
        return float("inf")


def _pick_mirror() -> tuple[str, str]:
    """Pick the fastest reachable mirror. Returns (name, base_url)."""
    # If user set HF_ENDPOINT, use it directly
    if os.environ.get("HF_ENDPOINT"):
        return ("Custom", os.environ["HF_ENDPOINT"])

    print("测速中，选择最快的模型下载源...")
    best = min(MIRRORS, key=lambda m: _test_url(m[1]))
    latency = _test_url(best[1])
    if latency == float("inf"):
        print("⚠ 所有镜像均不可达，使用 HuggingFace 官方")
        return ("HuggingFace", "https://huggingface.co")
    print(f"✅ 选择 {best[0]} (延迟 {latency*1000:.0f}ms)")
    return best


def download_model(model_name: str, mirror_url: str) -> bool:
    """Download a single model using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        print(f"\n下载 {model_name} ...")
        # Set HF_ENDPOINT for sentence-transformers
        if mirror_url:
            os.environ["HF_ENDPOINT"] = mirror_url

        t0 = time.time()
        SentenceTransformer(
            model_name,
            cache_folder=str(MODELS_DIR),
            trust_remote_code=True,
        )
        elapsed = time.time() - t0
        size = sum(
            f.stat().st_size
            for f in (MODELS_DIR / model_name.replace("/", "--")).rglob("*")
            if f.is_file()
        )
        print(f"  ✅ {model_name} 下载完成 ({size/1e9:.1f} GB, {elapsed:.0f}s)")
        return True
    except Exception as e:
        print(f"  ❌ {model_name} 下载失败: {e}")
        return False


def main():
    mirror_name, mirror_url = _pick_mirror()

    success = 0
    for model in MODELS:
        if download_model(model, mirror_url):
            success += 1

    print(f"\n{'='*50}")
    print(f"下载完成: {success}/{len(MODELS)} 个模型")

    if success < len(MODELS):
        print("部分模型下载失败，请检查网络后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()

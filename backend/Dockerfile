FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir fastapi uvicorn[standard] pymilvus zhipuai openai sentence-transformers rank-bm25 jieba kreuzberg sse-starlette pydantic-settings openpyxl python-docx python-pptx PyPDF2 Pillow xlrd PyMuPDF markdown

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "python /app/scripts/download_models.py && uvicorn app.main:app --host 0.0.0.0 --port 8080"]

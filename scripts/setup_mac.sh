#!/bin/bash
# RAG Agent — macOS Native Setup (Apple Silicon / Intel)
# 无需 Docker，直接在 macOS 上运行
#
# 使用: bash scripts/setup_mac.sh
#
set -e

echo "============================================"
echo "  RAG Agent — macOS 原生安装"
echo "============================================"
echo ""

# ── Check macOS ──────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "❌ 此脚本仅适用于 macOS。Linux/WSL 请使用 setup.sh + Docker。"
    exit 1
fi

ARCH=$(uname -m)
echo "检测到架构: $ARCH"
echo ""

# ── Homebrew ─────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    echo "⚠ 未检测到 Homebrew。正在安装..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Apple Silicon Homebrew path
    if [[ "$ARCH" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "✅ Homebrew 已安装"
fi

# ── System Dependencies ──────────────────────────────────────────────
echo ""
echo "安装系统依赖..."
brew install libmagic libjpeg libpng 2>/dev/null || true

# ── Python ───────────────────────────────────────────────────────────
PYTHON_MIN="3.11"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$(python3 -c "import sys; print(1 if sys.version_info >= (3,11) else 0)")" == "1" ]]; then
        echo "✅ Python $PY_VER 已安装"
    else
        echo "⚠ Python $PY_VER < $PYTHON_MIN，正在安装 Python 3.12..."
        brew install python@3.12
    fi
else
    echo "正在安装 Python 3.12..."
    brew install python@3.12
fi

# ── Node.js ──────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo "正在安装 Node.js..."
    brew install node@22
else
    echo "✅ Node.js $(node -v) 已安装"
fi

# ── API Keys ─────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  API Key 配置"
echo "============================================"
echo ""
echo "各平台 Key 申请链接："
echo "  DeepSeek:    https://platform.deepseek.com"
echo "  智谱 AI:     https://open.bigmodel.cn"
echo "  火山引擎:     https://console.volcengine.com (可选)"
echo ""

read -p "DeepSeek API Key: " LLM_API_KEY
read -p "智谱 AI API Key: " ZHIPU_API_KEY
read -p "火山引擎 API Key (可选，回车跳过): " VOLC_API_KEY

# ── Generate .env ────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cat > "$ROOT/.env" << ENDOFFILE
LLM_API_KEY=${LLM_API_KEY}
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
ZHIPU_API_KEY=${ZHIPU_API_KEY}
VOLC_API_KEY=${VOLC_API_KEY:-}
VOLC_VISION_MODEL=doubao-seed-2-0-pro-260215
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8080
MILVUS_URI=./data/milvus_lite.db
ENDOFFILE

echo ""
echo "✅ .env 已生成"

# ── Python Virtual Environment ───────────────────────────────────────
echo ""
echo "设置 Python 虚拟环境..."
cd "$ROOT"
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# ── Install Python Dependencies ──────────────────────────────────────
echo ""
echo "安装 Python 依赖..."
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pymilvus zhipuai openai \
    "sentence-transformers>=3.0.0" "rank-bm25>=0.2.2" jieba \
    kreuzberg "sse-starlette>=2.0.0" "pydantic-settings>=2.5.0" \
    openpyxl python-docx python-pptx PyPDF2 Pillow xlrd PyMuPDF markdown

# ── Install Frontend Dependencies ────────────────────────────────────
echo ""
echo "安装前端依赖..."
cd "$ROOT/frontend"
npm install

# ── Download Models ──────────────────────────────────────────────────
echo ""
echo "下载嵌入模型 (BGE-M3 + BGE-Reranker-v2-M3)..."
cd "$ROOT"
source .venv/bin/activate
python scripts/download_models.py

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  🎉 安装完成！"
echo "============================================"
echo ""
echo "启动方式 (开两个终端):"
echo ""
echo "  终端 1 — 后端:"
echo "    cd $ROOT"
echo "    source .venv/bin/activate"
echo "    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"
echo ""
echo "  终端 2 — 前端:"
echo "    cd $ROOT/frontend"
echo "    npm run dev"
echo ""
echo "然后打开浏览器访问: http://localhost:3000"
echo ""
echo "或使用 Docker 方式:"
echo "  docker-compose up -d"
echo ""

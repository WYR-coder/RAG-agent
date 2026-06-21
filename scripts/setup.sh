#!/bin/bash
# RAG Agent 交互式初始化脚本
# 用于首次使用时配置 API Key

set -e

echo "============================================"
echo "  RAG Agent — 初始化配置"
echo "============================================"
echo ""
echo "此脚本将引导你配置必要的 API Key。"
echo "各平台 Key 申请链接："
echo "  DeepSeek:    https://platform.deepseek.com"
echo "  智谱 AI:     https://open.bigmodel.cn"
echo "  火山引擎:     https://console.volcengine.com (可选)"
echo ""

read -p "DeepSeek API Key: " LLM_API_KEY
read -p "智谱 AI API Key: " ZHIPU_API_KEY
read -p "火山引擎 API Key (可选，回车跳过): " VOLC_API_KEY

# 生成 .env 文件
cat > .env << ENDOFFILE
LLM_API_KEY=${LLM_API_KEY}
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
ZHIPU_API_KEY=${ZHIPU_API_KEY}
VOLC_API_KEY=${VOLC_API_KEY:-}
VOLC_VISION_MODEL=doubao-seed-2-0-pro-260215
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8080
MILVUS_URI=../data/milvus_lite_v3.db
ENDOFFILE

echo ""
echo "✅ .env 已生成"
echo ""
echo "下一步："
echo "  1. 下载模型:     python scripts/download_models.py"
echo "  2. 启动服务:     docker-compose up -d"
echo "  3. 打开浏览器:   http://localhost:3000"

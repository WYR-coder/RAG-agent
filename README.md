# RAG Agent — 可私有化部署的 RAG 知识库构建工具

基于 BGE-M3 + Milvus Lite + DeepSeek 的五层检索增强架构，支持用户上传自己的文档、定义领域词典、一键构建知识库并问答。

## 核心理念：授人以渔，不授人以鱼

本工具交付的是完整的 RAG 工具链，不包含任何预置知识内容。你需要用自己的文档、领域术语和数据来构建属于你的知识库。

## 架构

```
文档 → 解析(S1) → 分块(S2) → 嵌入+索引(S3) → 检索(S4) → 生成回答(S5)
                     ↑                                ↓
                  图文绑定                          图片/表格后注入
```

详细架构文档见 `docs/` 目录。

## 快速开始

### 前提条件
- Docker & Docker Compose (跨平台推荐)
- DeepSeek API Key（[申请地址](https://platform.deepseek.com)）
- 智谱 AI API Key（[申请地址](https://open.bigmodel.cn)）
- 火山引擎 API Key（[申请地址](https://console.volcengine.com)）— 可选，用于图片分析

### 5 分钟部署 (Docker — 全平台通用)

```bash
git clone https://github.com/WYR-coder/RAG-agent.git
cd rag-agent

# 1. 交互式配置 API Key
bash scripts/setup.sh

# 2. 下载模型并启动
docker-compose up -d
```

浏览器打开 `http://localhost:3000`，按引导完成初始设置。

### macOS 原生部署 (无需 Docker)

如果你不想安装 Docker Desktop，可以直接在 macOS 上原生运行 (支持 Apple Silicon M1/M2/M3/M4 及 Intel)：

```bash
git clone https://github.com/WYR-coder/RAG-agent.git
cd rag-agent

# 一键安装 (Homebrew + Python + Node + 依赖 + 模型下载)
bash scripts/setup_mac.sh

# 启动 — 开两个终端
# 终端 1: 后端
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 终端 2: 前端
cd frontend && npm run dev
```

浏览器打开 `http://localhost:3000`。

### 使用流程

1. **⚙ 配置** — 在设置面板填写各平台 API Key
2. **📖 词典** — 上传领域专业术语，提升分词准确性
3. **📄 文档** — 拖拽上传设计文档、规范、手册等
4. **🔨 构建** — 一键完成解析 → 分块 → 索引
5. **💬 问答** — 基于你的知识库进行 RAG 问答

## 配置说明

| 环境变量 | 说明 | 必填 |
|----------|------|------|
| `LLM_API_KEY` | DeepSeek API Key | ✅ |
| `LLM_BASE_URL` | DeepSeek API 地址 | ✅ |
| `LLM_MODEL` | 模型名称 (如 deepseek-chat) | ✅ |
| `ZHIPU_API_KEY` | 智谱 AI API Key（文档解析） | ✅ |
| `VOLC_API_KEY` | 火山引擎 API Key（图片分析） | ❌ |
| `MILVUS_URI` | Milvus Lite 数据库路径 | ❌ |
| `BACKEND_HOST` | 后端监听地址 | ❌ |
| `BACKEND_PORT` | 后端端口 | ❌ |

## 项目结构

```
rag-agent/
├── backend/                # FastAPI 后端
│   ├── app/
│   │   ├── api/            # API 端点
│   │   ├── core/           # 核心组件 (嵌入, LLM, 数据库)
│   │   ├── services/       # 业务逻辑 (检索, 解析, 审计)
│   │   └── models/         # 数据模型
│   └── pipelines/          # ETL 流水线脚本
├── frontend/               # Next.js 前端
├── scripts/                # 初始化 & 模型下载脚本
├── data/                   # 运行时数据目录
└── docs/                   # 架构文档
```

## 技术栈

- **后端**: FastAPI + Milvus Lite + BGE-M3 + BM25 + BGE-Reranker
- **前端**: Next.js + React + shadcn/ui + Tailwind CSS
- **LLM**: DeepSeek-V3 (OpenAI 兼容 API)
- **模型**: BGE-M3 (嵌入), BGE-Reranker-v2-M3 (重排序)
- **部署**: Docker Compose

## 更新日志

### v1.1.0 (2026-06-25)

**检索管线简化**
- RRF 融合常数 `K` 从 60 降至 20，排名差距扩大约 6 倍
- 删除所有后置乘法加权（文件名 ×20、章节路径 ×5、图片 ×1.08、内容类型加权），此前 RRF 分数方差仅 2%，后置乘数可达 12.4 倍，检索信号被完全淹没
- 当前管线：RRF 融合 → 文件多样性过滤 → BGE-Reranker 精排

**追问上下文污染修复**
- 追问丰富化增加原始查询长度判断：≥8 字符的完整问句不再误判为碎片
- 修复"电力监控系统的设计要求"→ 仅 1 个 token 被误判 → 拼接上一轮充电桩 query 导致检索漂移

**分块逻辑修复**
- `split_by_h2` 改为同时识别 H1 (`#`) 和 H2 (`##`) 作为块边界
- 修复 DOCX 中子系统用 Heading 1、归类标题用 Heading 2 导致 10 个子系统被吞为单一 7826 字符巨块的问题

**启动脚本**
- 新增 `backend/start_backend.py`，自动清理端口占用后启动
- 跨平台支持：Windows 使用 `netstat` + `taskkill`，macOS/Linux 使用 `lsof` + `kill`

### v1.0.0 (初始版本)
- 五层 RAG 架构：解析 → 分块 → 嵌入 → 检索 → 生成
- BGE-M3 1024d 向量 + BM25 混合检索
- Milvus Lite 嵌入式向量数据库
- DeepSeek-V3 LLM 流式生成
- Docker / macOS 原生双部署方案

## License

Apache License 2.0 — 允许商用、修改、分发。

"""RAG chat and search endpoints."""

import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import jieba

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from ..core.llm import chat_stream, chat
from ..core.volc_vision import analyze_top_images
from ..models.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, SourceCitation
from ..services.retriever import retrieve, reload_config
from ..services.desensitizer import reload_rules
import markdown

router = APIRouter()

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "parsed"
_SOURCE_DIR = _DATA_DIR

SYSTEM_PROMPT = """你是知识库的专家助手，支持多轮对话。系统已将知识库按结构组织。

对话理解：
- 如果用户的问题是对上一轮的追问、补充或关联提问，结合历史对话理解意图
- 追问时可以直接引用上一轮已给出的数据，不必重复完整上下文

输出规则：
- 上下文按匹配度降序排列
- 只回答用户提问的内容，知识库中不直接相关的片段主动略过
- 涉及数值按条件分段取值，必须用markdown表格呈现，左列条件右列数值，不要用纯列表或逗号串

准确性规则（重要）：
- 数值区间匹配：知识库中数据常按区间给出，当用户查询的具体值落在该区间内，直接引用该区间数据
- 禁止跨分类推测：不同分类是不同的标准，不可互相替代
- 禁止无中生有：知识库中不存在的数据不得编造

模块输出格式：
- 上下文中有多个模块，选择与用户问题最匹配的一个作为主回答
- 其他模块如与问题相关，以"---"分隔线独立列出
- 每个模块的文字、表格、图片属于该模块自身，禁止跨模块拼合

图片规则：
- 不要输出任何图片或图片占位符；图片由系统自动附加

引用规则：
- 引用具体数据和尺寸，先给出直接答案
- 同一源文件的多个片段合并引用
"""

_CONTENT_STOP_WORDS = frozenset({
    "的", "是", "在", "有", "和", "与", "不", "了", "就", "也", "都", "要", "把",
    "被", "让", "从", "到", "对", "为", "以", "及", "或", "等", "而", "且", "但",
    "所", "其", "之", "则", "将", "已", "还", "又", "再", "才", "只", "可",
    "怎么", "如何", "什么", "哪些", "为什么", "吗", "呢", "吧", "啊", "哦", "嗯",
    "有没有", "没有", "是否", "是不是", "能不能", "可不可以", "请问",
    "这个", "那个", "哪个", "一下", "我", "你", "可以", "需要", "应该", "怎样",
    "什么样", "参考", "图片", "附图", "大样", "大样图",
    "标准", "取值", "要求", "规范", "规定", "做法", "尺寸",
    "麻烦", "帮忙", "告诉", "说明", "解释", "介绍", "了解",
    "系统", "设计",
})


def _normalize_tildes(text: str) -> str:
    text = text.replace('～', '-')
    text = text.replace('~', '-')
    return text


def _content_tokens(query: str) -> list[str]:
    return [w for w in jieba.cut(query) if len(w) >= 2 and w.strip() and w not in _CONTENT_STOP_WORDS]


def _strip_tables_from_text(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            in_table = True
            continue
        if in_table:
            if stripped and not stripped.startswith("|"):
                in_table = False
                result.append(line)
        else:
            result.append(line)
    clean = "\n".join(result).strip()
    return re.sub(r"\n{3,}", "\n\n", clean)


def _rewrite_images(content: str, category: str) -> str:
    return re.sub(
        r"\]\(images/([^)]+)\)",
        lambda m: f"](/api/images/{quote(category)}/{quote(m.group(1), safe='/')})",
        content,
    )


def _load_source_images(chunks: list[dict], query: str = "", answer: str = "", max_total: int = 8) -> tuple[list[str], list[str]]:
    if not chunks:
        return [], []
    query_tokens = set(_content_tokens(query))
    answer_tokens = set(_content_tokens(answer)) if answer else set()
    answer_rich = len(answer_tokens) >= 5
    direct_imgs: list[str] = []
    indirect_imgs: list[str] = []
    seen: set[str] = set()
    for relevance in ("direct", "indirect"):
        for c in chunks:
            if c.get("relevance") != relevance:
                continue
            content = c.get("content", "")
            if query_tokens and not any(tok in content for tok in query_tokens):
                continue
            if answer_rich and not any(tok in content for tok in answer_tokens):
                continue
            image_ids_str = c.get("image_ids", "")
            if not image_ids_str:
                continue
            for img_path in image_ids_str.split(","):
                img_path = img_path.strip()
                if not img_path or img_path in seen:
                    continue
                seen.add(img_path)
                target = direct_imgs if relevance == "direct" else indirect_imgs
                target.append(f"![附图](/api/images/{quote(c.get('category', ''))}/{quote(img_path, safe='/')})")
                if len(direct_imgs) + len(indirect_imgs) >= max_total:
                    return direct_imgs, indirect_imgs
    return direct_imgs, indirect_imgs


def _extract_tables_from_chunks(chunks: list[dict], query_tokens: list[str]) -> tuple[list[str], list[str]]:
    SEP_RE = re.compile(r"\|[-:\s]+\|")
    token_set = set(query_tokens) if query_tokens else set()
    scored_direct: list[tuple[float, str]] = []
    scored_indirect: list[tuple[float, str]] = []
    for c in chunks[:5]:
        content = c.get("content", "")
        if not SEP_RE.search(content):
            continue
        chunk_score = c.get("relevance_score", 0.0)
        relevance = c.get("relevance", "direct")
        if token_set and not any(tok in content for tok in token_set):
            continue
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if not SEP_RE.match(line):
                i += 1
                continue
            start = i - 1
            while start >= 0 and "|" in lines[start]:
                start -= 1
            start += 1
            end = i + 1
            while end < len(lines) and "|" in lines[end]:
                end += 1
            table_text = "\n".join(lines[start:end]).strip()
            if not table_text or len(table_text) < 20:
                i = end
                continue
            token_hits = sum(1 for tok in token_set if tok in table_text) if token_set else 0
            if relevance == "direct":
                if chunk_score > 0:
                    scored_direct.append((chunk_score * 10 + token_hits * 0.01, table_text))
            else:
                if token_hits >= 2:
                    scored_indirect.append((token_hits + chunk_score, table_text))
            i = end

    def _dedup_top(scored: list, max_count: int) -> list[str]:
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        tables: list[str] = []
        for _score, tbl in scored:
            if tbl not in seen:
                seen.add(tbl)
                tables.append(tbl)
                if len(tables) >= max_count:
                    break
        return tables
    return _dedup_top(scored_direct, 2), _dedup_top(scored_indirect, 2)


def _build_media_sections(direct_images, indirect_images, direct_tables, indirect_tables) -> str:
    sections: list[str] = []
    if direct_images:
        sections.append("### 直接参考图片\n" + "\n".join(direct_images))
    if direct_tables:
        sections.append("### 直接参考表格\n\n" + "\n\n".join(direct_tables))
    if indirect_images:
        sections.append("### 拓展参考图片\n" + "\n".join(indirect_images))
    if indirect_tables:
        sections.append("### 拓展参考表格\n\n" + "\n\n".join(indirect_tables))
    if not sections:
        return ""
    return "\n\n---\n" + "\n\n".join(sections)


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks[:5], 1):
        content = _strip_tables_from_text(c["content"])
        content = re.sub(r"~~(.+?)~~", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"!\[([^\]]*)\]\(images/[^)]+\)", r"【附图：\1】", content)
        section = c.get("section_path", "")
        header = f"【模块{i}】章节: {section} | 来源: {c['source_file']} | 类型: {c['content_type']}"
        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def _format_sources(chunks: list[dict]) -> list[SourceCitation]:
    return [
        SourceCitation(
            source_file=c["source_file"], section_path=c.get("section_path", ""),
            content_type=c["content_type"], excerpt=c["content"][:200],
            relevance_score=c["relevance_score"], category=c.get("category", ""),
            has_image=c.get("has_image", False), image_ids=c.get("image_ids", ""),
            relevance=c.get("relevance", "direct"),
        ) for c in chunks
    ]


@router.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.perf_counter()
    results = retrieve(query=req.query, top_k=req.top_k,
                       category_filter=req.category_filter,
                       content_type_filter=req.content_type_filter)
    elapsed = (time.perf_counter() - t0) * 1000
    return SearchResponse(results=_format_sources(results), retrieval_time_ms=round(elapsed, 2))


@router.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    t0 = time.perf_counter()
    enriched_query = req.query
    if req.history:
        last_user = next((h["content"] for h in reversed(req.history) if h["role"] == "user"), "")
        if last_user:
            last_terms = set(_content_tokens(last_user))
            curr_terms = set(_content_tokens(req.query))
            new_terms = curr_terms - last_terms
            if len(curr_terms) < 3 or not new_terms:
                enriched_query = f"{last_user} {req.query}"

    chunks = retrieve(query=enriched_query, top_k=req.top_k,
                      category_filter=req.category_filter, content_type_filter=None)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    context = _build_context(chunks)

    img_keywords = {"图", "图纸", "大样", "示意图", "标注", "尺寸图", "附图"}
    if any(kw in enriched_query for kw in img_keywords):
        top_direct, top_indirect = _load_source_images(chunks, enriched_query, max_total=2)
        top_images = top_direct + top_indirect
        img_paths = [u.split("(")[1].split(")")[0] if "(" in u else "" for u in top_images]
        img_paths = [p for p in img_paths if p]
        if img_paths:
            analysis = analyze_top_images(img_paths, req.query, chunks[0].get("category", ""))
            if analysis:
                context += "\n\n[实时图片分析]\n" + "\n".join(analysis)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-20:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": f"参考以下知识库内容回答问题。\n\n--- 知识库内容 ---\n{context}\n--- 结束 ---\n\n问题：{req.query}"})

    if not req.stream:
        t1 = time.perf_counter()
        answer = chat(messages)
        answer = _normalize_tildes(answer)
        answer = re.sub(r"~~(.+?)~~", r"\1", answer, flags=re.DOTALL)
        gen_ms = (time.perf_counter() - t1) * 1000
        _qtokens = _content_tokens(enriched_query)
        direct_tbls, indirect_tbls = _extract_tables_from_chunks(chunks, _qtokens)
        direct_imgs, indirect_imgs = _load_source_images(chunks, enriched_query, answer)
        media = _build_media_sections(direct_imgs, indirect_imgs, direct_tbls, indirect_tbls)
        if media:
            answer += media
        return ChatResponse(answer=answer, sources=_format_sources(chunks),
                            retrieval_time_ms=round(retrieval_ms, 2),
                            generation_time_ms=round(gen_ms, 2))

    async def event_generator():
        t1 = time.perf_counter()
        full_answer = ""
        try:
            for token in chat_stream(messages):
                token = _normalize_tildes(token)
                token = re.sub(r"~~(.+?)~~", r"\1", token, flags=re.DOTALL)
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            _qtokens2 = _content_tokens(enriched_query)
            direct_tbls2, indirect_tbls2 = _extract_tables_from_chunks(chunks, _qtokens2)
            direct_imgs, indirect_imgs = _load_source_images(chunks, enriched_query, full_answer)
            media = _build_media_sections(direct_imgs, indirect_imgs, direct_tbls2, indirect_tbls2)
            if media and "/api/images/" not in full_answer:
                full_answer += media
                yield f"data: {json.dumps({'token': media})}\n\n"

            gen_ms = (time.perf_counter() - t1) * 1000
            sources = _format_sources(chunks)
            yield f"data: {json.dumps({'done': True, 'retrieval_time_ms': round(retrieval_ms, 2), 'generation_time_ms': round(gen_ms, 2), 'sources': [s.model_dump() for s in sources]})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@router.get("/api/source/{category}/{filename:path}")
async def view_source(category: str, filename: str):
    fp = _SOURCE_DIR / category / filename
    if not fp.is_file():
        return HTMLResponse("<h2>File not found</h2>", status_code=404)
    raw = fp.read_text(encoding="utf-8")
    html = markdown.markdown(raw, extensions=["tables", "fenced_code"])
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{filename}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.7; color: #1a1a1a; }}
h2 {{ margin-top: 2rem; border-bottom: 1px solid #e5e5e5; padding-bottom: .3rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th,td {{ border: 1px solid #ddd; padding: .5rem .75rem; text-align: left; font-size: .9rem; }}
img {{ max-width: 100%; }}
</style></head><body>{html}</body></html>""")

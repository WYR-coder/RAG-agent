"""Stage 2: Unified chunking with image binding.

Split strategy: BRACKET > TABLE > H2(### hierarchy) > CN+DIGIT > NL
Output: data/chunks/chunks.jsonl + data/chunks/image_bindings.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
OUT = ROOT / "data" / "chunks"
OUT.mkdir(parents=True, exist_ok=True)

MIN_CHUNK_LEN = 50

# ── Noise patterns ────────────────────────────────────────────────────
PAGE_SLASH = re.compile(r"^\d+\s*/\s*\d+$")
PAGE_CHAPTER = re.compile(r"^第\d+页\s*/\s*第\d+页$")
PAGE_HEADER = re.compile(r"^第\d+页")
STANDALONE_NULL = re.compile(r"^null$")
DOT_LINE = re.compile(r"^[.\s·•\-—]+$")
TOC_DOTS = re.compile(r"\.{4,}")
EXTRACTION_META = re.compile(r"^本文件(共)?提取\s*\d+\s*[张个]")
IMG_DESC_HEADER = re.compile(r"^#{1,3}\s*图片")
IMG_RENDER = re.compile(r"^!\[.*render\.png\]")
TOC_ENTRY = re.compile(r"^\d+(?:\.\d+)*\s+\S.*\d{1,3}$")
TABLE_SEP_ONLY = re.compile(r"^\|[\s\-\|]+\|\s*$")


def _is_toc_block(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for l in lines if TOC_ENTRY.match(l)) >= len(lines) * 0.6


def is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if PAGE_SLASH.match(t) or PAGE_CHAPTER.match(t) or STANDALONE_NULL.match(t) or DOT_LINE.match(t):
        return True
    if EXTRACTION_META.match(t) or IMG_RENDER.match(t) or TOC_DOTS.search(t) or TABLE_SEP_ONLY.match(t):
        return True
    if IMG_DESC_HEADER.match(t) and len(t) < 30:
        return True
    if PAGE_HEADER.match(t) and len(t) < 25:
        return True
    if _is_toc_block(t):
        return True
    return False


# ── Section detection ─────────────────────────────────────────────────
H2_HEADER = re.compile(r"^##\s+(.+)$", re.MULTILINE)
H3_HEADER = re.compile(r"^###\s+(.+)$", re.MULTILINE)
CN_SECTION = re.compile(r"^[一二三四五六七八九十]+、\s*(.+)$", re.MULTILINE)
DIGIT_SECTION = re.compile(r"^\d+(?:\.\d+)*\s+(.+)$", re.MULTILINE)
CAT_HEADER = re.compile(r"^(总则|适用范围|编制说明|前言|编制目的|使用说明|原则|依据|引用|术语|目录)$", re.MULTILINE)
CN_SPACE = re.compile(r"^[一二三四五六七八九十]+\s{1,5}([^\d\s].{1,40})$", re.MULTILINE)
BAD_SECTION = re.compile(r"图片描述|第\s*\d+\s*页|render|page\d+|^\.{4,}")
ALL_SECTIONS = [H2_HEADER, H3_HEADER, CN_SECTION, DIGIT_SECTION, CAT_HEADER, CN_SPACE]


def detect_section(lines: list[str]) -> str | None:
    if not lines:
        return None
    first = lines[0].strip()
    if TOC_DOTS.search(first) or BAD_SECTION.search(first):
        return None
    for pat in ALL_SECTIONS:
        m = pat.match(first)
        if m:
            return first
    return None


# ── Content type classification ───────────────────────────────────────
DESIGN_RULE_KW = re.compile(r"(?:不应|不得|严禁|禁止|禁用|慎用|切勿|务必|必须|应按|应设|应采用|应配置)|(?:宜[^用]|不宜)|(?:优先执行|参照执行|按.*标准执行|配置标准|原则上)")
KV_SPEC = re.compile(r"^[^\s：:]+\s*[：:]\s*\S", re.MULTILINE)
IMG_CONTENT_KW = re.compile(r"GLM-5V-Turbo|通过.*视觉模型.*描述|图片附件")


def classify_content(text: str) -> str:
    lines = text.strip().split("\n")
    if IMG_CONTENT_KW.search(text):
        return "图片描述"
    is_table = False
    pipe_count = sum(1 for line in lines if "|" in line and len(line.strip()) > 5)
    tab_count = sum(1 for line in lines if "\t" in line)
    if pipe_count >= 2 or tab_count >= 2:
        is_table = True
    kv_lines = sum(1 for line in lines if KV_SPEC.match(line.strip()))
    if kv_lines >= 4:
        is_table = True
    if "#DIV/0!" in text or "#REF!" in text:
        is_table = True
    if is_table:
        return "参数表"
    if DESIGN_RULE_KW.search(text):
        return "设计规则"
    return "说明文本"


# ── Image handling ────────────────────────────────────────────────────
IMG_REF = re.compile(r"!\[([^\]]*)\]\(images/([^)]+)\)")


def extract_images_from_md(content: str) -> list[dict]:
    return [{"description": m.group(1), "path": m.group(2), "markdown": m.group(0)} for m in IMG_REF.finditer(content)]


# ── Image inline embedding (v3.1) ─────────────────────────────────────
_CN_NUM = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "七": 6, "八": 7, "九": 8, "十": 9}
_FIG_NUM = re.compile(r"图\s*(\d+)")
_FIG_CN = re.compile(r"附图\s*([一二三四五六七八九十]+)")
_REF_DWG = re.compile(r"参考大样图|参考图|示意图|见附图|详见附图|如下图所示")
_IMG_SECTION = re.compile(r"(?:##\s+)?图片附件\s*\n.*?(?=\n##\s|\Z)", re.DOTALL)
NL = chr(10)


def _embed_images_inline(text: str) -> str:
    img_match = _IMG_SECTION.search(text)
    if not img_match:
        return text
    images = extract_images_from_md(img_match.group(0))
    if not images:
        return text
    text = _IMG_SECTION.sub("", text).rstrip()
    text = re.sub(r"\n*本文件[共包含]*\s*\d+\s*[张个]图片[。，]?\s*\n*", "\n", text)
    placed = set()
    inserts = []

    for m in _FIG_NUM.finditer(text):
        n = int(m.group(1))
        idx = n - 1
        if 0 <= idx < len(images) and idx not in placed:
            nl = text.find(NL, m.end())
            pos = nl + 1 if nl > 0 else len(text)
            inserts.append((pos, images[idx]["markdown"]))
            placed.add(idx)

    for m in _FIG_CN.finditer(text):
        idx = _CN_NUM.get(m.group(1), -1)
        if 0 <= idx < len(images) and idx not in placed:
            nl = text.find(NL, m.end())
            pos = nl + 1 if nl > 0 else len(text)
            inserts.append((pos, images[idx]["markdown"]))
            placed.add(idx)

    for pos, md in sorted(inserts, reverse=True):
        text = text[:pos] + md + NL + text[pos:]

    unplaced = [i for i in range(len(images)) if i not in placed]
    if unplaced:
        section_starts = [m.end() for m in re.finditer(r"^##\s+.+$", text, re.MULTILINE)]
        if not section_starts:
            text += NL + NL + NL.join(images[i]["markdown"] for i in unplaced)
        else:
            for j, idx in enumerate(unplaced):
                si = j % len(section_starts)
                ss = section_starts[si]
                nl = text.find(NL, ss)
                p = nl + 1 if nl > 0 else ss
                text = text[:p] + images[idx]["markdown"] + NL + text[p:]
    return text


# ── Split strategies ──────────────────────────────────────────────────
H2_SPLIT = re.compile(r"^(##\s+.+)$", re.MULTILINE)
H3_SPLIT = re.compile(r"^(###\s+.+)$", re.MULTILINE)
CN_DIGIT_SPLIT = re.compile(r"^(?:(?:[一-鿿]+、)|(?:\d+(?:\.\d+)*\s+[^\d]))", re.MULTILINE)


def _split_h3_sections(body: str, parent_title: str = "") -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    first_h3 = H3_HEADER.search(body)
    prefix = body[:first_h3.start()].strip() if first_h3 else body.strip()
    if prefix and not is_noise(prefix):
        blocks.append((parent_title, prefix))
    if not first_h3:
        return blocks
    sub_body = body[first_h3.start():]
    parts = H3_SPLIT.split(sub_body)
    if parts and not parts[0].strip():
        parts = parts[1:]
    si = 0
    while si < len(parts):
        sub_title, sub_content = "", ""
        if si + 1 < len(parts) and parts[si].strip():
            sub_title = parts[si].strip()
            sub_content = parts[si + 1].strip() if si + 1 < len(parts) else ""
            si += 2
        else:
            sub_content = parts[si].strip()
            si += 1
        if is_noise(sub_content):
            sub_content = ""
        full_title = f"{parent_title} > {sub_title}" if parent_title else sub_title
        blocks.append((full_title, sub_content))
    if not blocks and parent_title:
        blocks.append((parent_title, ""))
    return blocks


def split_by_h2(text):
    blocks = []
    parts = H2_SPLIT.split(text)
    i = 0
    if parts and parts[0].strip():
        preamble = parts[0].strip()
        if not is_noise(preamble):
            if H3_HEADER.search(preamble):
                blocks.extend(_split_h3_sections(preamble, ""))
            else:
                blocks.append(("", preamble))
    i = 1
    while i < len(parts):
        title, body = "", ""
        if i + 1 < len(parts) and parts[i].strip():
            title = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2
        else:
            body = parts[i].strip()
            i += 1
        if is_noise(body):
            body = ""
        if body and H3_HEADER.search(body):
            blocks.extend(_split_h3_sections(body, title))
        else:
            if body or title:
                blocks.append((title, body))
    return blocks


def split_by_cn_digit(text):
    blocks = []
    parts = CN_DIGIT_SPLIT.split(text)
    if parts and not parts[0].strip():
        parts = parts[1:]
    i = 0
    while i < len(parts):
        title, body = "", ""
        if i + 1 < len(parts) and parts[i].strip():
            title = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2
        else:
            body = parts[i].strip()
            i += 1
        if is_noise(body):
            body = ""
        blocks.append((title, body))
    return blocks


def split_by_newlines(text):
    parts = re.split(r"\n\s*\n", text)
    return [("", p.strip()) for p in parts if p.strip() and not is_noise(p.strip())]


# ── Bracket-based split (【】highest priority) ────────────────────────
BRACKET_BLOCK = re.compile(r'【[^】]+】', re.DOTALL)


def has_bracket_blocks(text: str) -> bool:
    return len(BRACKET_BLOCK.findall(text)) >= 2


def split_by_brackets(text: str) -> list[tuple[str, str]]:
    blocks = []
    last_end = 0
    for m in BRACKET_BLOCK.finditer(text):
        start, end = m.span()
        between = text[last_end:start].strip()
        if between and blocks:
            blocks[-1] = (blocks[-1][0], blocks[-1][1] + "\n" + between)
        inner = m.group(0)[1:-1].strip()
        first_line = inner.split("\n")[0].strip()
        blocks.append((first_line, inner))
        last_end = end
    remaining = text[last_end:].strip()
    if remaining and blocks:
        blocks[-1] = (blocks[-1][0], blocks[-1][1] + "\n" + remaining)
    return blocks


# ── Merge & Main ─────────────────────────────────────────────────────
def merge_short_chunks(chunks):
    if not chunks:
        return chunks
    merged = []
    buf_title, buf_parts = "", []
    for title, body in chunks:
        stripped = body.strip()
        if not stripped or is_noise(stripped):
            continue
        if len(stripped) < MIN_CHUNK_LEN:
            if not buf_title and title:
                buf_title = title
            buf_parts.append(stripped)
        else:
            if buf_parts:
                merged.append((buf_title, "\n\n".join(buf_parts)))
                buf_title, buf_parts = "", []
            merged.append((title, stripped))
    if buf_parts:
        merged.append((buf_title or "", "\n\n".join(buf_parts)))
    return merged


def process_file(filepath: Path, category: str) -> tuple[str, list[dict]]:
    text = filepath.read_text(encoding="utf-8")
    text = _embed_images_inline(text)

    if has_bracket_blocks(text):
        raw = split_by_brackets(text)
        strategy = "BRACKET"
    elif H2_HEADER.search(text) or H3_HEADER.search(text):
        raw = split_by_h2(text)
        strategy = "H2"
    elif CN_SECTION.search(text) or DIGIT_SECTION.search(text):
        raw = split_by_cn_digit(text)
        strategy = "CN+DIGIT"
    else:
        raw = split_by_newlines(text)
        strategy = "NL"

    if strategy != "BRACKET":
        raw = merge_short_chunks(raw)

    results = []
    current_section = None
    for section_title, chunk_text in raw:
        lines = chunk_text.split("\n")
        detected = detect_section(lines)
        if detected:
            current_section = detected
        elif section_title:
            current_section = section_title

        content_type = classify_content(chunk_text)
        heading = current_section or section_title
        if heading and not chunk_text.startswith(heading):
            chunk_text = heading + "\n\n" + chunk_text

        results.append({
            "category": category,
            "source_file": filepath.name,
            "section_path": current_section or section_title or "",
            "content_type": content_type,
            "content": chunk_text,
        })

    for i, chunk in enumerate(results):
        imgs = extract_images_from_md(chunk["content"])
        chunk["has_image"] = len(imgs) > 0
        chunk["image_ids"] = ",".join(img["path"] for img in imgs)

    _NEIGHBOR_CHARS = 200
    for i, chunk in enumerate(results):
        if not chunk.get("has_image"):
            continue
        neighbors: list[str] = []
        if i > 0 and not results[i - 1].get("has_image"):
            neighbors.append(results[i - 1]["content"][:_NEIGHBOR_CHARS])
        if i < len(results) - 1 and not results[i + 1].get("has_image"):
            neighbors.append(results[i + 1]["content"][:_NEIGHBOR_CHARS])
        if neighbors:
            chunk["enriched_content"] = chunk["content"] + "\n\n" + "\n".join(neighbors)

    return strategy, results


def main():
    all_chunks = []
    stats: dict[str, int] = defaultdict(int)
    file_stats = []

    categories = [d.name for d in PARSED.iterdir() if d.is_dir() and d.name not in ("images",)]
    if not categories:
        print("No parsed documents found. Run parse_docs.py first.")
        return

    for category in sorted(categories):
        src = PARSED / category
        files = sorted(src.glob("*.md"))
        print(f"\n{'='*60}")
        print(f"{category}: {len(files)} files")
        print(f"{'='*60}")

        for fp in files:
            strategy, chunks = process_file(fp, category)
            for c in chunks:
                c["chunk_index"] = len(all_chunks)
                stats[c["content_type"]] += 1
                all_chunks.append(c)
            file_stats.append({
                "category": category, "source_file": fp.name,
                "strategy": strategy, "chunks": len(chunks),
            })
            print(f"  [{strategy:8s}] {fp.name}: {len(chunks):3d} chunks")

    out_file = OUT / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Image bindings
    bindings_file = OUT / "image_bindings.json"
    bindings_file.write_text("{}", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_chunks)} chunks -> {out_file}")
    print(f"\nContent type distribution:")
    for ct, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")
    print("Done.")


if __name__ == "__main__":
    main()

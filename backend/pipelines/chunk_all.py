"""Stage 2 v3: Unified chunking for 强电+弱电 with image binding.

Split strategy (same for both):
  ## headers → Chinese numbered → digit numbered → double newline

Output:
  data/chunks/chunks_v3.jsonl
  data/chunks/image_bindings_v3.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
PARSED = ROOT / "data" / "parsed"
OUT = ROOT / "data" / "chunks"
OUT.mkdir(parents=True, exist_ok=True)

MIN_CHUNK_LEN = 50  # up from 30

# ── Noise patterns ───────────────────────────────────────────────────────

PAGE_SLASH = re.compile(r"^\d+\s*/\s*\d+$")
PAGE_CHAPTER = re.compile(r"^第\d+页\s*/\s*第\d+页$")
PAGE_HEADER = re.compile(r"^第\d+页")
STANDALONE_NULL = re.compile(r"^null$")
DOT_LINE = re.compile(r"^[\.\s·•\-—]+$")
TOC_DOTS = re.compile(r"\.{4,}")
EXTRACTION_META = re.compile(r"^本文件(共)?提取\s*\d+\s*[张个]")
IMG_DESC_HEADER = re.compile(r"^#{1,3}\s*图片")
IMG_RENDER = re.compile(r"^!\[.*render\.png\]")
TOC_ENTRY = re.compile(r"^\d+(?:\.\d+)*\s+\S.*\d{1,3}$")

# Table separator line only (no data) — filter these
TABLE_SEP_ONLY = re.compile(r"^\|[\s\-\|]+\|\s*$")


def _is_toc_block(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    toc_count = sum(1 for l in lines if TOC_ENTRY.match(l))
    return toc_count >= len(lines) * 0.6


def is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if PAGE_SLASH.match(t) or PAGE_CHAPTER.match(t):
        return True
    if STANDALONE_NULL.match(t) or DOT_LINE.match(t):
        return True
    if EXTRACTION_META.match(t):
        return True
    if IMG_DESC_HEADER.match(t) and len(t) < 30:
        return True
    if IMG_RENDER.match(t):
        return True
    if PAGE_HEADER.match(t) and len(t) < 25:
        return True
    if TOC_DOTS.search(t):
        return True
    if TABLE_SEP_ONLY.match(t):
        return True
    if _is_toc_block(t):
        return True
    return False


# ── Section detection ─────────────────────────────────────────────────────

H2_HEADER = re.compile(r"^##\s+(.+)$", re.MULTILINE)
H3_HEADER = re.compile(r"^###\s+(.+)$", re.MULTILINE)
CN_SECTION = re.compile(r"^[一二三四五六七八九十]+、\s*(.+)$", re.MULTILINE)
CN_CHAPTER = re.compile(r"^第[一二三四五六七八九十百\d]+章\s*(.+)$", re.MULTILINE)
DIGIT_SECTION = re.compile(r"^\d+(?:\.\d+)*\s+(.+)$", re.MULTILINE)
CAT_HEADER = re.compile(r"^(总则|适用范围|编制说明|前言|编制目的|使用说明|原则|依据|引用|术语|目录)$", re.MULTILINE)
CN_SPACE = re.compile(r"^[一二三四五六七八九十]+\s{1,5}([^\d\s].{1,40})$", re.MULTILINE)
BAD_SECTION = re.compile(r"图片描述|第\s*\d+\s*页|render|page\d+|^\.{4,}")

ALL_SECTIONS = [H2_HEADER, H3_HEADER, CN_SECTION, CN_CHAPTER, CN_SPACE, DIGIT_SECTION, CAT_HEADER]


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


# ── Content type classification ───────────────────────────────────────────

STANDARD_CODE = re.compile(
    r"(?:GB|GB/T|JGJ|CJJ|CECS|DGJ|DBJ|GBZ|GB/TZ)\s*[\d\.]+[–—-]*\d*"
)
DESIGN_RULE_KW = re.compile(
    r"(?:不应|不得|严禁|禁止|禁用|慎用|切勿|务必|必须|应按|应设|应采用|应配置)"
    r"|(?:宜[^用]|不宜)"
    r"|(?:优先执行|参照执行|按.*标准执行|配置标准|原则上)"
)
KV_SPEC = re.compile(r"^[^\s：:]+\s*[：:]\s*\S", re.MULTILINE)
_SUB_TABLE_KW = re.compile(r"（含）|小于.*（含）|大于.*（含）|不超过.*（含）|不小于.*（含）")
IMG_CONTENT_KW = re.compile(
    r"GLM-5V-Turbo|通过.*视觉模型.*描述|图片附件"
)


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
        # Count table rows with "（不含）" vs "（含）" range patterns
        table_rows = [l for l in lines if "|" in l]
        count_buhan = sum(1 for l in table_rows if "（不含）" in l)
        count_han = sum(1 for l in table_rows if "（含）" in l)
        # Main table uses 不含; sub-table uses 含 for range boundaries
        if count_han > count_buhan:
            return "副表"
        return "参数表"

    if DESIGN_RULE_KW.search(text):
        return "设计规则"

    return "说明文本"


# ── Image binding ─────────────────────────────────────────────────────────

IMG_REF = re.compile(r"!\[([^\]]*)\]\(images/([^)]+)\)")


def extract_images_from_md(content: str) -> list[dict]:
    """Extract image references from markdown content."""
    images = []
    for m in IMG_REF.finditer(content):
        images.append({
            "description": m.group(1),
            "path": m.group(2),
            "markdown": m.group(0),
        })
    return images


def build_image_bindings(source_md: Path, chunks: list[dict]) -> dict:
    """Build image→chunk mappings for a single source file."""
    full_text = source_md.read_text(encoding="utf-8")
    all_images = extract_images_from_md(full_text)

    if not all_images:
        return {}

    bindings = {}
    # Match images to chunks: if an image reference appears in/near a chunk
    for i, chunk in enumerate(chunks):
        chunk_images = extract_images_from_md(chunk["content"])
        if chunk_images:
            bindings[f"chunk_{i}"] = {
                "images": [img["path"] for img in chunk_images],
                "markdowns": [img["markdown"] for img in chunk_images],
                "descriptions": [img["description"] for img in chunk_images],
            }

    # Also create standalone image chunks for images not in any text chunk
    bound_images = set()
    for v in bindings.values():
        bound_images.update(v["images"])

    unbound = [img for img in all_images if img["path"] not in bound_images]

    return {"bound": bindings, "unbound_images": unbound, "all_images": all_images}



# ── Image inline embedding (v3.1: 图片随文走) ─────────────────────────────

_CN_NUM = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4,
    "六": 5, "七": 6, "八": 7, "九": 8, "十": 9,
}

_FIG_NUM = re.compile(r"图\s*(\d+)")
_FIG_CN = re.compile(r"附图\s*([一二三四五六七八九十]+)")
_REF_DWG = re.compile(r"参考大样图|参考图|示意图|见附图|详见附图|如下图所示")
_IMG_SECTION = re.compile(r"(?:##\s+)?图片附件\s*\n.*?(?=\n##\s|\Z)", re.DOTALL)

NL = chr(10)

def _embed_images_inline(text: str) -> str:
    img_match = _IMG_SECTION.search(text)
    if not img_match:
        return text
    img_section_text = img_match.group(0)
    images = extract_images_from_md(img_section_text)
    if not images:
        return text
    text = _IMG_SECTION.sub("", text).rstrip()
    text = re.sub(r"\n*本文件[共包含]*\s*\d+\s*[张个]图片[。，]?\s*\n*", "\n", text)
    placed = set()
    inserts = []  # (position, markdown) tuples, will sort and insert in reverse

    # 1. Collect 图N matches
    for m in _FIG_NUM.finditer(text):
        n = int(m.group(1))
        idx = n - 1
        if 0 <= idx < len(images) and idx not in placed:
            nl = text.find(NL, m.end())
            pos = nl + 1 if nl > 0 else len(text)
            inserts.append((pos, images[idx]["markdown"]))
            placed.add(idx)

    # 2. Collect 附图N matches
    for m in _FIG_CN.finditer(text):
        cn = m.group(1)
        idx = _CN_NUM.get(cn, -1)
        if 0 <= idx < len(images) and idx not in placed:
            nl = text.find(NL, m.end())
            pos = nl + 1 if nl > 0 else len(text)
            inserts.append((pos, images[idx]["markdown"]))
            placed.add(idx)

    # Insert in reverse order (highest position first)
    for pos, md in sorted(inserts, reverse=True):
        text = text[:pos] + md + NL + text[pos:]

    # 3. Match 参考大样图/参考图/示意图 markers
    unplaced = [i for i in range(len(images)) if i not in placed]
    ref_inserts = []  # (position, markdown)

    for m in _REF_DWG.finditer(text):
        if not unplaced:
            break
        nl = text.find(NL, m.end())
        pos = nl + 1 if nl > 0 else len(text)
        after = text[pos:pos + 800]
        qk_lines = re.findall(r"^情况[一二三四五六七八九十]+[：:].*$", after, re.MULTILINE)
        if qk_lines and len(unplaced) >= len(qk_lines):
            for i, qk in enumerate(qk_lines):
                if i >= len(unplaced):
                    break
                qk_pos = after.find(qk)
                qk_nl = after.find(NL, qk_pos + len(qk))
                actual = pos + (qk_nl + 1 if qk_nl > 0 else qk_pos + len(qk))
                ref_inserts.append((actual, images[unplaced[i]]["markdown"]))
                placed.add(unplaced[i])
            unplaced = [i for i in range(len(images)) if i not in placed]
        else:
            ref_inserts.append((pos, images[unplaced[0]]["markdown"]))
            placed.add(unplaced.pop(0))

    for pos, md in sorted(ref_inserts, reverse=True):
        text = text[:pos] + md + NL + text[pos:]

    # 4. Distribute remaining images across sections
    unplaced = [i for i in range(len(images)) if i not in placed]
    if unplaced:
        section_starts = [m.end() for m in re.finditer(r"^##\s+.+$", text, re.MULTILINE)]
        if not section_starts:
            text += NL + NL + NL.join(images[i]["markdown"] for i in unplaced)
        else:
            n_sections = len(section_starts)
            section_inserts = []
            for j, idx in enumerate(unplaced):
                si = j % n_sections
                ss = section_starts[si]
                nl = text.find(NL, ss)
                p = nl + 1 if nl > 0 else ss
                section_inserts.append((p, images[idx]["markdown"]))
            for pos, md in sorted(section_inserts, reverse=True):
                text = text[:pos] + md + NL + text[pos:]

    return text


H1_H2_SPLIT = re.compile(r"^(#{1,2}\s+.+)$", re.MULTILINE)
H2_SPLIT = re.compile(r"^(##\s+.+)$", re.MULTILINE)
H3_SPLIT = re.compile(r"^(###\s+.+)$", re.MULTILINE)
CN_DIGIT_SPLIT = re.compile(
    r"^(?:(?:[一-鿿]+、)|(?:\d+(?:\.\d+)*\s+[^\d]))",
    re.MULTILINE,
)

def _split_h3_sections(body: str, parent_title: str = "") -> list[tuple[str, str]]:
    """Split body text by ### headers, carrying parent ## context forward.

    Content before the first ### belongs to the parent; each ### section
    gets a '## parent > ### child' title so the full hierarchy is searchable.
    """
    blocks: list[tuple[str, str]] = []
    first_h3 = H3_HEADER.search(body)
    prefix = body[:first_h3.start()].strip()

    if prefix and not is_noise(prefix):
        blocks.append((parent_title, prefix))

    sub_body = body[first_h3.start():]
    parts = H3_SPLIT.split(sub_body)
    if parts and not parts[0].strip():
        parts = parts[1:]
    si = 0
    while si < len(parts):
        sub_title = ""
        sub_content = ""
        if si + 1 < len(parts) and parts[si].strip():
            sub_title = parts[si].strip()  # capture group already has ### prefix
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
    """Split on # and ## headers, preserving ### sub-section hierarchy.

    Both H1 and H2 are treated as chunk boundaries. This prevents one
    long H2 section from swallowing multiple H1 subsystems that follow
    it (e.g. DOCX where subsystems are Heading 1 but their parent
    category is Heading 2).

    ### children carry their parent title as a section_path prefix
    (e.g. "## 住宅户内用电需要系数 > ### 成都区域"), ensuring the
    parent context is embedded and searchable even when the parent
    body is empty.
    """
    blocks = []
    parts = H1_H2_SPLIT.split(text)
    # parts: [preamble] ["# title"] [body] ["## title"] [body] ...

    i = 0
    # Preamble (content before first header) → standalone block
    if parts and parts[0].strip():
        preamble = parts[0].strip()
        if not is_noise(preamble):
            if H3_HEADER.search(preamble):
                blocks.extend(_split_h3_sections(preamble, ""))
            else:
                blocks.append(("", preamble))
    i = 1

    # # / ## title + body pairs
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
    blocks = []
    for p in parts:
        p = p.strip()
        if p and not is_noise(p):
            blocks.append(("", p))
    return blocks


# ── Table-section split (for Excel-generated markdown) ─────────────────────
# Excel sheets often produce one continuous markdown table with inline section
# title rows (e.g. "| 电梯需要系数取值分段表 |  |  |"). Split at those boundaries.

def _is_table_title_row(line: str) -> bool:
    """Check if a markdown table row is a section title (not data).
    
    Only matches rows where: first cell has text, other cells empty, 
    AND title is long enough to NOT be a sub-label like region name.
    """
    if not line.startswith("|"):
        return False
    cells = [c.strip() for c in line.split("|")]
    cells = [c for c in cells if c]
    if not cells:
        return False
    # Must be a single-cell row (section title)
    if len(cells) != 1:
        return False
    # Minimum meaningful title length; sub-labels (region names, etc.) 
    # are < 10 chars and should NOT split
    if len(cells[0]) < 10:
        return False
    return True


def _is_sub_label(title: str) -> bool:
    """Check if a table section title is a sub-label (region name, etc.)
    that should carry forward its parent's context.
    """
    # Region-like: ends with "区域"
    if title.endswith("区域"):
        return True
    # Very short titles (< 8 chars) that are likely sub-categories
    if len(title) <= 6:
        return True
    return False


def has_table_sections(text: str) -> bool:
    """Check if text is primarily a markdown table with inline section titles."""
    lines = text.split("\n")
    table_rows = [l for l in lines if l.strip().startswith("|")]
    if len(table_rows) < 4:
        return False
    # At least 50% of lines should be table rows (it's a table-heavy doc)
    if len(table_rows) < len(lines) * 0.3:
        return False
    # Must have at least one section title row
    return any(_is_table_title_row(l) for l in table_rows)


TABLE_SEP_RE = re.compile(r"^\|[\s\-:]+\|\s*$")


def split_by_table_sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown table at inline section title rows.

    Each sub-table keeps the original header row for completeness.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    last_sep = ""  # Most recent table separator row
    preamble: list[str] = []  # Heading lines before current table block
    header_row = ""  # Original header row (first data row after separator)

    for line in lines:
        stripped = line.strip()

        # Track table separators for sub-table reconstruction
        if TABLE_SEP_RE.match(stripped):
            last_sep = line
            current_lines.append(line)
            # The first separator marks the boundary between header and data;
            # capture the header row if we haven't yet
            if not header_row and current_lines:
                # Look backwards in current_lines for the header row
                for prev in reversed(current_lines[:-1]):
                    if prev.strip().startswith("|") and not TABLE_SEP_RE.match(prev.strip()):
                        header_row = prev
                        break
            continue

        if not stripped.startswith("|"):
            # Non-table line: flush accumulated table content
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body and not is_noise(body):
                    heading = "\n".join(preamble).strip() if preamble else ""
                    if heading and not is_noise(heading):
                        body = heading + "\n\n" + body
                    blocks.append((current_title, body))
                current_lines = []
                current_title = ""
                preamble = []
            # Accumulate heading lines for next block's context
            preamble.append(line)
            continue

        # In a table: check for section title row
        if _is_table_title_row(stripped):
            # Flush preceding table block
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body and not is_noise(body):
                    heading = "\n".join(preamble).strip() if preamble else ""
                    if heading and not is_noise(heading):
                        body = heading + "\n\n" + body
                    blocks.append((current_title, body))
                preamble = []
            # Start new sub-table with header row + separator for completeness
            raw_title = [c.strip() for c in stripped.split("|") if c.strip()][0].rstrip("：").rstrip(":")
            # Carry forward parent context: if previous title exists and new title
            # is a short sub-label (region name, etc.), prepend parent.
            if current_title and _is_sub_label(raw_title):
                current_title = f"{current_title}-{raw_title}"
            else:
                current_title = raw_title
            if header_row:
                current_lines = [header_row, last_sep] if last_sep else [header_row]
            else:
                current_lines = [last_sep] if last_sep else []
            continue

        # Capture header row (first non-separator table row)
        if not header_row and stripped.startswith("|"):
            header_row = line

        current_lines.append(line)

    # Flush remaining
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body and not is_noise(body):
            heading = "\n".join(preamble).strip() if preamble else ""
            if heading and not is_noise(heading):
                body = heading + "\n\n" + body
            blocks.append((current_title, body))

    # If no blocks found, fall back to whole text as one chunk
    if not blocks:
        body = text.strip()
        if body:
            blocks.append(("", body))

    return blocks

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
        if not buf_title:
            buf_title = ""
        merged.append((buf_title, "\n\n".join(buf_parts)))
    return merged


DIGIT_SUB_ITEM = re.compile(r'^\d+[、.．)\s]')


def merge_digit_prefixed(chunks):
    """Merge chunks that start with a digit sub-item pattern into the
    preceding chunk.  '1、xxx', '2. xxx', '3) xxx' are sub-items,
    not standalone sections."""
    if not chunks:
        return chunks
    merged = []
    for title, body in chunks:
        stripped = body.strip()
        if merged and DIGIT_SUB_ITEM.match(stripped):
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, prev_body + "\n" + stripped)
        else:
            merged.append((title, stripped))
    return merged

def keep_table_intact(chunks):
    return chunks



# ── Bracket-based split (v3.2: 【】highest priority) ──────────────────────────

BRACKET_BLOCK = re.compile(r'【[^】]+】', re.DOTALL)


def split_by_brackets(text: str) -> list[tuple[str, str]]:
    """Split text by 【】 blocks. Each block = one chunk. Images between blocks
    are attached to the preceding block."""
    blocks = []
    last_end = 0

    for m in BRACKET_BLOCK.finditer(text):
        start, end = m.span()

        # Content before first block → prepend to first block
        between = text[last_end:start].strip()
        # Content between blocks → attach to preceding block
        if between and blocks:
            blocks[-1] = (blocks[-1][0], blocks[-1][1] + "\n" + between)

        inner = m.group(0)[1:-1].strip()  # Remove 【】
        first_line = inner.split("\n")[0].strip()
        # Use first line as section title
        blocks.append((first_line, inner))

        last_end = end

    # Remainder after last block
    remaining = text[last_end:].strip()
    if remaining and blocks:
        blocks[-1] = (blocks[-1][0], blocks[-1][1] + "\n" + remaining)

    return blocks


def has_bracket_blocks(text: str) -> bool:
    """Require >=2 【】 blocks so a single accidental bracket in technical
    specs won't hijack a file that should use H2 split."""
    return len(BRACKET_BLOCK.findall(text)) >= 2


# ── Main ──────────────────────────────────────────────────────────────────


def process_file(filepath: Path, specialty: str) -> tuple[str, list[dict]]:
    text = filepath.read_text(encoding="utf-8")

    # Pre-process: embed images inline (v3.1: images follow text)
    text = _embed_images_inline(text)

    # Choose split strategy — 【】brackets have highest priority
    if has_bracket_blocks(text):
        raw = split_by_brackets(text)
        strategy = "BRACKET"
    elif has_table_sections(text):
        raw = split_by_table_sections(text)
        strategy = "TABLE"
    elif H2_HEADER.search(text) or H3_HEADER.search(text):
        raw = split_by_h2(text)
        strategy = "H2"
    elif CN_SECTION.search(text) or DIGIT_SECTION.search(text):
        raw = split_by_cn_digit(text)
        strategy = "CN+DIGIT"
    else:
        raw = split_by_newlines(text)
        strategy = "NL"

    # Only merge short chunks for non-bracket strategies (【】blocks are sacrosanct)
    if strategy != "BRACKET":
        raw = keep_table_intact(raw)
        raw = merge_short_chunks(raw)
        raw = merge_digit_prefixed(raw)

    results = []
    current_section: str | None = None

    for section_title, chunk_text in raw:
        lines = chunk_text.split("\n")
        detected = detect_section(lines)
        if detected:
            current_section = detected
        elif section_title:
            current_section = section_title

        content_type = classify_content(chunk_text)

        # Prepend section title to content so it is searchable
        heading = current_section or section_title
        if heading and not chunk_text.startswith(heading):
            chunk_text = heading + "\n\n" + chunk_text

        results.append({
            "specialty": specialty,
            "source_file": filepath.name,
            "section_path": current_section or section_title or "",
            "content_type": content_type,
            "content": chunk_text,
            "split_strategy": strategy,
        })

    # Add image metadata to each chunk
    for i, chunk in enumerate(results):
        imgs = extract_images_from_md(chunk["content"])
        chunk["has_image"] = len(imgs) > 0
        chunk["image_ids"] = ",".join(img["path"] for img in imgs)

    # ── Image context enrichment (inspired by RAGFlow attach_media_context) ──
    # For chunks with embedded images, inject neighbor-chunk text into a
    # separate enriched_content field so the image chunk becomes semantically
    # retrievable — its embedding will capture surrounding context, not just
    # the (often short) image description text.
    _NEIGHBOR_CHARS = 200  # chars to take from each neighbor
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
    global_image_bindings = {}
    stats: dict[str, int] = defaultdict(int)
    file_stats = []

    for specialty in ["强电", "弱电"]:
        src = PARSED / specialty
        if not src.is_dir():
            print(f"  Skip {specialty}: not found")
            continue

        files = sorted(src.glob("*.md"))
        print(f"\n{'='*60}")
        print(f"{specialty}: {len(files)} files")
        print(f"{'='*60}")

        for fp in files:
            strategy, chunks = process_file(fp, specialty)
            # Build image bindings from source
            bindings = build_image_bindings(fp, chunks)
            if bindings:
                global_image_bindings[fp.name] = bindings

            # Add chunk_index
            for c in chunks:
                c["chunk_index"] = len(all_chunks)
                stats[c["content_type"]] += 1
                all_chunks.append(c)
            file_stats.append({
                "specialty": specialty,
                "source_file": fp.name,
                "strategy": strategy,
                "chunks": len(chunks),
                "images_referenced": len(bindings.get("all_images", [])),
            })
            print(f"  [{strategy:8s}] {fp.name}: {len(chunks):3d} chunks, "
                  f"{len(bindings.get('all_images', [])):3d} images ref'd")

    # Write chunks
    out_file = OUT / "chunks_v3.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write image bindings
    bindings_file = OUT / "image_bindings_v3.json"
    with open(bindings_file, "w", encoding="utf-8") as f:
        json.dump(global_image_bindings, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_chunks)} chunks → {out_file}")
    print(f"Image bindings: {bindings_file}")
    print(f"\nContent type distribution:")
    for ct, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")

    # Files with images
    img_files = [s for s in file_stats if s["images_referenced"] > 0]
    print(f"\nFiles with image references: {len(img_files)}")
    total_images = sum(s["images_referenced"] for s in file_stats)
    print(f"Total image references: {total_images}")
    print("Done.")


if __name__ == "__main__":
    main()

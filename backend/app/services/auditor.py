"""Pipeline auditor — validates parse -> chunk -> index -> retrieve end-to-end.

Fixed workflow: runs automatically after every pipeline completion.
Seven checks across three severity levels:
  ERROR (4): chunk_count, retrievability, embedding_dims, bm25_index
  WARN  (3): section_paths, table_chunks, file_diversity
  INFO  (1): hierarchy_depth
"""

import json
import pickle
import random
from pathlib import Path
from dataclasses import dataclass, field

import jieba


@dataclass
class AuditCheck:
    name: str
    passed: bool
    detail: str
    level: str = "INFO"  # ERROR | WARN | INFO


@dataclass
class AuditReport:
    checks: list[AuditCheck] = field(default_factory=list)
    passed: bool = True
    error: str = ""

    def model_dump(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail, "level": c.level} for c in self.checks],
            "error": self.error,
        }


class PipelineAuditor:
    def __init__(self, chunks_file: Path, milvus_dir: str, bm25_file: Path, collection: str):
        self.chunks_file = chunks_file
        self.milvus_dir = milvus_dir
        self.bm25_file = bm25_file
        self.collection = collection

    def run_all_checks(self) -> AuditReport:
        report = AuditReport()

        checks = [
            self._check_chunk_count(),
            self._check_section_paths(),
            self._check_hierarchy_depth(),
            self._check_retrievability(),
            self._check_table_chunks(),
            self._check_file_diversity(),
            self._check_embedding_dims(),
            self._check_bm25_index(),
        ]

        has_error = any(c.level == "ERROR" and not c.passed for c in checks)
        report.checks = checks
        report.passed = not has_error
        return report

    # ── ERROR level checks ──────────────────────────────────────────────

    def _check_chunk_count(self) -> AuditCheck:
        if not self.chunks_file.exists():
            return AuditCheck("chunk_count", False, "chunks file not found", "ERROR")
        chunks = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        empty = sum(1 for c in chunks if not c.get("content", "").strip())
        if empty > len(chunks) * 0.1:
            return AuditCheck("chunk_count", False, f"{empty}/{len(chunks)} chunks have empty body", "ERROR")
        if len(chunks) == 0:
            return AuditCheck("chunk_count", False, "0 chunks indexed", "ERROR")
        return AuditCheck("chunk_count", True, f"{len(chunks)} chunks indexed", "ERROR")

    def _check_retrievability(self) -> AuditCheck:
        """Sample 3 chunks, search for their content, verify self-hit."""
        if not self.chunks_file.exists():
            return AuditCheck("retrievability", False, "chunks file not found", "ERROR")
        chunks = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        if len(chunks) < 3:
            return AuditCheck("retrievability", False, "too few chunks to sample", "ERROR")

        # Load BM25 for local scoring
        bm25, tokenized = None, None
        if self.bm25_file.exists():
            with open(self.bm25_file, "rb") as f:
                bm25, tokenized = pickle.load(f)

        if bm25 is None or tokenized is None:
            return AuditCheck("retrievability", False, "BM25 index not available for audit", "ERROR")

        sample = random.sample(chunks, min(3, len(chunks)))
        hits = 0
        for c in sample:
            content = c.get("content", "")
            if not content.strip():
                continue
            # Extract 2-3 meaningful terms for search
            words = [w for w in jieba.cut(content) if len(w.strip()) >= 2]
            query = " ".join(words[:5]) if words else content[:20]
            scores = bm25.get_scores([w for w in jieba.cut(query) if w.strip()])
            top_idx = max(range(len(scores)), key=lambda i: scores[i])
            if top_idx == c.get("chunk_index", -1):
                hits += 1

        if hits >= 2:
            return AuditCheck("retrievability", True, f"self-hit {hits}/{len(sample)} sampled chunks", "ERROR")
        return AuditCheck("retrievability", False, f"only {hits}/{len(sample)} chunks self-retrievable", "ERROR")

    def _check_embedding_dims(self) -> AuditCheck:
        """Verify all vectors are 1024-dimensional."""
        vectors_file = self.chunks_file.parent / "vectors.pkl"
        if not vectors_file.exists():
            return AuditCheck("embedding_dims", False, "vectors.pkl not found", "ERROR")
        try:
            with open(vectors_file, "rb") as f:
                _, vectors = pickle.load(f)
            dims = {len(v) for v in vectors}
            if len(dims) == 1 and 1024 in dims:
                return AuditCheck("embedding_dims", True, f"all {list(dims)[0]}d", "ERROR")
            return AuditCheck("embedding_dims", False, f"inconsistent dims: {dims}", "ERROR")
        except Exception as e:
            return AuditCheck("embedding_dims", False, str(e)[:100], "ERROR")

    def _check_bm25_index(self) -> AuditCheck:
        if not self.bm25_file.exists():
            return AuditCheck("bm25_index", False, "BM25 index file not found", "ERROR")
        try:
            with open(self.bm25_file, "rb") as f:
                bm25, tokenized = pickle.load(f)
            total_tokens = sum(len(t) for t in tokenized)
            if total_tokens == 0:
                return AuditCheck("bm25_index", False, "BM25 index has 0 tokens", "ERROR")
            return AuditCheck("bm25_index", True, f"{total_tokens} tokens indexed, {len(tokenized)} docs", "ERROR")
        except Exception as e:
            return AuditCheck("bm25_index", False, str(e)[:100], "ERROR")

    # ── WARN level checks ───────────────────────────────────────────────

    def _check_section_paths(self) -> AuditCheck:
        if not self.chunks_file.exists():
            return AuditCheck("section_paths", False, "chunks file not found", "WARN")
        chunks = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        if not chunks:
            return AuditCheck("section_paths", False, "no chunks", "WARN")
        with_sp = sum(1 for c in chunks if c.get("section_path", "").strip())
        ratio = with_sp / len(chunks)
        if ratio >= 0.7:
            return AuditCheck("section_paths", True, f"{ratio:.0%} chunks have section_path", "WARN")
        return AuditCheck("section_paths", False, f"only {ratio:.0%} have section_path (target 70%)", "WARN")

    def _check_table_chunks(self) -> AuditCheck:
        if not self.chunks_file.exists():
            return AuditCheck("table_chunks", False, "chunks file not found", "WARN")
        chunks = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        table_chunks = [c for c in chunks if c.get("content_type") in ("参数表", "副表")]
        if not table_chunks:
            return AuditCheck("table_chunks", True, "no table chunks", "WARN")

        # Check strip survivability: tables with | separator should have content after strip
        import re
        SEP = re.compile(r"\|[-:\s]+\|")
        stripped_empty = 0
        for c in table_chunks:
            content = c.get("content", "")
            if SEP.search(content):
                # Would be stripped by _strip_tables_from_text
                non_table_lines = [l for l in content.split("\n") if not l.strip().startswith("|")]
                if not non_table_lines or len("\n".join(non_table_lines).strip()) < 20:
                    stripped_empty += 1

        if stripped_empty > 0:
            return AuditCheck("table_chunks", False, f"{stripped_empty} table chunks may be invisible to LLM after strip", "WARN")
        return AuditCheck("table_chunks", True, f"{len(table_chunks)} table chunks, all have surrounding text", "WARN")

    def _check_file_diversity(self) -> AuditCheck:
        if not self.chunks_file.exists():
            return AuditCheck("file_diversity", False, "chunks file not found", "WARN")
        chunks = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        files = {c.get("source_file", "") for c in chunks}
        if len(files) >= 2:
            return AuditCheck("file_diversity", True, f"{len(files)} unique source files", "WARN")
        return AuditCheck("file_diversity", False, f"only {len(files)} source files (need ≥2 for diversity)", "WARN")

    # ── INFO level checks ───────────────────────────────────────────────

    def _check_hierarchy_depth(self) -> AuditCheck:
        if not self.chunks_file.exists():
            return AuditCheck("hierarchy_depth", False, "chunks file not found", "INFO")
        chunks = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        if not chunks:
            return AuditCheck("hierarchy_depth", False, "no chunks", "INFO")
        with_hierarchy = sum(1 for c in chunks if ">" in c.get("section_path", ""))
        ratio = with_hierarchy / len(chunks)
        return AuditCheck("hierarchy_depth", True, f"{ratio:.0%} chunks have multi-level hierarchy", "INFO")

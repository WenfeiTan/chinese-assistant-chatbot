from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.db import BASE_DIR, get_connection, init_db, utc_now
from app.model_client import embed_texts, get_embedding_model, get_embedding_provider


KB_DIR = BASE_DIR / "知识库min"
TEACHING_FILES = [
    KB_DIR / "教材" / "发展汉语高级阅读1.md",
    KB_DIR / "教材" / "发展汉语高级阅读1参考答案.md",
]
LEVEL_FILES = [
    KB_DIR / "水平表" / "HSK4词汇.md",
    KB_DIR / "水平表" / "HSK5词汇.md",
    KB_DIR / "水平表" / "HSK语法1-6.md",
]

ARTICLE_RE = re.compile(r"^#\s*(文章[一二三四五六七八九十]+)\s*(.+)?\s*$")
PRACTICAL_ARTICLE_RE = re.compile(r"^##\s*(实用阅读)\s+(.+?)\s*$")
PARAGRAPH_RE = re.compile(r"^\[(\d+)\]\s*(.+)")
NOTE_RE = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩])\s*([^：:（(]+)")
SECTION_RE = re.compile(r"^(?:##\s*)?([一二三四五六七八九十]+)、\s*(.+)")
QUESTION_RE = re.compile(r"^(\d+)[.．、]\s*(.*)")
OPTION_RE = re.compile(r"^([A-H])[.．、]\s*(.*)")
INLINE_OPTION_RE = re.compile(r"([A-H])[.．、]?([^A-H]+?)(?=(?:[A-H][.．、]?)|$)")
ANSWER_RE = re.compile(r"(\d+)[.．、]\s*([^0-9]+?)(?=\s+\d+[.．、]|$)")
ARTICLE_NUMERALS = "一二三四五六七八九十"


@dataclass(frozen=True)
class Chunk:
    id: str
    index_name: str
    source: str
    source_file: str
    chunk_type: str
    heading: str | None
    content: str
    metadata: dict[str, Any]

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "id": self.id,
                "index_name": self.index_name,
                "source": self.source,
                "chunk_type": self.chunk_type,
                "content": self.content,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative_source(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_knowledge_files() -> None:
    missing = [path for path in [*TEACHING_FILES, *LEVEL_FILES] if not path.exists()]
    if missing:
        formatted = "\n".join(str(path.relative_to(BASE_DIR)) for path in missing)
        raise FileNotFoundError(f"Missing knowledge base files:\n{formatted}")


def article_id_from_count(count: int) -> str:
    return f"article_{count:03d}"


def section_id_from_count(count: int) -> str:
    return f"section_{count:03d}"


def base_metadata(
    chunk_id: str,
    source: str,
    source_file: str,
    chunk_type: str,
    article_id: str | None,
    article_title: str | None,
    section_id: str | None = None,
    section_title: str | None = None,
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "index_name": "teaching_index",
        "source": source,
        "source_file": source_file,
        "chunk_type": chunk_type,
        "article_id": article_id,
        "article_title": article_title,
        "section_id": section_id,
        "section_title": section_title,
        "paragraph_no": None,
        "question_no": None,
    }


def split_long_paragraph(text: str, limit: int = 1000) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[。！？；;])", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > limit and current:
            parts.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        parts.append(current.strip())
    return parts or [text]


def parse_option_labels(lines: list[str]) -> list[str]:
    labels: list[str] = []
    for line in lines:
        match = OPTION_RE.match(line.strip())
        if match and match.group(1) not in labels:
            labels.append(match.group(1))
    return labels


def flush_question(
    chunks: list[Chunk],
    source: str,
    source_file: str,
    article_id: str | None,
    article_title: str | None,
    section_id: str | None,
    section_title: str | None,
    question_no: str | None,
    lines: list[str],
) -> None:
    if not question_no or not lines:
        return
    safe_section_id = section_id or "section_000"
    question_id = f"{int(question_no):03d}" if question_no.isdigit() else question_no
    chunk_id = f"{article_id or 'article_000'}_{safe_section_id}_q_{question_id}"
    content = "\n".join(line.strip() for line in lines if line.strip())
    option_labels = parse_option_labels(lines)
    metadata = base_metadata(
        chunk_id,
        source,
        source_file,
        "question_block",
        article_id,
        article_title,
        section_id,
        section_title,
    )
    metadata.update(
        {
            "question_no": question_no,
            "has_options": bool(option_labels),
            "option_labels": option_labels,
        }
    )
    chunks.append(
        Chunk(
            id=chunk_id,
            index_name="teaching_index",
            source=source,
            source_file=source_file,
            chunk_type="question_block",
            heading=section_title or article_title,
            content=content,
            metadata=metadata,
        )
    )


def parse_teaching_markdown(path: Path) -> list[Chunk]:
    source = relative_source(path)
    source_file = path.name
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    article_count = 0
    section_count = 0
    article_id: str | None = None
    article_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    question_no: str | None = None
    question_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("![](") or line.startswith("(选自"):
            continue

        article_match = ARTICLE_RE.match(line)
        practical_match = PRACTICAL_ARTICLE_RE.match(line)
        if article_match or practical_match:
            if question_no != "section" or len(question_lines) > 1:
                flush_question(
                    chunks,
                    source,
                    source_file,
                    article_id,
                    article_title,
                    section_id,
                    section_title,
                    question_no,
                    question_lines,
                )
            question_no = None
            question_lines = []
            article_count += 1
            section_count = 0
            article_id = article_id_from_count(article_count)
            if article_match:
                article_title = f"{article_match.group(1)} {article_match.group(2) or ''}".strip()
            else:
                article_title = f"{practical_match.group(1)} {practical_match.group(2)}"
            section_id = None
            section_title = None
            chunk_id = f"{article_id}_title"
            metadata = base_metadata(chunk_id, source, source_file, "article_title", article_id, article_title)
            chunks.append(
                Chunk(chunk_id, "teaching_index", source, source_file, "article_title", article_title, article_title, metadata)
            )
            continue

        section_match = SECTION_RE.match(line)
        if section_match and article_id:
            if question_no != "section" or len(question_lines) > 1:
                flush_question(
                    chunks,
                    source,
                    source_file,
                    article_id,
                    article_title,
                    section_id,
                    section_title,
                    question_no,
                    question_lines,
                )
            question_no = None
            question_lines = []
            section_count += 1
            section_id = section_id_from_count(section_count)
            section_title = f"{section_match.group(1)}、{section_match.group(2).strip()}"
            question_no = "section"
            question_lines = [section_title]
            chunk_id = f"{article_id}_{section_id}"
            metadata = base_metadata(
                chunk_id,
                source,
                source_file,
                "exercise_section",
                article_id,
                article_title,
                section_id,
                section_title,
            )
            chunks.append(
                Chunk(chunk_id, "teaching_index", source, source_file, "exercise_section", article_title, section_title, metadata)
            )
            continue

        paragraph_match = PARAGRAPH_RE.match(line)
        if paragraph_match and article_id and not section_id:
            paragraph_no = int(paragraph_match.group(1))
            paragraph_text = line
            for part_no, paragraph_part in enumerate(split_long_paragraph(paragraph_text), start=1):
                suffix = f"_part_{part_no:02d}" if part_no > 1 else ""
                chunk_id = f"{article_id}_p_{paragraph_no:03d}{suffix}"
                metadata = base_metadata(chunk_id, source, source_file, "paragraph", article_id, article_title)
                metadata["paragraph_no"] = paragraph_no
                if part_no > 1:
                    metadata["part_no"] = part_no
                chunks.append(
                    Chunk(chunk_id, "teaching_index", source, source_file, "paragraph", article_title, paragraph_part, metadata)
                )
            continue

        note_match = NOTE_RE.match(line)
        if note_match and article_id:
            note_no = note_match.group(1)
            term = note_match.group(2).strip()
            chunk_id = f"{article_id}_note_{note_no}"
            metadata = base_metadata(chunk_id, source, source_file, "note", article_id, article_title, section_id, section_title)
            metadata.update({"note_no": note_no, "term": term})
            chunks.append(Chunk(chunk_id, "teaching_index", source, source_file, "note", article_title, line, metadata))
            continue

        question_match = QUESTION_RE.match(line)
        if question_match and article_id and (section_id or line.endswith("（）") or "（" in line):
            if question_no != "section" or len(question_lines) > 1:
                flush_question(chunks, source, source_file, article_id, article_title, section_id, section_title, question_no, question_lines)
            question_no = question_match.group(1)
            question_lines = [line]
            inline_options = INLINE_OPTION_RE.findall(question_match.group(2))
            if inline_options:
                question_lines = [re.sub(r"\s*[A-D][.．、]?.*$", "", line).strip()]
                question_lines.extend(f"{label}. {text.strip()}" for label, text in inline_options if text.strip())
            continue

        if question_no:
            if OPTION_RE.match(line):
                question_lines.append(line)
                continue
            if question_no == "section" and not re.match(r"^(#|##|\[|[①②③④⑤⑥⑦⑧⑨⑩])", line):
                question_lines.append(line)
                continue
            if question_no != "section" and not re.match(r"^(#|##|\[|[①②③④⑤⑥⑦⑧⑨⑩])", line):
                question_lines.append(line)
                continue

        if article_id and not section_id and not line.startswith("##") and not line.startswith("【"):
            next_no = sum(
                1
                for chunk in chunks
                if chunk.metadata.get("article_id") == article_id and chunk.chunk_type == "paragraph"
            ) + 1
            chunk_id = f"{article_id}_p_{next_no:03d}"
            metadata = base_metadata(chunk_id, source, source_file, "paragraph", article_id, article_title)
            metadata["paragraph_no"] = next_no
            chunks.append(Chunk(chunk_id, "teaching_index", source, source_file, "paragraph", article_title, line, metadata))

    if question_no != "section" or len(question_lines) > 1:
        flush_question(chunks, source, source_file, article_id, article_title, section_id, section_title, question_no, question_lines)
    return chunks


def parse_answer_markdown(path: Path) -> list[Chunk]:
    source = relative_source(path)
    source_file = path.name
    chunks: list[Chunk] = []
    article_count = 0
    section_count = 0
    article_id: str | None = None
    article_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("# 发展汉语") or line.startswith("## 第"):
            continue
        article_match = re.match(r"^##\s*(文章[一二三四五六七八九十]+)\s*(.+)$", line)
        practical_match = re.match(r"^实用阅读\s+(.+?)\s+(.+)$", line)
        if article_match:
            article_count += 1
            section_count = 0
            article_id = article_id_from_count(article_count)
            article_title = f"{article_match.group(1)} {article_match.group(2)}"
            section_id = None
            section_title = None
            continue
        if practical_match:
            article_count += 1
            section_count = 1
            article_id = article_id_from_count(article_count)
            article_title = f"实用阅读 {practical_match.group(1)}"
            section_id = section_id_from_count(section_count)
            section_title = "根据选课通知选择正确答案"
            line = practical_match.group(2)

        section_match = re.match(r"^(?:##\s*)?([一二三四五六七八九十]+)、\s*(.+)$", line)
        if section_match and article_id:
            section_count += 1
            section_id = section_id_from_count(section_count)
            section_title = f"{section_match.group(1)}、{section_match.group(2).strip()}"
            answer_tail = section_match.group(2).strip()
        else:
            answer_tail = line

        if not article_id:
            continue
        matches = ANSWER_RE.findall(answer_tail)
        if matches:
            for question_no, answer in matches:
                answer_text = answer.strip()
                chunk_id = f"{article_id}_{section_id or 'section_000'}_a_{int(question_no):03d}"
                metadata = base_metadata(
                    chunk_id,
                    source,
                    source_file,
                    "answer",
                    article_id,
                    article_title,
                    section_id,
                    section_title,
                )
                metadata.update({"question_no": question_no, "answer": answer_text})
                chunks.append(
                    Chunk(chunk_id, "teaching_index", source, source_file, "answer", section_title or article_title, answer_text, metadata)
                )
        elif section_id and answer_tail and not answer_tail.startswith("##"):
            chunk_id = f"{article_id}_{section_id}_a_section"
            metadata = base_metadata(
                chunk_id,
                source,
                source_file,
                "answer",
                article_id,
                article_title,
                section_id,
                section_title,
            )
            metadata.update({"question_no": "section", "answer": answer_tail})
            chunks.append(Chunk(chunk_id, "teaching_index", source, source_file, "answer", section_title, answer_tail, metadata))
    return chunks


def parse_level_markdown(path: Path) -> list[Chunk]:
    source = relative_source(path)
    source_file = path.name
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    if "词汇" in source_file:
        blocks = re.split(r"\n(?=\d+\s*(?:【|\S+\s+[a-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü]))", text)
        seen_item_numbers: Counter[int] = Counter()
        for block in blocks:
            block = block.strip()
            bracket_match = re.match(r"^(\d+)\s*【([^】]+)】", block)
            plain_match = re.match(r"^(\d+)\s+(\S+)\s+", block)
            match = bracket_match or plain_match
            if not match:
                continue
            item_no = int(match.group(1))
            seen_item_numbers[item_no] += 1
            term = match.group(2).strip()
            occurrence = seen_item_numbers[item_no]
            occurrence_suffix = f"_{occurrence:02d}" if occurrence > 1 else ""
            chunk_id = f"{source_file.removesuffix('.md')}_vocab_{item_no:04d}{occurrence_suffix}"
            metadata = {
                "id": chunk_id,
                "index_name": "level_index",
                "source": source,
                "source_file": source_file,
                "chunk_type": "vocabulary",
                "item_no": item_no,
                "term": term,
                "level": "HSK4" if "HSK4" in source_file else "HSK5",
            }
            chunks.append(Chunk(chunk_id, "level_index", source, source_file, "vocabulary", term, block, metadata))
        return chunks

    tables = re.findall(r"<table>.*?</table>", text, flags=re.S)
    for item_no, table in enumerate(tables, start=1):
        plain = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", table)
        plain = re.sub(r"<[^>]+>", "", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        level_match = re.search(r"([一二三四五六]级)语法项目表", plain)
        level = level_match.group(1) if level_match else None
        chunk_id = f"HSK语法1-6_grammar_{item_no:03d}"
        metadata = {
            "id": chunk_id,
            "index_name": "level_index",
            "source": source,
            "source_file": source_file,
            "chunk_type": "grammar",
            "item_no": item_no,
            "level": level,
        }
        chunks.append(Chunk(chunk_id, "level_index", source, source_file, "grammar", level, plain, metadata))
    return chunks


def parse_source(path: Path, index_name: str) -> list[Chunk]:
    if index_name == "level_index":
        return parse_level_markdown(path)
    if path.name.endswith("参考答案.md"):
        return parse_answer_markdown(path)
    return parse_teaching_markdown(path)


def fetch_embedding_cache(connection: sqlite3.Connection, hashes: list[str], model: str) -> dict[str, list[float]]:
    if not hashes:
        return {}
    placeholders = ",".join("?" for _ in hashes)
    embedding_provider = get_embedding_provider()
    rows = connection.execute(
        f"""
        SELECT content_hash, embedding_json
        FROM knowledge_chunks
        WHERE embedding_provider = ?
          AND embedding_model = ?
          AND content_hash IN ({placeholders})
        """,
        [embedding_provider, model, *hashes],
    ).fetchall()
    return {row["content_hash"]: json.loads(row["embedding_json"]) for row in rows}


def store_source_chunks(path: Path, index_name: str, chunks: list[Chunk], force: bool = False) -> int:
    source = relative_source(path)
    source_hash = file_hash(path)
    embedding_provider = get_embedding_provider()
    embedding_model = get_embedding_model()
    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT source_hash, embedding_model, chunk_count
            FROM knowledge_sources
            WHERE source = ? AND embedding_provider = ?
            """,
            (source, embedding_provider),
        ).fetchone()
        existing_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM knowledge_chunks
            WHERE source = ? AND embedding_provider = ? AND embedding_model = ?
            """,
            (source, embedding_provider, embedding_model),
        ).fetchone()["count"]
        if (
            not force
            and existing
            and existing["source_hash"] == source_hash
            and existing["embedding_model"] == embedding_model
            and existing["chunk_count"] == len(chunks)
            and existing_count == len(chunks)
        ):
            return 0

        hashes = [chunk.content_hash for chunk in chunks]
        cache = {} if force else fetch_embedding_cache(connection, hashes, embedding_model)
        missing_chunks = [chunk for chunk in chunks if chunk.content_hash not in cache]
        if missing_chunks:
            embeddings = embed_texts([chunk.content for chunk in missing_chunks])
            for chunk, embedding in zip(missing_chunks, embeddings, strict=True):
                cache[chunk.content_hash] = embedding

        now = utc_now()
        connection.execute("DELETE FROM knowledge_chunks WHERE source = ?", (source,))
        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO knowledge_chunks (
                    id, index_name, source, source_file, chunk_type, heading, content,
                    content_hash, metadata_json, embedding_provider, embedding_model,
                    embedding_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.index_name,
                    chunk.source,
                    chunk.source_file,
                    chunk.chunk_type,
                    chunk.heading,
                    chunk.content,
                    chunk.content_hash,
                    json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    embedding_provider,
                    embedding_model,
                    json.dumps(cache[chunk.content_hash]),
                    now,
                ),
            )
        connection.execute(
            """
            INSERT INTO knowledge_sources (
                source, index_name, source_hash, embedding_provider, embedding_model,
                chunk_count, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                index_name = excluded.index_name,
                source_hash = excluded.source_hash,
                embedding_provider = excluded.embedding_provider,
                embedding_model = excluded.embedding_model,
                chunk_count = excluded.chunk_count,
                updated_at = excluded.updated_at
            """,
            (source, index_name, source_hash, embedding_provider, embedding_model, len(chunks), now),
        )
        return len(chunks)


def build_indexes(force: bool = False) -> dict[str, Any]:
    require_knowledge_files()
    init_db()
    result: dict[str, Any] = {"rebuilt_sources": [], "sources": [], "total_chunks": 0}
    for index_name, paths in (("teaching_index", TEACHING_FILES), ("level_index", LEVEL_FILES)):
        for path in paths:
            chunks = parse_source(path, index_name)
            rebuilt = store_source_chunks(path, index_name, chunks, force=force)
            result["total_chunks"] += len(chunks)
            source_summary = {
                "index_name": index_name,
                "source": relative_source(path),
                "chunk_count": len(chunks),
                "rebuilt": rebuilt > 0,
            }
            result["sources"].append(source_summary)
            if rebuilt > 0:
                result["rebuilt_sources"].append(source_summary)
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_index(query: str, index_name: str, top_k: int = 3) -> list[dict[str, Any]]:
    init_db()
    query_embedding = embed_texts([query])[0]
    embedding_provider = get_embedding_provider()
    embedding_model = get_embedding_model()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, index_name, source, source_file, chunk_type, heading, content,
                   content_hash, metadata_json, embedding_json
            FROM knowledge_chunks
            WHERE index_name = ? AND embedding_provider = ? AND embedding_model = ?
            """,
            (index_name, embedding_provider, embedding_model),
        ).fetchall()
    scored = []
    for row in rows:
        score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
        metadata = json.loads(row["metadata_json"])
        scored.append(
            {
                "score": score,
                "id": row["id"],
                "index_name": row["index_name"],
                "source": row["source"],
                "source_file": row["source_file"],
                "chunk_type": row["chunk_type"],
                "heading": row["heading"],
                "content": row["content"],
                "metadata": metadata,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]

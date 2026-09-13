from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.chat_service import handle_mock_chat
from app.db import CHAT_LOG_DIR, DB_PATH, init_db, utc_now
from prompt_loader import (
    WORKFLOW_FILES,
    build_prompt_bundle,
    detect_problem_type,
    load_kernel_prompt,
    load_workflow_prompt,
)


def main() -> None:
    init_db()
    result = handle_mock_chat("我昨天去图书馆学习中文早上。")
    prompt_bundle = build_prompt_bundle("我昨天去图书馆学习中文早上。")

    with sqlite3.connect(DB_PATH) as connection:
        messages = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at
            """,
            (result["conversation_id"],),
        ).fetchall()
        events = connection.execute(
            """
            SELECT event_type
            FROM agent_events
            WHERE conversation_id = ?
            """,
            (result["conversation_id"],),
        ).fetchall()

    log_path = Path(CHAT_LOG_DIR) / f"{utc_now()[:10]}.jsonl"
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if result["conversation_id"] in line
    ]

    assert [row[0] for row in messages] == ["user", "assistant"]
    assert events and events[0][0] == "mock_chat_completed"
    assert len(log_records) >= 3
    assert prompt_bundle["problem_type"] == "sentence_correction"
    assert "prompt/chatbot_sys_prompt.md" not in prompt_bundle["prompt"]
    assert 1200 <= len(load_kernel_prompt()) <= 1800
    for problem_type in WORKFLOW_FILES:
        assert load_workflow_prompt(problem_type)
    assert detect_problem_type("这个词是什么意思？") == "vocabulary_question"
    assert detect_problem_type("课文里这句话怎么理解？") == "reading_expression"
    assert detect_problem_type("哪个更自然？") == "expression_naturalness"

    print(f"conversation_id={result['conversation_id']}")
    print(f"db={DB_PATH}")
    print(f"jsonl={log_path}")
    print(f"messages={len(messages)} agent_events={len(events)} jsonl_records={len(log_records)}")
    print(f"problem_type={prompt_bundle['problem_type']}")
    print(f"kernel_chars={len(load_kernel_prompt())}")


if __name__ == "__main__":
    main()

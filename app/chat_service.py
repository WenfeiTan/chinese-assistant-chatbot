from __future__ import annotations

from collections.abc import Iterable

from app.db import ensure_conversation, save_agent_event, save_message
from prompt_loader import build_prompt_bundle


def build_mock_reply(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    if not cleaned:
        return "请先输入一个中文问题。"

    prompt_bundle = build_prompt_bundle(cleaned)
    return (
        f"我收到你的问题：“{cleaned}”。"
        f"这轮我会按“{prompt_bundle['problem_type']}”来处理，"
        "先看一个最核心的点。后续接入真实模型后，会使用 kernel、对应 workflow 和本轮 context 生成回答。"
    )


def chunk_text(text: str, size: int = 12) -> Iterable[str]:
    for start in range(0, len(text), size):
        yield text[start : start + size]


def handle_mock_chat(message: str, conversation_id: str | None = None) -> dict[str, str]:
    resolved_conversation_id = ensure_conversation(
        conversation_id,
        title=message.strip()[:40] or "Untitled conversation",
        metadata={"source": "mock_chat"},
    )
    save_message(
        resolved_conversation_id,
        "user",
        message,
        metadata={"source": "api_chat"},
    )

    prompt_bundle = build_prompt_bundle(message)
    reply = build_mock_reply(message)
    save_message(
        resolved_conversation_id,
        "assistant",
        reply,
        metadata={
            "source": "mock_chat",
            "model": "mock",
            "problem_type": prompt_bundle["problem_type"],
        },
    )
    save_agent_event(
        resolved_conversation_id,
        "mock_chat_completed",
        {
            "provider": "mock",
            "model": "mock",
            "rag_enabled": False,
            "prompt_mode": "kernel_workflow_runtime",
            "problem_type": prompt_bundle["problem_type"],
        },
    )

    return {"conversation_id": resolved_conversation_id, "reply": reply}

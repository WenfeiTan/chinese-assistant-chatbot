from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Chinese Assistant Chatbot", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    conversation_id = payload.conversation_id or f"conv_{uuid.uuid4().hex}"

    async def stream_response() -> AsyncIterator[str]:
        yield sse("metadata", {"conversation_id": conversation_id})

        for chunk in chunk_text(build_mock_reply(message)):
            await asyncio.sleep(0.08)
            yield sse("token", chunk)

        yield sse("done", {"conversation_id": conversation_id})

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def sse(event: str, data: dict[str, str] | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def build_mock_reply(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    return (
        f"我收到你的问题：“{cleaned}”。"
        "这是一个本地 mock 回复：我会先帮你看最核心的一点，"
        "后续分支再接入 RAG 和真实模型。"
    )


def chunk_text(text: str, size: int = 12) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]

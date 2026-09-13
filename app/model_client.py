from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any


EMBEDDING_PROVIDER = "gcp"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
MOCK_EMBEDDING_PROVIDER = "mock"
MOCK_EMBEDDING_MODEL = "mock-embedding-128"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv_if_present(path: Path = DOTENV_PATH) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_if_present()


def get_embedding_model() -> str:
    if is_mock_embeddings_enabled():
        return MOCK_EMBEDDING_MODEL
    return os.getenv("GCP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_provider() -> str:
    if is_mock_embeddings_enabled():
        return MOCK_EMBEDDING_PROVIDER
    return EMBEDDING_PROVIDER


def is_mock_embeddings_enabled() -> bool:
    return os.getenv("MOCK_EMBEDDINGS") == "1"


def _extract_embedding_values(embedding: Any) -> list[float]:
    values = getattr(embedding, "values", None)
    if values is not None:
        return [float(value) for value in values]
    if isinstance(embedding, dict):
        raw_values = embedding.get("values") or embedding.get("embedding")
        if raw_values is not None:
            return [float(value) for value in raw_values]
    if isinstance(embedding, list):
        return [float(value) for value in embedding]
    raise RuntimeError(f"Unsupported embedding response item: {type(embedding)!r}")


def _extract_result_embeddings(result: Any) -> list[list[float]]:
    embeddings = getattr(result, "embeddings", None)
    if embeddings is None and isinstance(result, dict):
        embeddings = result.get("embeddings")
    if embeddings is not None:
        return [_extract_embedding_values(embedding) for embedding in embeddings]

    embedding = getattr(result, "embedding", None)
    if embedding is None and isinstance(result, dict):
        embedding = result.get("embedding")
    if embedding is not None:
        return [_extract_embedding_values(embedding)]

    raise RuntimeError("GCP GenAI embedding response did not include embeddings.")


def _mock_embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in text:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if is_mock_embeddings_enabled():
        return [_mock_embedding(text) for text in texts]

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for real embeddings. "
            "Install dependencies and configure GEMINI_API_KEY, or set MOCK_EMBEDDINGS=1 for local verification."
        ) from exc

    client = genai.Client()
    model = get_embedding_model()

    result = client.models.embed_content(model=model, contents=texts)
    embeddings = _extract_result_embeddings(result)
    if len(embeddings) == len(texts):
        return embeddings

    if len(texts) == 1:
        raise RuntimeError(
            f"GCP GenAI returned {len(embeddings)} embeddings for 1 text."
        )

    per_text_embeddings: list[list[float]] = []
    for text in texts:
        single_result = client.models.embed_content(model=model, contents=text)
        single_embeddings = _extract_result_embeddings(single_result)
        if len(single_embeddings) != 1:
            raise RuntimeError(
                f"GCP GenAI returned {len(single_embeddings)} embeddings for a single text."
            )
        per_text_embeddings.append(single_embeddings[0])
    return per_text_embeddings

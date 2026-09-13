# chinese-assistant-chatbot

中文学习 Agent MVP 的最小聊天壳，并包含当前切片的双路 RAG embedding index。

## 当前分支范围

`feature/rag-embedding-index` 当前只包含：

- FastAPI 项目骨架
- 静态聊天页面
- `GET /api/health`
- `POST /api/chat` mock SSE streaming
- 读取 `知识库min/`
- 构建 `teaching_index` 和 `level_index`
- 教材按文章标题、自然段、注释、题型 section、question_block、answer 结构化切分
- embedding 通过 `app/model_client.py::embed_texts` 生成，并缓存到 SQLite
- 源文件 hash 变化时重建对应来源

当前版本不做聊天生成 RAG，不把教材题目接成练习册答疑主流程。

## 本地启动

使用项目内虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
uvicorn app.main:app --reload
```

打开：

```text
http://127.0.0.1:8000
```

## 验证方式

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

预期返回：

```json
{"status":"ok"}
```

聊天流式接口：

```bash
curl -N \
  -H "Content-Type: application/json" \
  -d '{"message":"我昨天去图书馆学习中文早上。","history":[]}' \
  http://127.0.0.1:8000/api/chat
```

预期看到多段 `data:` SSE 输出。网页端输入消息后，应看到用户气泡、Agent 气泡逐段出现、发送中状态和自动滚动。

## RAG index 验证

真实 embedding 使用 GCP GenAI：

```bash
export GEMINI_API_KEY=...
export GCP_EMBEDDING_MODEL=gemini-embedding-2
uv run python scripts/verify_rag_index.py --force
```

没有 API key 时，可只验证结构化切分、SQLite 缓存、metadata 和 top-k 流程：

```bash
uv run python scripts/verify_rag_index.py --mock-embeddings
```

常用参数：

```bash
uv run python scripts/verify_rag_index.py --mock-embeddings --force --top-k 5 --query "我昨天去图书馆学习中文早上。"
```

脚本会输出 chunk 数量、metadata 样例、当前生效的 embedding provider/model/dimension，以及 `teaching_index` / `level_index` 各自的 top-k 检索结果。第二次运行若源文件 hash 和 embedding model 未变化，`rebuilt_sources` 应为空。

如果从 `--mock-embeddings` 切到真实 GCP embedding，第一次真实运行务必带 `--force`，避免复用旧缓存。真实检索结果的 top-k 分数不应全部为 `0.0000`。

# Customer Support Bot

End-to-end autonomous support agent using LangChain, RAG, and local LLMs (Ollama).
Includes memory management, context-aware escalation logic, safety guardrails,
an evaluation stack, and a minimal web UI.

<img width="1078" height="1094" alt="Screenshot 2026-02-07 at 12 09 03 PM" src="https://github.com/user-attachments/assets/5e8fb34e-57ec-43a7-af31-78443f81fc82" />


## What This App Does
- Maintains session memory (last N turns, summary stub)
- Retrieves relevant RAG chunks and builds prompts
- Applies guardrails (PII + restricted topics)
- Escalates based on confidence, guardrails, user request/frustration
- Serves responses via API/UI with per-session state in memory

## Architecture Overview
```
API/UI -> SupportAgent -> Memory
                   -> Guardrails
                   -> EscalationLogic
                   -> RAG Pipeline -> Vector Store
                   -> LLM (Ollama)
```

## Requirements
- Python 3.10+
- Ollama running locally (for real LLM responses)

## Quick Start
1. Install dependencies:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -e ".[dev]"`
2. Start Ollama and pull models:
   - `ollama serve`
   - `ollama pull llama3`
   - `ollama pull nomic-embed-text`
3. Run a demo request:
   - `python -m app.main --query "How do I reset my password?"`

## API + UI
1. Start the API server:
   - `python -m app.main --serve`
2. Open the UI in your browser:
   - `http://127.0.0.1:8000`
3. Ingest documents:
   - `curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"documents":["Reset your password via Settings > Security."]}'`
4. Sample RAG document:
   - `data/docs/sample_support.md`
5. Ingest files from `data/docs` automatically:
   - `curl -X POST http://127.0.0.1:8000/ingest-path -H "Content-Type: application/json" -d '{}'`

## Configuration
Environment variables are prefixed with `CSB_` and follow Pydantic nested
notation, for example:
- `CSB_OLLAMA__MODEL=llama3`
- `CSB_RAG__TOP_K=4`
- `CSB_RAG__MIN_SCORE=0.15`
- `CSB_RAG__VECTOR_STORE=chroma`
- `CSB_RAG__PERSIST_DIRECTORY=data/vector_store`
- `CSB_ESCALATION__CONFIDENCE_THRESHOLD=0.55`
- `CSB_API_HOST=127.0.0.1`
- `CSB_API_PORT=8000`

## RAG Retrieval Behavior
Only retrieved chunks are added to the prompt:
- `top_k` limits retrieval count
- `min_score` filters out low-similarity chunks

## Evaluation
The evaluation stack includes:
- Synthetic tests: `src/eval/synthetic.py`
- Shadow mode validator: `src/eval/shadow.py`
- Report aggregation: `src/eval/report.py`

Shadow mode runs a candidate model alongside the live model on the same inputs
without exposing its responses to users, enabling safe comparison.

## Tests
Run all tests:
```
pytest
```

## Troubleshooting
- `pytest: command not found`: use `python -m pytest`
- Ollama not responding: ensure `ollama serve` is running
- UI loads but responses are generic: ingest docs and confirm embeddings model is pulled
- `/ingest-path` returns Not Found: restart the server after updates

## Configuration
All defaults are defined in `src/app/config.py`. You can override via environment
variables or direct instantiation of `AppConfig`.

## Project Layout
- `src/app`: entrypoint + configuration
- `src/agent`: agent orchestration, memory, guardrails, escalation
- `src/rag`: RAG pipeline + vector store stubs
- `src/eval`: synthetic + shadow evaluation and reporting
- `tests`: minimal decision flow tests

## Evaluation
Use `src/eval/synthetic.py` for generating synthetic cases and scoring, and
`src/eval/shadow.py` for validating a live agent against a baseline.

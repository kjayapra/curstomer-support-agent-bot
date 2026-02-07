from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.agent import SupportAgent
from agent.escalation import EscalationLogic
from agent.guardrails import GuardrailEngine
from agent.memory import ConversationMemory
from app.config import AppConfig
from rag.index import VectorStore, create_vector_store
from rag.pipeline import RagPipeline

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover
    OllamaLLM = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    escalated: bool
    escalation_reason: Optional[str]
    confidence: float
    session_id: str


class IngestRequest(BaseModel):
    documents: list[str]


class IngestPathRequest(BaseModel):
    path: Optional[str] = None


@dataclass
class AgentRegistry:
    config: AppConfig
    vector_store: VectorStore
    agents: Dict[str, SupportAgent]

    def get_agent(self, session_id: str) -> SupportAgent:
        if session_id not in self.agents:
            memory = ConversationMemory(max_turns=6, summary_trigger=10)
            guardrails = GuardrailEngine(config=self.config.guardrails)
            escalation = EscalationLogic(config=self.config.escalation)
            llm = None
            if OllamaLLM is not None:
                llm = OllamaLLM(model=self.config.ollama.model)
            rag = RagPipeline(config=self.config.rag, vector_store=self.vector_store, llm=llm)
            self.agents[session_id] = SupportAgent(
                config=self.config,
                memory=memory,
                rag=rag,
                guardrails=guardrails,
                escalation=escalation,
            )
        return self.agents[session_id]


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    app = FastAPI(title="Customer Support Bot")
    config = config or AppConfig()
    vector_store = create_vector_store(config.rag)
    registry = AgentRegistry(config=config, vector_store=vector_store, agents={})

    base_dir = Path(__file__).resolve().parents[2]
    static_dir = base_dir / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(base_dir / "web" / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(payload: IngestRequest) -> dict:
        vector_store.add(payload.documents)
        return {"ingested": len(payload.documents)}

    @app.post("/ingest-path")
    def ingest_path(payload: IngestPathRequest) -> dict:
        target = base_dir / (payload.path or "data/docs")
        target = target.resolve()
        if base_dir not in target.parents and target != base_dir:
            return {"error": "path_outside_project"}
        if not target.exists():
            return {"error": "path_not_found"}
        files = [
            p for p in target.rglob("*") if p.suffix.lower() in {".md", ".txt"}
        ]
        documents = [p.read_text(encoding="utf-8") for p in files]
        vector_store.add(documents)
        return {"ingested": len(documents), "files": [str(p) for p in files]}

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        session_id = payload.session_id or str(uuid.uuid4())
        agent = registry.get_agent(session_id)
        result = agent.handle_message(payload.message)
        return ChatResponse(
            response=result.response,
            escalated=result.escalated,
            escalation_reason=result.escalation_reason,
            confidence=result.confidence,
            session_id=session_id,
        )

    return app

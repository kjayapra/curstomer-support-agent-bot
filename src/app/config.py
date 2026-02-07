from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


class OllamaConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3")
    embedding_model: str = Field(default="nomic-embed-text")


class RAGConfig(BaseModel):
    top_k: int = Field(default=4, ge=1)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    vector_store: str = Field(default="chroma")
    persist_directory: str = Field(default="data/vector_store")


class GuardrailConfig(BaseModel):
    block_pii: bool = Field(default=True)
    allow_sensitive_actions: bool = Field(default=False)


class EscalationConfig(BaseModel):
    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_turns_without_resolution: int = Field(default=3, ge=1)


class EvalConfig(BaseModel):
    latency_budget_ms: int = Field(default=3000, ge=1)


class AppConfig(BaseSettings):
    ollama: OllamaConfig = OllamaConfig()
    rag: RAGConfig = RAGConfig()
    guardrails: GuardrailConfig = GuardrailConfig()
    escalation: EscalationConfig = EscalationConfig()
    eval: EvalConfig = EvalConfig()
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)

    model_config = ConfigDict(env_prefix="CSB_")

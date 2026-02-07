from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import RAGConfig
from rag.index import RetrievedChunk, VectorStore

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover - optional dependency at runtime
    OllamaLLM = None


@dataclass
class RagPipeline:
    config: RAGConfig
    vector_store: VectorStore
    llm: Optional[object] = None

    def __post_init__(self) -> None:
        if self.llm is None and OllamaLLM is not None:
            self.llm = OllamaLLM(model="llama3")

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        results = self.vector_store.search(query, top_k=self.config.top_k)
        return [chunk for chunk in results if chunk.score >= self.config.min_score]

    def build_prompt(self, query: str, context: str, chunks: List[RetrievedChunk]) -> str:
        sources = "\n".join(f"- {chunk.content}" for chunk in chunks)
        return (
            "You are a customer support agent.\n"
            f"Conversation context:\n{context}\n"
            f"Knowledge snippets:\n{sources}\n"
            f"User question: {query}\n"
            "Respond with a helpful, concise answer.\n"
        )

    def generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        chunks = self.retrieve(query)
        prompt = self.build_prompt(query, context, chunks)

        if self.llm is None:
            return (
                "Thanks for reaching out. I can help with that, but I need more details.",
                0.5,
            )

        response = self.llm.invoke(prompt)
        # If the LLM is available, allow answers without retrieval while
        # still boosting confidence when relevant context exists.
        confidence = min(0.9, 0.6 + (0.1 * len(chunks)))
        return response, confidence

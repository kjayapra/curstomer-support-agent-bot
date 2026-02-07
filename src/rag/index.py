from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from app.config import RAGConfig

try:
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover
    Chroma = None
    OllamaEmbeddings = None


@dataclass
class RetrievedChunk:
    content: str
    score: float


class VectorStore:
    def add(self, documents: List[str]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        raise NotImplementedError


@dataclass
class InMemoryVectorStore(VectorStore):
    _documents: List[str] = field(default_factory=list)

    def add(self, documents: List[str]) -> None:
        self._documents.extend(documents)

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        # Naive retrieval by keyword overlap as a placeholder.
        ranked: List[Tuple[str, float]] = []
        query_terms = set(query.lower().split())
        for doc in self._documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            if overlap > 0:
                ranked.append((doc, overlap / max(len(query_terms), 1)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [RetrievedChunk(content=doc, score=score) for doc, score in ranked[:top_k]]


@dataclass
class ChromaVectorStore(VectorStore):
    persist_directory: str
    collection_name: str = "support_docs"
    _store: object = field(init=False)

    def __post_init__(self) -> None:
        if Chroma is None or OllamaEmbeddings is None:
            raise RuntimeError("Chroma or Ollama embeddings are unavailable.")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )

    def add(self, documents: List[str]) -> None:
        if documents:
            self._store.add_texts(documents)
            self._store.persist()

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            RetrievedChunk(content=doc.page_content, score=score)
            for doc, score in results
        ]


def create_vector_store(config: RAGConfig) -> VectorStore:
    if config.vector_store == "chroma":
        if Chroma is None or OllamaEmbeddings is None:
            return InMemoryVectorStore()
        persist_path = Path(config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        return ChromaVectorStore(persist_directory=str(persist_path))
    return InMemoryVectorStore()

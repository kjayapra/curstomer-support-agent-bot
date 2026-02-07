from __future__ import annotations

import argparse
import time

from agent.agent import SupportAgent
from agent.escalation import EscalationLogic
from agent.guardrails import GuardrailEngine
from agent.memory import ConversationMemory
from app.config import AppConfig
from app.server import create_app
from rag.index import create_vector_store
from rag.pipeline import RagPipeline

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover
    OllamaLLM = None


def build_agent(config: AppConfig) -> SupportAgent:
    memory = ConversationMemory(max_turns=6, summary_trigger=10)
    vector_store = create_vector_store(config.rag)
    llm = OllamaLLM(model=config.ollama.model) if OllamaLLM is not None else None
    rag = RagPipeline(config=config.rag, vector_store=vector_store, llm=llm)
    guardrails = GuardrailEngine(config=config.guardrails)
    escalation = EscalationLogic(config=config.escalation)
    return SupportAgent(
        config=config,
        memory=memory,
        rag=rag,
        guardrails=guardrails,
        escalation=escalation,
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Customer Support Bot demo")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="User query")
    mode.add_argument("--serve", action="store_true", help="Run API server")
    args = parser.parse_args()

    config = AppConfig()

    if args.serve:
        import uvicorn

        app = create_app(config)
        uvicorn.run(app, host=config.api_host, port=config.api_port)
        return

    agent = build_agent(config)

    start = time.time()
    result = agent.handle_message(args.query)
    elapsed_ms = int((time.time() - start) * 1000)

    print(f"Response: {result.response}")
    print(f"Escalated: {result.escalated} ({result.escalation_reason})")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Latency: {elapsed_ms}ms")


if __name__ == "__main__":
    run_cli()

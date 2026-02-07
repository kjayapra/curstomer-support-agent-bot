from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig, EscalationConfig, GuardrailConfig
from agent.agent import SupportAgent
from agent.escalation import EscalationLogic
from agent.guardrails import GuardrailEngine
from agent.memory import ConversationMemory


@dataclass
class StubRag:
    answer: str
    confidence: float

    def generate_answer(self, query: str, context: str):
        return self.answer, self.confidence


def build_agent(confidence: float, allow_sensitive: bool = False) -> SupportAgent:
    config = AppConfig(
        guardrails=GuardrailConfig(allow_sensitive_actions=allow_sensitive),
        escalation=EscalationConfig(confidence_threshold=0.6),
    )
    memory = ConversationMemory()
    rag = StubRag(answer="Here is a safe response.", confidence=confidence)
    guardrails = GuardrailEngine(config.guardrails)
    escalation = EscalationLogic(config.escalation)
    return SupportAgent(
        config=config,
        memory=memory,
        rag=rag,
        guardrails=guardrails,
        escalation=escalation,
    )


def test_guardrail_escalation():
    agent = build_agent(confidence=0.9)
    result = agent.handle_message("I want a refund now.")
    assert result.escalated is True
    assert result.escalation_reason == "guardrail_triggered"


def test_low_confidence_escalation():
    agent = build_agent(confidence=0.4, allow_sensitive=True)
    result = agent.handle_message("How do I reset my password?")
    assert result.escalated is True
    assert result.escalation_reason == "low_confidence"


def test_successful_response():
    agent = build_agent(confidence=0.9, allow_sensitive=True)
    result = agent.handle_message("How do I reset my password?")
    assert result.escalated is False
    assert result.response


def test_user_requests_human_escalation():
    agent = build_agent(confidence=0.9, allow_sensitive=True)
    result = agent.handle_message("I want to talk to a human")
    assert result.escalated is True
    assert result.escalation_reason == "user_requested_human"

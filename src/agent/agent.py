from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import AppConfig
from agent.escalation import EscalationDecision, EscalationLogic
from agent.guardrails import GuardrailEngine
from agent.memory import ConversationMemory
from rag.pipeline import RagPipeline


@dataclass
class AgentResult:
    response: str
    escalated: bool
    escalation_reason: Optional[str]
    confidence: float


class SupportAgent:
    def __init__(
        self,
        config: AppConfig,
        memory: ConversationMemory,
        rag: RagPipeline,
        guardrails: GuardrailEngine,
        escalation: EscalationLogic,
    ) -> None:
        self.config = config
        self.memory = memory
        self.rag = rag
        self.guardrails = guardrails
        self.escalation = escalation
        self._unresolved_turns = 0

    def handle_message(self, user_input: str) -> AgentResult:
        guardrail = self.guardrails.evaluate(user_input)
        safe_input = guardrail.redacted_input

        context = self.memory.context()
        answer, confidence = self.rag.generate_answer(safe_input, context=context)

        decision = self.escalation.evaluate(
            confidence=confidence,
            guardrail_reasons=guardrail.reasons,
            unresolved_turns=self._unresolved_turns,
            user_message=user_input,
        )

        if decision.escalate:
            response = (
                "Your request needs a specialist. I will escalate this to a human agent."
            )
            self._unresolved_turns += 1
        else:
            response = answer
            self._unresolved_turns = 0

        self.memory.add_turn(user_input, response)
        if self.memory.should_summarize():
            self.memory.update_summary("Conversation summary pending.")

        return AgentResult(
            response=response,
            escalated=decision.escalate,
            escalation_reason=decision.reason,
            confidence=confidence,
        )

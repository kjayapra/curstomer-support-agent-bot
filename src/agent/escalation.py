from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.config import EscalationConfig


ESCALATION_KEYWORDS = {
    "human",
    "representative",
    "agent",
    "specialist",
    "talk to a person",
    "escalate",
}

FRUSTRATION_KEYWORDS = {
    "frustrating",
    "not helpful",
    "upset",
    "angry",
}


@dataclass
class EscalationDecision:
    escalate: bool
    reason: Optional[str]


class EscalationLogic:
    def __init__(self, config: EscalationConfig) -> None:
        self.config = config

    def evaluate(
        self,
        confidence: float,
        guardrail_reasons: List[str],
        unresolved_turns: int,
        user_message: str,
    ) -> EscalationDecision:
        lowered = user_message.lower()
        if any(keyword in lowered for keyword in ESCALATION_KEYWORDS):
            return EscalationDecision(True, "user_requested_human")
        if any(keyword in lowered for keyword in FRUSTRATION_KEYWORDS):
            return EscalationDecision(True, "user_frustrated")
        if guardrail_reasons:
            return EscalationDecision(True, "guardrail_triggered")
        if confidence < self.config.confidence_threshold:
            return EscalationDecision(True, "low_confidence")
        if unresolved_turns >= self.config.max_turns_without_resolution:
            return EscalationDecision(True, "too_many_turns")
        return EscalationDecision(False, None)

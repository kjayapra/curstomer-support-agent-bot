from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.config import GuardrailConfig


PII_PATTERNS = [
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
]

RESTRICTED_KEYWORDS = {
    "refund",
    "chargeback",
    "legal threat",
    "account takeover",
}


@dataclass
class GuardrailResult:
    safe: bool
    reasons: List[str]
    redacted_input: str


class GuardrailEngine:
    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config

    def evaluate(self, text: str) -> GuardrailResult:
        reasons: List[str] = []
        redacted = text

        if self.config.block_pii:
            for pattern in PII_PATTERNS:
                if pattern.search(redacted):
                    reasons.append("pii_detected")
                    redacted = pattern.sub("[REDACTED]", redacted)

        lowered = text.lower()
        if not self.config.allow_sensitive_actions:
            for keyword in RESTRICTED_KEYWORDS:
                if keyword in lowered:
                    reasons.append("restricted_action")
                    break

        safe = len(reasons) == 0
        return GuardrailResult(safe=safe, reasons=reasons, redacted_input=redacted)

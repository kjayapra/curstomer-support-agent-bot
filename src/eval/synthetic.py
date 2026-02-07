from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class SyntheticCase:
    query: str
    expected_escalation: bool


def generate_cases() -> List[SyntheticCase]:
    return [
        SyntheticCase(query="How do I reset my password?", expected_escalation=False),
        SyntheticCase(query="I want a refund now", expected_escalation=True),
        SyntheticCase(query="My account was hacked", expected_escalation=True),
        SyntheticCase(query="Where can I download invoices?", expected_escalation=False),
    ]


def run_synthetic_eval(
    predict: Callable[[str], bool]
) -> List[bool]:
    cases = generate_cases()
    results = []
    for case in cases:
        results.append(predict(case.query) == case.expected_escalation)
    return results

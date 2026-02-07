from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List


@dataclass
class EvalReport:
    accuracy: float
    autonomy_rate: float
    avg_latency_ms: float


def build_report(
    correctness: List[bool],
    escalations: List[bool],
    latencies_ms: List[int],
) -> EvalReport:
    accuracy = mean(correctness) if correctness else 0.0
    autonomy_rate = 1.0 - mean(escalations) if escalations else 0.0
    avg_latency_ms = mean(latencies_ms) if latencies_ms else 0.0
    return EvalReport(
        accuracy=accuracy,
        autonomy_rate=autonomy_rate,
        avg_latency_ms=avg_latency_ms,
    )

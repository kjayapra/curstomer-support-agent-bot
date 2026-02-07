from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class ShadowResult:
    query: str
    live_escalated: bool
    baseline_escalated: bool
    match: bool


def run_shadow_mode(
    queries: List[str],
    live_predict: Callable[[str], bool],
    baseline_predict: Callable[[str], bool],
) -> List[ShadowResult]:
    results: List[ShadowResult] = []
    for query in queries:
        live = live_predict(query)
        baseline = baseline_predict(query)
        results.append(
            ShadowResult(
                query=query,
                live_escalated=live,
                baseline_escalated=baseline,
                match=live == baseline,
            )
        )
    return results

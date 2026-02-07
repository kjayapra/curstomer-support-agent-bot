from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ConversationMemory:
    max_turns: int = 6
    summary_trigger: int = 10
    _turns: List[Tuple[str, str]] = field(default_factory=list)
    _summary: str = ""

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def update_summary(self, summary: str) -> None:
        self._summary = summary.strip()

    def should_summarize(self) -> bool:
        return len(self._turns) >= self.summary_trigger

    def context(self) -> str:
        history = "\n".join(
            f"User: {user}\nAssistant: {assistant}" for user, assistant in self._turns
        )
        if self._summary:
            return f"Summary: {self._summary}\n{history}"
        return history

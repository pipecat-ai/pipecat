"""Turn timeline for voice session transcripts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TurnRole = Literal["user", "assistant"]


def iso_to_ts(ts: str) -> float:
    """Parse an ISO 8601 timestamp string to Unix seconds."""
    return datetime.fromisoformat(ts).timestamp()


class Turn(BaseModel):
    """A single user or assistant turn with optional timing and metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: TurnRole
    content: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    latency: float | None = None
    interrupted: bool | None = None


class TurnManager:
    """Records turns and session timing for transcript export."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.recording_started_at: float | None = None
        self.recording_ended_at: float | None = None
        self.current_user_start: float | None = None
        self.current_assistant_start: float | None = None
        self._latency: float | None = None

    def start_user_turn(self, timestamp: str) -> None:
        self.current_user_start = iso_to_ts(timestamp)

    def end_user_turn(self, content: str, timestamp: str) -> None:
        end_ts = iso_to_ts(timestamp)
        start_ts = self.current_user_start
        turn = Turn(
            role="user",
            content=content,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        self.turns.append(turn)
        self.current_user_start = None

    def start_assistant_turn(self, timestamp: str, latency_seconds: float) -> None:
        self.current_assistant_start = iso_to_ts(timestamp)
        self._latency = latency_seconds

    def end_assistant_turn(
        self, content: str, is_interrupted: bool, timestamp: str
    ) -> None:
        end_ts = iso_to_ts(timestamp)
        start_ts = self.current_assistant_start
        turn = Turn(
            role="assistant",
            content=content,
            start_ts=start_ts,
            end_ts=end_ts,
            interrupted=is_interrupted,
            latency=self._latency,
        )
        self.turns.append(turn)
        self.current_assistant_start = None
        self._latency = None

    def to_json(self) -> dict:
        """Build a JSON-serializable session payload (encode with ``json.dumps`` at the call site)."""
        return {
            "recording_started_at": self.recording_started_at,
            "recording_ended_at": self.recording_ended_at,
            "turns": [turn.model_dump(exclude_none=False) for turn in self.turns],
        }

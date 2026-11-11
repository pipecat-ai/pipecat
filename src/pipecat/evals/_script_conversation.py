#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A script's observed conversation, independent of expectation matching."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field

from pipecat.evals.conversation import EvalConversation
from pipecat.evals.serializer import EvalFunctionCallCancelFrame
from pipecat.frames.frames import (
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallInProgressFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)


@dataclass
class _Entry:
    at: float
    kind: str
    text: str = ""
    name: str = ""
    tool_call_id: str = ""
    arguments: dict = field(default_factory=dict)
    parts: list[str] = field(default_factory=list)


class _ScriptConversation:
    """Keep submitted inputs and observed bot output in source order.

    A delayed transcription belongs where its audio began, before a later
    submitted input. Matcher queues may discard that speech without removing
    it from this record. Snapshots use the existing conversation-file format.
    """

    def __init__(self, *, bot_audio: bool):
        self._bot_audio = bot_audio
        self._entries: list[_Entry] = []
        self._response: _Entry | None = None

    @contextmanager
    def submitting(self, text: str | None) -> Iterator[None]:
        """Reserve the input's place before enqueueing; retain it only on success.

        This records submission to the client, not receipt by the target bot.
        Preparation such as file loading or synthesis must precede this scope.
        """
        entry = _Entry(time.monotonic(), "user", text=text) if text else None
        if entry is not None:
            self._entries.append(entry)
        try:
            yield
        except BaseException:
            if entry is not None:
                self._entries = [item for item in self._entries if item is not entry]
            raise

    def observe(self, frame: Frame) -> None:
        """Keep text and tool frames before the matcher filters them."""
        at = time.monotonic()
        if isinstance(frame, (FunctionCallInProgressFrame, ExternalFunctionCallInProgressFrame)):
            self._entries.append(
                _Entry(
                    at,
                    "tool",
                    name=frame.function_name,
                    tool_call_id=frame.tool_call_id,
                    arguments=deepcopy(frame.arguments or {}),
                )
            )
        elif isinstance(
            frame,
            (FunctionCallCancelFrame, EvalFunctionCallCancelFrame, ExternalFunctionCallCancelFrame),
        ):
            self._entries.append(
                _Entry(at, "cancel", name=frame.function_name, tool_call_id=frame.tool_call_id)
            )
        elif not self._bot_audio:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._response = _Entry(at, "text")
                self._entries.append(self._response)
            elif isinstance(frame, LLMTextFrame):
                if self._response is None:
                    self._response = _Entry(at, "text")
                    self._entries.append(self._response)
                self._response.parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._response = None

    def speech(self, text: str, *, started_at: float | None = None) -> None:
        """Keep stable recognized speech, using a known source time when available."""
        if self._bot_audio and text.strip():
            self._entries.append(
                _Entry(time.monotonic() if started_at is None else started_at, "speech", text=text)
            )

    def snapshot(self) -> list[dict]:
        """Return a detached conversation, including the bot's observed prefix."""
        conversation = EvalConversation()
        calls: dict[str, dict] = {}

        # Stable sorting preserves observation order for equal source times.
        for entry in sorted(self._entries, key=lambda item: item.at):
            if entry.kind == "text":
                conversation.add_assistant_message("".join(entry.parts))
            elif entry.kind == "speech":
                conversation.add_assistant_message(entry.text)
            elif entry.kind == "user":
                conversation.add_user_message(entry.text)
            elif entry.kind == "tool":
                conversation.add_tool_call(entry.name, entry.arguments)
                if entry.tool_call_id:
                    calls[entry.tool_call_id] = conversation.messages[-1]
            elif entry.kind == "cancel":
                if entry.tool_call_id:
                    if call := calls.get(entry.tool_call_id):
                        call["cancelled"] = True
                else:
                    matches = [
                        message
                        for message in conversation.messages
                        if message["role"] == "tool_call"
                        and message["name"] == entry.name
                        and not message["cancelled"]
                    ]
                    if len(matches) == 1:
                        matches[0]["cancelled"] = True
        conversation.end_bot_turn()
        return deepcopy(conversation.messages)

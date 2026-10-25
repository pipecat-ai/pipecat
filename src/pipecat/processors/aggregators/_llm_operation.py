#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Captured context operations for services that commit after provider acceptance."""

import copy
import time
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    SystemFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage


@dataclass
class _LLMContextBindFrame(SystemFrame):
    """Bind shared context before an acknowledged service releases buffered input."""

    context: LLMContext
    interruptible: bool = field(default=False, init=False)


@dataclass
class _LLMContextOperationFrame(DataFrame):
    """A proposal with original request identity and retained record identities."""

    request: Frame
    context: LLMContext
    base: tuple[LLMContextMessage, ...]
    messages: list[LLMContextMessage] | None = None
    retained: tuple[int | None, ...] = ()
    validation_error: str | None = None
    captured_at: float = field(default_factory=time.monotonic)
    interruptible: bool = field(default=False, init=False)

    @classmethod
    def capture(cls, request: Frame, context: LLMContext) -> "_LLMContextOperationFrame":
        """Capture mutable values without identifying records by text or position."""
        base = tuple(context.get_messages())
        try:
            snapshot = copy.deepcopy(list(base))
            if isinstance(request, LLMMessagesTransformFrame):
                candidates = copy.deepcopy(list(base))
                identities = {id(message): i for i, message in enumerate(candidates)}
                proposed = request.transform(candidates)
                if not isinstance(proposed, list):
                    raise ValueError("A context transform must return a list of messages")
            else:
                identities = {id(message): i for i, message in enumerate(base)}
                proposed = (
                    list(base)
                    if isinstance(request, (LLMContextFrame, LLMRunFrame))
                    else getattr(request, "messages", None)
                )
            retained = []
            for message in proposed or []:
                index = identities.get(id(message))
                retained.append(index if index is not None and message == snapshot[index] else None)
            messages = copy.deepcopy(proposed)
            captured = copy.copy(request)
            if isinstance(captured, (LLMMessagesAppendFrame, LLMMessagesUpdateFrame)):
                captured.messages = copy.deepcopy(messages or [])
            if isinstance(captured, LLMSetToolsFrame):
                captured.tools = copy.deepcopy(captured.tools)
            return cls(
                request=captured,
                context=context,
                base=base,
                messages=messages,
                retained=tuple(retained),
            )
        except Exception as error:
            return cls(
                request=copy.copy(request), context=context, base=base, validation_error=str(error)
            )

    def commit_messages(
        self, messages: list[LLMContextMessage], *, append: bool, base: tuple | None = None
    ) -> None:
        """Commit once while retaining records that arrived after capture."""
        if append:
            self.context.add_messages(messages)
        else:
            base_ids = {id(message) for message in (self.base if base is None else base)}
            arrivals = [m for m in self.context.get_messages() if id(m) not in base_ids]
            self.context.set_messages([*messages, *arrivals])

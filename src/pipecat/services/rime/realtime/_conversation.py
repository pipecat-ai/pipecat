#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism connection admission and retained response ownership."""

import asyncio
import math
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from rimelabs_sdk import Rime, RimeResourceLimitError, RimeUnavailableError
from rimelabs_sdk import realtime as r


class RimeStartupParams(BaseModel):
    """Limits for opening a new Prism conversation.

    Parameters:
        max_retries: Additional attempts after eligible HTTP refusals.
        retry_interval: Minimum seconds between attempts.
        attempt_timeout: Maximum seconds for one connection and initialization.
        total_timeout: Maximum seconds for all attempts and delays.
    """

    model_config = ConfigDict(allow_inf_nan=False)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_interval: float = Field(default=2, ge=0)
    attempt_timeout: float = Field(default=30, gt=0)
    total_timeout: float = Field(default=60, gt=0)


@asynccontextmanager
async def connect(client: Rime, policy: RimeStartupParams, **settings: Any):
    """Retry refused openings without replaying an established conversation."""
    deadline = asyncio.get_running_loop().time() + policy.total_timeout
    connection = None
    for attempt in range(policy.max_retries + 1):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Prism startup deadline expired")
        connection = client.realtime.connect(**settings)
        try:
            async with asyncio.timeout(min(remaining, policy.attempt_timeout)):
                session = await connection.__aenter__()
            break
        except (RimeUnavailableError, RimeResourceLimitError) as error:
            if attempt == policy.max_retries or error.status_code not in {
                408,
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            delay = policy.retry_interval
            retry_after = error.retry_after_s
            if isinstance(retry_after, (int, float)) and math.isfinite(retry_after):
                delay = max(delay, retry_after)
            if delay >= deadline - asyncio.get_running_loop().time():
                raise
            await asyncio.sleep(delay)
    try:
        yield session
    finally:
        assert connection is not None
        await connection.__aexit__(None, None, None)


@dataclass
class Output:
    """One generated message and its estimated output position."""

    ref: r.OutputRef
    context_id: str = field(default_factory=lambda: uuid4().hex)
    text: str = ""
    audio_ms: float = 0
    started_at: float | None = None
    closed: bool = False
    recorded: bool = False


@dataclass
class Response:
    """Response state retained until its operations release ownership."""

    ref: r.ResponseRef
    instruction_item_id: str | None = None
    outputs: dict[str, Output] = field(default_factory=dict)
    calls: set[str] = field(default_factory=set)
    input_ids: tuple[str, ...] = ()
    ended: bool = False
    stopped: bool = False
    audio_response: bool = True
    done: asyncio.Event = field(default_factory=asyncio.Event)
    continued: bool = False
    tools_disabled: bool = False
    hold: r.ResponseHold | None = None
    operations: int = 0
    reply_owed: bool = False
    report_instruction: str | None = None


@dataclass
class Tool:
    """One application call with ordered progress and final submissions."""

    event: r.ToolCall
    submitted: bool = False
    final: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Conversation:
    """Own response references independently of completed output retention."""

    def __init__(self):
        self.responses: OrderedDict[str, Response] = OrderedDict()
        self.calls: dict[str, Tool] = {}
        self.settings_lock = asyncio.Lock()
        self.suppressed_revisions: set[int] = set()
        self.policy_floor = 0
        self.suppressing_tools = False

    def start(self, sdk: r.RealtimeSession, event: r.ResponseStarted) -> Response:
        """Retain a response while generation, tools, or requests need it."""
        self.prune()
        protected = sum(value.hold is not None for value in self.responses.values())
        if protected >= 128:
            raise RimeResourceLimitError("Prism protected response limit exceeded")
        if event.settings_revision is not None and event.settings_revision < self.policy_floor:
            raise RimeResourceLimitError("Prism response tool policy expired")
        response = Response(
            event.response,
            instruction_item_id=event.instruction_item_id,
            input_ids=event.input_item_ids,
            hold=sdk.hold_response(event.response),
            tools_disabled=(
                event.settings_revision in self.suppressed_revisions
                if event.settings_revision is not None
                else self.suppressing_tools
            ),
        )
        self.responses[event.response.response_id] = response
        return response

    def release(self, response: Response) -> None:
        """Release SDK ownership only after all dependent work settles."""
        if response.ended and not response.calls and not response.operations and response.hold:
            response.hold.release()
            response.hold = None

    def prune(self) -> None:
        """Retain at most 128 ordinary responses without evicting protected owners."""
        ordinary = [key for key, value in self.responses.items() if value.hold is None]
        for key in ordinary[:-127]:
            del self.responses[key]
            self.calls = {
                call_id: call
                for call_id, call in self.calls.items()
                if call.event.call.response_id != key
            }

    def suppress(self, info: r.SessionInfo) -> None:
        """Remember the tool policy selected by replies at this revision."""
        if info.settings_revision is not None:
            self.suppressed_revisions.add(info.settings_revision)
            if len(self.suppressed_revisions) > 128:
                oldest = min(self.suppressed_revisions)
                self.suppressed_revisions.remove(oldest)
                self.policy_floor = oldest + 1

    def close(self) -> None:
        """Release all references when the conversation closes."""
        for response in self.responses.values():
            if response.hold:
                response.hold.release()
                response.hold = None

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Strict audio admission holds the simulated caller until target validation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.evals.client import _BotFrameSink
from pipecat.evals.client_transport import EvalClientTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.serializer import eval_capabilities_message
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.evals.simulation_session import EvalSimulationSession
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport
from tests.test_evals_session import _ScriptedPersonaLLM


@pytest.mark.asyncio
async def test_strict_audio_persona_does_not_activate_before_capability_admission():
    session = EvalSimulationSession(
        EvalSimulationScenario(
            name="gate", persona="caller", goal="ask", success="answer", user_audio=True
        ),
        "ws://localhost:0",
        params=EvalSessionParams(require_audio_support=True),
        persona_llm=_ScriptedPersonaLLM({}),
        judge=None,
    )
    client = session._client
    client._processors(client._transport())
    sink = client._sink
    sink.push_frame = AsyncMock()
    assert not client._handshake_over.is_set()
    assert client._audio_capabilities is None
    with patch.object(FrameProcessor, "process_frame", new=AsyncMock()):
        for frame in (
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="Hello caller"),
            LLMFullResponseEndFrame(),
        ):
            await sink.process_frame(frame, FrameDirection.DOWNSTREAM)
    llm_requests = [
        c for c in sink.push_frame.await_args_list if isinstance(c.args[0], LLMContextFrame)
    ]
    assert not llm_requests, "persona was activated before strict audio target admission"


@pytest.mark.asyncio
async def test_pending_greeting_canary_is_cleared_on_cleanup_without_activation():
    sink = _BotFrameSink(
        EvalEventStream(bot_audio=False, trace=EvalTrace()),
        persona=SimpleNamespace(hung_up=False),
        answer_enabled=False,
    )
    sink.push_frame = AsyncMock()
    frame = LLMContextFrame(LLMContext([{"role": "user", "content": "PRIVATE_REVIEW_GREETING"}]))
    await sink._answer(frame)
    assert sink._pending_answer is frame
    with patch.object(FrameProcessor, "cleanup", new=AsyncMock()):
        await sink.cleanup()
    assert sink._pending_answer is None
    await sink.allow_answers()
    sink.push_frame.assert_not_called()


@pytest.mark.asyncio
async def test_hung_up_persona_does_not_replay_pending_greeting():
    persona = SimpleNamespace(hung_up=False)
    sink = _BotFrameSink(
        EvalEventStream(bot_audio=False, trace=EvalTrace()), persona=persona, answer_enabled=False
    )
    sink.push_frame = AsyncMock()
    await sink._answer(LLMContextFrame(LLMContext()))
    persona.hung_up = True
    await sink.allow_answers()
    assert sink._pending_answer is None
    sink.push_frame.assert_not_called()


@pytest.mark.asyncio
async def test_supported_admission_replays_pending_context_exactly_once():
    sink = _BotFrameSink(
        EvalEventStream(bot_audio=False, trace=EvalTrace()),
        persona=SimpleNamespace(hung_up=False),
        answer_enabled=False,
    )
    sink.push_frame = AsyncMock()
    first, latest = LLMContextFrame(LLMContext()), LLMContextFrame(LLMContext())
    await sink._answer(first)
    await sink._answer(latest)
    await sink.allow_answers()
    await sink.allow_answers()
    sink.push_frame.assert_awaited_once_with(latest)
    assert sink._pending_answer is None


@pytest.mark.asyncio
async def test_late_capabilities_cannot_upgrade_a_completed_handshake():
    transport = EvalClientTransport("ws://localhost:0")
    transport._bot_ready = True
    with patch.object(RTVIClientTransport, "_on_message", new=AsyncMock()) as base:
        await transport._on_message(
            None, eval_capabilities_message(audio_input=True, audio_output=True)
        )
    base.assert_awaited_once()
    assert transport.audio_capabilities is None

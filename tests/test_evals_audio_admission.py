#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio admission holds the simulated caller until the target's audio support is checked."""

import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf

from pipecat.evals.client import EvalBotUnavailableError, _BotFrameSink
from pipecat.evals.client_transport import EvalClientTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.script import EvalExpectation, EvalScriptScenario, EvalScriptTurn
from pipecat.evals.script_session import EvalScriptSession
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
from tests.test_evals_session import _FakeRTVIServer, _free_port, _rtvi, _ScriptedPersonaLLM


def _simulation_session(check: str) -> EvalSimulationSession:
    return EvalSimulationSession(
        EvalSimulationScenario(
            name="gate", persona="caller", goal="ask", success="answer", user_audio=True
        ),
        "ws://localhost:0",
        params=EvalSessionParams(audio_support_check=check),
        persona_llm=_ScriptedPersonaLLM({}),
        judge=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["when_advertised", "required"])
async def test_audio_persona_does_not_activate_before_capability_admission(check):
    session = _simulation_session(check)
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
    assert not llm_requests, "persona was activated before audio target admission"


def test_an_unchecked_audio_persona_answers_at_once():
    client = _simulation_session("off")._client
    client._processors(client._transport())
    assert client._sink._answer_enabled


def test_a_simulation_result_carries_the_audio_support_outcome():
    session = _simulation_session("when_advertised")
    session._client._audio_support = "unverified"
    assert session._result(time.monotonic(), []).audio_support == "unverified"


class _AdvertisingRTVIServer(_FakeRTVIServer):
    """A fake target that advertises its audio directions on connect, as eval transports do."""

    def __init__(self, port: int, advertisement: str | None):
        super().__init__(port)
        self.advertisement = advertisement

    async def _handler(self, ws):
        if self.advertisement is not None:
            await ws.send(self.advertisement)
        await super()._handler(ws)


def _advertisement(advertised: bool | str | None) -> str | None:
    """What the fake target sends before bot-ready: nothing, its input direction, or garbage."""
    if advertised is None:
        return None
    if advertised == "unreadable":
        message = json.loads(eval_capabilities_message(audio_input=True, audio_output=False))
        message["data"]["audio_input"] = "true"
        return json.dumps(message)
    return eval_capabilities_message(audio_input=advertised, audio_output=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check,advertised,refusal,audio_support",
    [
        ("when_advertised", None, None, "unverified"),
        ("when_advertised", "unreadable", None, "unverified"),
        ("when_advertised", True, None, "verified"),
        ("when_advertised", False, "audio_unsupported", "unsupported"),
        ("required", None, "audio_unverified", "unverified"),
        ("required", "unreadable", "audio_unverified", "unverified"),
        ("required", True, None, "verified"),
        ("off", False, None, None),
    ],
)
async def test_an_audio_run_against_each_kind_of_target(
    tmp_path, check, advertised, refusal, audio_support
):
    rate = 16000
    t = np.linspace(0, 0.5, rate // 2, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    sf.write(str(tmp_path / "hi.wav"), tone, rate)
    server = _AdvertisingRTVIServer(_free_port(), _advertisement(advertised))
    server.on_text(
        "hello there",
        _rtvi("bot-llm-started"),
        _rtvi("bot-llm-text", {"text": "Hi!"}),
        _rtvi("bot-llm-stopped"),
    )
    scenario = EvalScriptScenario(
        name="audio-turn",
        bot_audio=False,
        user_audio=True,
        turns=[
            EvalScriptTurn(
                user="hello there",
                audio=str(tmp_path / "hi.wav"),
                expect=[EvalExpectation(event="llm_started", within_ms=2000)],
            )
        ],
    )
    await server.start()
    try:
        result = await EvalScriptSession.from_scenario(
            scenario, server.url, params=EvalSessionParams(audio_support_check=check)
        ).run()
    finally:
        await server.stop()

    assert result.audio_support == audio_support
    sent = b"".join(
        base64.b64decode(m["data"]["base64Audio"])
        for m in server.received
        if m.get("type") == "raw-audio"
    )
    if refusal is None:
        assert result.passed, [str(f) for f in result.failures]
        assert tone.tobytes() in sent
        if audio_support is not None:
            assert any(f"audio support {audio_support}" in line for line in result.debug_log)
    else:
        assert [(f.event_name, f.kind) for f in result.failures] == [("<bot-ready>", refusal)]
        assert tone.tobytes() not in sent


def _held_greeting_client(check: str):
    """A simulation's client past bot-ready, with a greeting answer held and the sink spied on."""
    client = _simulation_session(check)._client
    client._processors(client._transport())
    client._handshake_over.set()
    client._send_eval = AsyncMock()
    sink = client._sink
    sink.push_frame = AsyncMock()
    greeting = LLMContextFrame(LLMContext([{"role": "assistant", "content": "Hello caller"}]))
    sink._pending_answer = greeting
    return client, sink, greeting


@pytest.mark.asyncio
async def test_an_unverified_target_gets_the_held_greeting_answered_once():
    client, sink, greeting = _held_greeting_client("when_advertised")
    await client.handshake()
    assert client.audio_support == "unverified"
    sink.push_frame.assert_awaited_once_with(greeting)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check,capabilities",
    [
        ("required", None),
        ("when_advertised", SimpleNamespace(audio_input=False, audio_output=True)),
    ],
)
async def test_a_refused_simulation_never_answers(check, capabilities):
    client, sink, greeting = _held_greeting_client(check)
    client._audio_capabilities = capabilities
    with pytest.raises(EvalBotUnavailableError):
        await client.handshake()
    sink.push_frame.assert_not_awaited()
    assert not sink._answer_enabled
    assert sink._pending_answer is greeting


@pytest.mark.asyncio
async def test_cleanup_drops_a_held_greeting():
    sink = _BotFrameSink(
        EvalEventStream(bot_audio=False, trace=EvalTrace()),
        persona=SimpleNamespace(hung_up=False),
        answer_enabled=False,
    )
    sink.push_frame = AsyncMock()
    frame = LLMContextFrame(LLMContext([{"role": "user", "content": "Hello caller"}]))
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

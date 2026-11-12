#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An audio run checks the audio directions its target transport advertises."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pipecat.evals.client import EvalBotUnavailableError
from pipecat.evals.client_transport import EvalClientTransport
from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.evals.results import AUDIO_SUPPORT_OUTCOMES, EvalScriptResult, EvalSimulationResult
from pipecat.evals.script import EvalScriptScenario
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.serializer import (
    EvalConnectionFlags,
    EvalSerializer,
    eval_capabilities_message,
    parse_eval_capabilities,
)
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.suite import (
    EvalRun,
    _result_from_dict,
    _scenario_record,
    _simulation_record,
    _simulation_result_from_dict,
)
from pipecat.evals.transport import EvalTransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport

AUDIO_RUNS = [(True, False), (False, True), (True, True)]
CHECKS = ["when_advertised", "required"]


def _client(user_audio=False, bot_audio=False, check="required", capabilities=None):
    session = EvalScriptSession(
        EvalScriptScenario(
            name="audio-support", turns=[], user_audio=user_audio, bot_audio=bot_audio
        ),
        "ws://localhost:0",
        params=EvalSessionParams(audio_support_check=check),
    )
    client = session._client
    client._audio_capabilities = capabilities
    client._handshake_over.set()
    client._send_eval = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("user_audio,bot_audio", AUDIO_RUNS)
async def test_required_check_refuses_an_unadvertised_target(user_audio, bot_audio):
    client = _client(user_audio=user_audio, bot_audio=bot_audio)
    with pytest.raises(EvalBotUnavailableError) as raised:
        await client.handshake()
    assert raised.value.kind == "audio_unverified"
    assert "did not advertise" in str(raised.value)
    assert client.audio_support == "unverified"
    client._send_eval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_audio,bot_audio", AUDIO_RUNS)
async def test_when_advertised_check_runs_an_unadvertised_target_unverified(user_audio, bot_audio):
    client = _client(user_audio=user_audio, bot_audio=bot_audio, check="when_advertised")
    await client.handshake()
    assert client.audio_support == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize(
    "user_audio,bot_audio,audio_input,audio_output,names",
    [
        (True, False, False, True, ["audio_in_enabled"]),
        (False, True, True, False, ["audio_out_enabled"]),
        (True, True, True, False, ["audio_out_enabled"]),
        (True, True, False, False, ["audio_in_enabled", "audio_out_enabled"]),
    ],
)
async def test_a_direction_advertised_as_off_is_refused(
    check, user_audio, bot_audio, audio_input, audio_output, names
):
    client = _client(
        user_audio=user_audio,
        bot_audio=bot_audio,
        check=check,
        capabilities=SimpleNamespace(audio_input=audio_input, audio_output=audio_output),
    )
    with pytest.raises(EvalBotUnavailableError) as raised:
        await client.handshake()
    assert raised.value.kind == "audio_unsupported"
    for name in names:
        assert name in str(raised.value)
    assert client.audio_support == "unsupported"
    client._send_eval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("user_audio,bot_audio", AUDIO_RUNS)
async def test_an_advertised_direction_verifies_the_run(check, user_audio, bot_audio):
    client = _client(
        user_audio=user_audio,
        bot_audio=bot_audio,
        check=check,
        capabilities=SimpleNamespace(audio_input=user_audio, audio_output=bot_audio),
    )
    await client.handshake()
    assert client.audio_support == "verified"


@pytest.mark.asyncio
@pytest.mark.parametrize("check", CHECKS)
async def test_a_text_run_checks_nothing(check):
    client = _client(
        check=check, capabilities=SimpleNamespace(audio_input=False, audio_output=False)
    )
    await client.handshake()
    assert client.audio_support is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capabilities", [None, SimpleNamespace(audio_input=False, audio_output=False)]
)
async def test_off_runs_any_target_unchecked(capabilities):
    client = _client(user_audio=True, bot_audio=True, check="off", capabilities=capabilities)
    await client.handshake()
    assert client.audio_support is None


def test_the_check_is_off_unless_asked_for():
    assert EvalSessionParams().audio_support_check == "off"


def test_an_unknown_check_is_rejected():
    with pytest.raises(ValidationError):
        EvalSessionParams(audio_support_check="sometimes")


@pytest.mark.parametrize("outcome", [None, *AUDIO_SUPPORT_OUTCOMES])
def test_a_suite_keeps_the_audio_support_outcome(outcome):
    script = EvalScriptResult(scenario_name="s", passed=True, audio_support=outcome)
    simulation = EvalSimulationResult(simulation_name="s", succeeded=True, audio_support=outcome)
    # A suite worker writes its result as JSON; the suite rebuilds it.
    assert _result_from_dict(json.loads(json.dumps(asdict(script)))).audio_support == outcome
    rebuilt = _simulation_result_from_dict(json.loads(json.dumps(asdict(simulation))))
    assert rebuilt.audio_support == outcome

    run = EvalRun(bot="bot.py", scenario="s", scenario_path=Path("s.yaml"))
    run.result = script
    assert _scenario_record(run, {})["audio_support"] == outcome
    run.result = simulation
    assert _simulation_record(run, {})["audio_support"] == outcome


@pytest.mark.parametrize(
    "audio_input,audio_output", [(False, False), (True, False), (False, True), (True, True)]
)
def test_audio_capabilities_roundtrip(audio_input, audio_output):
    message = json.loads(
        eval_capabilities_message(audio_input=audio_input, audio_output=audio_output)
    )
    parsed = parse_eval_capabilities(message)
    assert parsed.audio_input is audio_input
    assert parsed.audio_output is audio_output


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"audio_input": "true", "audio_output": True},
        {"audio_input": True, "audio_output": 1},
    ],
)
def test_malformed_capabilities_cannot_claim_support(payload):
    message = json.loads(eval_capabilities_message(audio_input=True, audio_output=True))
    message["data"] = payload
    assert parse_eval_capabilities(message) is None


@pytest.mark.asyncio
async def test_client_reads_capabilities_before_the_bot_ready_callback():
    transport = EvalClientTransport("ws://localhost:0")
    message = eval_capabilities_message(audio_input=True, audio_output=False)
    with patch.object(RTVIClientTransport, "_on_message", new=AsyncMock()) as base_message:
        await transport._on_message(None, message)
        base_message.assert_not_awaited()
        assert transport.audio_capabilities.audio_input is True
        assert transport.audio_capabilities.audio_output is False
        await transport._on_message(None, '{"type":"bot-ready"}')
        base_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_latest_advertisement_invalidates_previous_support():
    transport = EvalClientTransport("ws://localhost:0")
    message = eval_capabilities_message(audio_input=True, audio_output=True)
    await transport._on_message(None, message)
    assert transport.audio_capabilities is not None
    invalid = json.loads(message)
    invalid["data"]["audio_input"] = "true"
    await transport._on_message(None, json.dumps(invalid))
    assert transport.audio_capabilities is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio_input,audio_output", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_target_advertises_configured_directions_before_connected(audio_input, audio_output):
    websocket = MagicMock()
    websocket.send_text = AsyncMock()
    transport = EvalFastAPIWebsocketTransport(
        websocket,
        EvalTransportParams(
            audio_in_enabled=audio_input,
            audio_out_enabled=audio_output,
            serializer=EvalSerializer(),
        ),
        EvalConnectionFlags(),
    )
    transport.input().push_frame = AsyncMock()

    async def connected(_websocket):
        websocket.send_text.assert_awaited_once()

    with patch.object(FastAPIWebsocketTransport, "_on_client_connected", side_effect=connected):
        await transport._on_client_connected(websocket)
    parsed = parse_eval_capabilities(json.loads(websocket.send_text.await_args.args[0]))
    assert parsed.audio_input is audio_input
    assert parsed.audio_output is audio_output

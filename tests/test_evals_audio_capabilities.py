#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An audio run can require evidence that its target transport supports audio."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.evals.client import EvalBotUnavailableError
from pipecat.evals.client_transport import EvalClientTransport
from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.evals.script import EvalScriptScenario
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.serializer import (
    EvalConnectionFlags,
    EvalSerializer,
    eval_capabilities_message,
    parse_eval_capabilities,
)
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport


def _client(user_audio=False, bot_audio=False, require=True, capabilities=None):
    session = EvalScriptSession(
        EvalScriptScenario(
            name="audio-support", turns=[], user_audio=user_audio, bot_audio=bot_audio
        ),
        "ws://localhost:0",
        params=EvalSessionParams(require_audio_support=require),
    )
    client = session._client
    client._audio_capabilities = capabilities
    client._handshake_over.set()
    client._send_eval = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("user_audio,bot_audio", [(True, False), (False, True), (True, True)])
async def test_required_audio_support_refuses_unadvertised_target(user_audio, bot_audio):
    client = _client(user_audio=user_audio, bot_audio=bot_audio)
    with pytest.raises(EvalBotUnavailableError) as raised:
        await client.handshake()
    assert raised.value.kind == "error"
    assert "audio" in str(raised.value)
    client._send_eval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_audio,bot_audio,audio_input,audio_output",
    [(True, False, False, True), (False, True, True, False), (True, True, True, False)],
)
async def test_required_direction_must_be_configured(
    user_audio, bot_audio, audio_input, audio_output
):
    client = _client(
        user_audio=user_audio,
        bot_audio=bot_audio,
        capabilities=SimpleNamespace(audio_input=audio_input, audio_output=audio_output),
    )
    with pytest.raises(EvalBotUnavailableError) as raised:
        await client.handshake()
    assert raised.value.kind == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_audio,bot_audio", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_configured_transport_supports_each_combination(user_audio, bot_audio):
    client = _client(
        user_audio=user_audio,
        bot_audio=bot_audio,
        capabilities=SimpleNamespace(audio_input=True, audio_output=True),
    )
    await client.handshake()


@pytest.mark.asyncio
async def test_text_handshake_does_not_require_new_target_metadata():
    await _client().handshake()


@pytest.mark.asyncio
async def test_existing_audio_clients_can_use_an_older_target():
    await _client(user_audio=True, bot_audio=True, require=False).handshake()


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

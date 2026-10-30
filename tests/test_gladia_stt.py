#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN, incoming=None):
        self.state = state
        self.sent = []
        self.closed = False
        self._incoming = incoming or []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._incoming:
            yield json.dumps(message)


def _connected_service(incoming=None):
    """Build a service holding an open fake socket, without touching the network."""
    service = GladiaSTTService(api_key="test-key")
    service._setup = frame_processor_setup(TaskManager())
    websocket = _FakeWebsocket(incoming=incoming)
    service._websocket = websocket
    service._connection_active = True
    return service, websocket


def _message_types(websocket):
    return [message["type"] for message in websocket.sent]


@pytest.mark.asyncio
async def test_stop_sends_stop_recording_while_the_socket_is_open():
    service, websocket = _connected_service()

    await service.stop(EndFrame())

    assert _message_types(websocket) == ["stop_recording"]
    assert websocket.closed


@pytest.mark.asyncio
async def test_stop_disconnects_once():
    service, _ = _connected_service()
    service._disconnect_websocket = AsyncMock(wraps=service._disconnect_websocket)

    await service.stop(EndFrame())

    service._disconnect_websocket.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_does_not_send_stop_recording():
    service, websocket = _connected_service()

    await service.cancel(CancelFrame())

    assert _message_types(websocket) == []
    assert websocket.closed


@pytest.mark.asyncio
async def test_audio_chunk_not_acknowledged_reports_error():
    """A rejected audio_chunk carries an ``error`` and no ``acknowledged: true``.

    The rejection is reported upstream as an error.
    """
    service, _ = _connected_service(
        incoming=[
            {
                "type": "audio_chunk",
                "acknowledged": False,
                "error": {"message": "quota exceeded"},
                "data": None,
            }
        ]
    )
    service.push_error = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    assert "quota exceeded" in service.push_error.call_args.kwargs["error_msg"]


@pytest.mark.asyncio
async def test_translation_addon_error_reports_error_instead_of_crashing():
    """A failed translation addon call arrives with ``error`` set and ``data: null``.

    The failure is reported upstream as an error and the receive loop keeps
    running, so the connection is not torn down.
    """
    service, _ = _connected_service(
        incoming=[
            {
                "type": "translation",
                "error": {"message": "translation addon failed"},
                "data": None,
            }
        ]
    )
    service.push_error = AsyncMock()
    service.push_frame = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    assert "translation addon failed" in service.push_error.call_args.kwargs["error_msg"]
    service.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_translation_without_error_still_pushes_translation_frame():
    """A translation with ``error: null`` pushes a ``TranslationFrame``."""
    service, _ = _connected_service(
        incoming=[
            {
                "type": "translation",
                "error": None,
                "data": {
                    "original_language": "en",
                    "translated_utterance": {"language": "fr", "text": "Bonjour"},
                },
            }
        ]
    )
    service.push_error = AsyncMock()
    service.push_frame = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_not_awaited()
    service.push_frame.assert_awaited_once()
    pushed_frame = service.push_frame.call_args.args[0]
    assert pushed_frame.text == "Bonjour"


@pytest.mark.asyncio
async def test_vad_stop_sends_finalize_when_enabled():
    """With ``finalize_on_vad_stop``, the end of speech asks for the final right away."""
    service, websocket = _connected_service()
    service._finalize_on_vad_stop = True
    service.push_frame = AsyncMock()

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert _message_types(websocket) == ["finalize"]
    assert service._finalize_requested


@pytest.mark.asyncio
async def test_vad_stop_sends_nothing_by_default():
    """Off by default: an API without ``finalize`` gets nothing new."""
    service, websocket = _connected_service()
    service.push_frame = AsyncMock()

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert _message_types(websocket) == []


def _final(text, *, from_finalize=False):
    data = {
        "id": "00_00000000",
        "is_final": True,
        "utterance": {"text": text, "language": "en", "start": 0.0, "end": 1.0},
    }
    if from_finalize:
        data["from_finalize"] = True
    return {"type": "transcript", "data": data}


@pytest.mark.asyncio
async def test_a_final_from_finalize_is_marked_finalized():
    """The final Gladia closed for our request is pushed as finalized."""
    service, _ = _connected_service(incoming=[_final("hello there", from_finalize=True)])
    service.request_finalize()
    pushed = []

    async def push(frame, *_args, **_kwargs):
        pushed.append(frame)

    service.push_frame = push
    service.emit_stt_usage_metrics = AsyncMock()

    await service._receive_messages()

    assert service._finalize_pending
    assert any(isinstance(f, TranscriptionFrame) for f in pushed)


@pytest.mark.asyncio
async def test_a_final_without_from_finalize_is_not_confirmed():
    service, _ = _connected_service(incoming=[_final("hello there")])
    service.request_finalize()
    service.push_frame = AsyncMock()
    service.emit_stt_usage_metrics = AsyncMock()

    await service._receive_messages()

    assert not service._finalize_pending
    assert service._finalize_requested


@pytest.mark.asyncio
async def test_nothing_to_finalize_reports_ttfb_at_once():
    """``finalized: false``: the final already went out, TTFB is reported now."""
    service, _ = _connected_service(
        incoming=[
            {
                "type": "finalize",
                "acknowledged": True,
                "error": None,
                "data": {"channel": 0, "finalized": False, "utterance_id": None},
            }
        ]
    )
    service.stop_ttfb_metrics = AsyncMock()
    service._cancel_ttfb_timeout = AsyncMock()

    await service._receive_messages()

    service.stop_ttfb_metrics.assert_awaited_once()
    service._cancel_ttfb_timeout.assert_awaited_once()

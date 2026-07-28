#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SmallestSTTService v4 Pulse WebSocket behavior."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.smallest.stt import SmallestSTTService, SmallestSTTSettings
from pipecat.transcriptions.language import Language

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(**kwargs) -> SmallestSTTService:
    return SmallestSTTService(api_key="test-key", **kwargs)


def _transcript_msg(text: str, is_final: bool, is_last: bool = False, language: str = "en") -> str:
    return json.dumps(
        {
            "transcript": text,
            "is_final": is_final,
            "is_last": is_last,
            "language": language,
            "session_id": "sess-test",
        }
    )


async def _drive_receive(service: SmallestSTTService, messages: list[str]):
    """Run _receive_messages over a scripted stream, capturing pushed frames."""
    pushed: list = []

    async def fake_push_frame(frame, *args, **kwargs):
        pushed.append(frame)

    async def noop(*args, **kwargs):
        pass

    async def fake_ws():
        for msg in messages:
            yield msg

    service.push_frame = fake_push_frame
    service.stop_processing_metrics = noop
    service._handle_transcription = noop
    service._get_websocket = fake_ws

    await service._receive_messages()
    return pushed


# ---------------------------------------------------------------------------
# Unit tests – URL construction
# ---------------------------------------------------------------------------


def test_websocket_url_uses_new_v4_endpoint():
    """The WebSocket URL must use /waves/v1/stt/live with model=pulse."""
    service = _make_service()

    captured_url = None

    async def fake_connect(url, **kwargs):
        nonlocal captured_url
        captured_url = url
        ws = MagicMock()
        ws.state = MagicMock()
        return ws

    async def fake_call_event_handler(*args, **kwargs):
        pass

    service._call_event_handler = fake_call_event_handler

    with patch(
        "pipecat.services.smallest.stt.websocket_connect",
        side_effect=fake_connect,
    ):
        asyncio.run(service._connect_websocket())

    assert captured_url is not None
    assert "/waves/v1/stt/live" in captured_url
    assert "model=pulse" in captured_url
    assert "get_text" not in captured_url


def test_websocket_url_includes_required_params():
    """The URL must include language, encoding, sample_rate, word_timestamps."""
    service = _make_service(
        settings=SmallestSTTSettings(
            language=Language.HI,
            word_timestamps=True,
        ),
    )

    captured_url = None

    async def fake_connect(url, **kwargs):
        nonlocal captured_url
        captured_url = url
        ws = MagicMock()
        ws.state = MagicMock()
        return ws

    async def fake_call_event_handler(*args, **kwargs):
        pass

    service._call_event_handler = fake_call_event_handler

    # sample_rate is a read-only property populated by StartFrame; mock it.
    with (
        patch.object(type(service), "sample_rate", new_callable=PropertyMock, return_value=16000),
        patch(
            "pipecat.services.smallest.stt.websocket_connect",
            side_effect=fake_connect,
        ),
    ):
        asyncio.run(service._connect_websocket())

    assert "language=hi" in captured_url
    assert "encoding=linear16" in captured_url
    assert "sample_rate=16000" in captured_url
    assert "word_timestamps=true" in captured_url


def test_websocket_url_default_feature_params():
    """endpointing and format default on; keywords defaults to empty (boosting off)."""
    service = _make_service()

    captured_url = None

    async def fake_connect(url, **kwargs):
        nonlocal captured_url
        captured_url = url
        ws = MagicMock()
        ws.state = MagicMock()
        return ws

    async def fake_call_event_handler(*args, **kwargs):
        pass

    service._call_event_handler = fake_call_event_handler

    with (
        patch.object(type(service), "sample_rate", new_callable=PropertyMock, return_value=16000),
        patch(
            "pipecat.services.smallest.stt.websocket_connect",
            side_effect=fake_connect,
        ),
    ):
        asyncio.run(service._connect_websocket())

    assert "endpointing=true" in captured_url
    assert "format=true" in captured_url
    assert "keywords=" in captured_url
    assert "keywords=%3A" not in captured_url  # no leftover keyword pairs by default


def test_websocket_url_includes_custom_feature_params():
    """endpointing, keywords, and format should reflect user-provided settings."""
    service = _make_service(
        settings=SmallestSTTSettings(
            endpointing=False,
            keywords="NVIDIA:2,Blackwell:1",
            format=False,
        ),
    )

    captured_url = None

    async def fake_connect(url, **kwargs):
        nonlocal captured_url
        captured_url = url
        ws = MagicMock()
        ws.state = MagicMock()
        return ws

    async def fake_call_event_handler(*args, **kwargs):
        pass

    service._call_event_handler = fake_call_event_handler

    with (
        patch.object(type(service), "sample_rate", new_callable=PropertyMock, return_value=16000),
        patch(
            "pipecat.services.smallest.stt.websocket_connect",
            side_effect=fake_connect,
        ),
    ):
        asyncio.run(service._connect_websocket())

    assert "endpointing=false" in captured_url
    assert "format=false" in captured_url
    assert "keywords=NVIDIA%3A2%2CBlackwell%3A1" in captured_url


# ---------------------------------------------------------------------------
# Unit tests – finalize on VAD stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vad_stop_sends_finalize():
    """VADUserStoppedSpeakingFrame must send a finalize message to flush the transcript."""
    from websockets.protocol import State

    from pipecat.frames.frames import VADUserStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    service = _make_service()

    sent_messages: list[str] = []
    ws = MagicMock()
    ws.state = State.OPEN
    ws.send = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))
    service._websocket = ws

    with patch.object(SmallestSTTService.__bases__[0], "process_frame", new=AsyncMock()):
        await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert len(sent_messages) == 1
    assert json.loads(sent_messages[0]) == {"type": "finalize"}


@pytest.mark.asyncio
async def test_vad_stop_does_nothing_when_websocket_closed():
    """No message should be sent if the websocket isn't open."""
    from websockets.protocol import State

    from pipecat.frames.frames import VADUserStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    service = _make_service()

    ws = MagicMock()
    ws.state = State.CLOSED
    ws.send = AsyncMock()
    service._websocket = ws

    with patch.object(SmallestSTTService.__bases__[0], "process_frame", new=AsyncMock()):
        await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    ws.send.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests – response parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interim_transcript_produces_interim_frame():
    service = _make_service()
    messages = [_transcript_msg("hello", is_final=False)]
    pushed = await _drive_receive(service, messages)
    assert len(pushed) == 1
    assert isinstance(pushed[0], InterimTranscriptionFrame)
    assert pushed[0].text == "hello"


@pytest.mark.asyncio
async def test_final_transcript_produces_transcription_frame():
    service = _make_service()
    messages = [_transcript_msg("hello world", is_final=True, is_last=True)]
    pushed = await _drive_receive(service, messages)
    assert len(pushed) == 1
    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "hello world"


@pytest.mark.asyncio
async def test_empty_transcript_produces_no_frame():
    service = _make_service()
    messages = [_transcript_msg("", is_final=True, is_last=True)]
    pushed = await _drive_receive(service, messages)
    assert pushed == []


@pytest.mark.asyncio
async def test_is_last_without_text_produces_no_frame():
    """is_last=True with empty transcript should not push any frame."""
    service = _make_service()
    messages = [json.dumps({"is_final": True, "is_last": True, "transcript": "", "language": "en"})]
    pushed = await _drive_receive(service, messages)
    assert pushed == []


@pytest.mark.asyncio
async def test_multiple_interims_then_final():
    """A realistic sequence: two interim results followed by a final."""
    service = _make_service()
    messages = [
        _transcript_msg("hel", is_final=False),
        _transcript_msg("hello", is_final=False),
        _transcript_msg("hello there", is_final=True, is_last=True),
    ]
    pushed = await _drive_receive(service, messages)
    assert len(pushed) == 3
    assert isinstance(pushed[0], InterimTranscriptionFrame)
    assert isinstance(pushed[1], InterimTranscriptionFrame)
    assert isinstance(pushed[2], TranscriptionFrame)
    assert pushed[2].text == "hello there"


@pytest.mark.asyncio
async def test_language_propagated_in_frame():
    """Language from the response should be set on the pushed frame."""
    service = _make_service()
    messages = [_transcript_msg("नमस्ते", is_final=True, is_last=True, language="hi")]
    pushed = await _drive_receive(service, messages)
    assert len(pushed) == 1
    assert pushed[0].language == "hi"


# ---------------------------------------------------------------------------
# Unit tests – receive task lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_creates_receive_task_when_none_exists():
    """_connect must start a receive task when none is running yet."""
    from websockets.protocol import State

    service = _make_service()
    service._receive_task = None

    created_tasks: list = []

    async def fake_connect_websocket():
        ws = MagicMock()
        ws.state = State.OPEN
        service._websocket = ws

    def fake_create_task(coro, *args, **kwargs):
        task = MagicMock()
        created_tasks.append(task)
        if hasattr(coro, "close"):
            coro.close()
        return task

    service._connect_websocket = fake_connect_websocket
    service.create_task = fake_create_task

    with patch.object(SmallestSTTService.__bases__[0], "_connect", new=AsyncMock()):
        await service._connect()

    assert len(created_tasks) == 1
    assert service._receive_task is created_tasks[0]


@pytest.mark.asyncio
async def test_connect_does_not_create_duplicate_receive_task():
    """_connect must not start a second receive task while one is already running."""
    from websockets.protocol import State

    service = _make_service()
    existing_task = MagicMock()
    service._receive_task = existing_task

    created_tasks: list = []

    async def fake_connect_websocket():
        ws = MagicMock()
        ws.state = State.OPEN
        service._websocket = ws

    def fake_create_task(coro, *args, **kwargs):
        task = MagicMock()
        created_tasks.append(task)
        if hasattr(coro, "close"):
            coro.close()
        return task

    service._connect_websocket = fake_connect_websocket
    service.create_task = fake_create_task

    with patch.object(SmallestSTTService.__bases__[0], "_connect", new=AsyncMock()):
        await service._connect()

    assert created_tasks == []
    assert service._receive_task is existing_task

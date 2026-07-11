#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for user-audio streaming in OpenAIRealtimeLLMService.

These tests drive the service's audio/turn handlers directly with a fake
``send_client_event`` and assert on the client events emitted to the service.
They cover the manual turn-detection (server-VAD-disabled) pre-roll behavior —
audio is appended to the input buffer and mirrored into a rolling pre-roll
buffer that is replayed after an interruption clears the input buffer — and the
server-VAD-enabled path, where no pre-roll is maintained.
"""

import base64
from typing import Any

import pytest

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InputAudioRawFrame, SpeechControlParamsFrame
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import (
    AUTOSIZED_USER_AUDIO_PREROLL_MARGIN_SECS,
    OpenAIRealtimeLLMService,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Records the client events sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]

    def append_payloads(self) -> list[str]:
        return [e.audio for e in self.events if isinstance(e, events.InputAudioBufferAppendEvent)]


def _make_service(*, manual_turn_detection: bool, preroll_secs: float | None = None):
    """Construct a service wired to a fake send_client_event. ``__init__`` does no I/O."""
    if manual_turn_detection:
        settings = OpenAIRealtimeLLMService.Settings(
            session_properties=SessionProperties(
                audio=AudioConfiguration(input=AudioInput(turn_detection=False))
            )
        )
    else:
        settings = None

    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=settings,
        user_audio_preroll_secs=preroll_secs,
    )

    recorder = _EventRecorder()
    service.send_client_event = recorder  # type: ignore[method-assign]

    async def _noop(*args, **kwargs):
        pass

    # _handle_interruption stops metrics, which needs a started pipeline; stub
    # it out so the handler can run in isolation.
    service.stop_all_metrics = _noop  # type: ignore[method-assign]
    return service, recorder


def _audio_frame(
    *, sample_rate: int = 16000, data: bytes = b"\x01\x02" * 160
) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=data, sample_rate=sample_rate, num_channels=1)


# ---------------------------------------------------------------------------
# Manual turn detection: maintain and replay the pre-roll
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_mode_appends_and_buffers_audio():
    """Audio is appended to the input buffer and mirrored into the pre-roll buffer."""
    service, recorder = _make_service(manual_turn_detection=True)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    assert recorder.kinds() == ["InputAudioBufferAppendEvent", "InputAudioBufferAppendEvent"]
    assert recorder.append_payloads() == [
        base64.b64encode(b"\xaa\xbb").decode(),
        base64.b64encode(b"\xcc\xdd").decode(),
    ]
    assert bytes(service._user_audio_preroll_buffer) == b"\xaa\xbb\xcc\xdd"


@pytest.mark.asyncio
async def test_manual_mode_replays_preroll_after_interruption():
    """An interruption clears the input buffer, then replays the buffered onset."""
    service, recorder = _make_service(manual_turn_detection=True)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    await service._handle_interruption()

    # Clear must come before the replay append, and the response cancel after.
    assert recorder.kinds() == [
        "InputAudioBufferAppendEvent",
        "InputAudioBufferAppendEvent",
        "InputAudioBufferClearEvent",
        "InputAudioBufferAppendEvent",
        "ResponseCancelEvent",
    ]
    # The replay re-appends the full buffered onset.
    assert recorder.append_payloads()[-1] == base64.b64encode(b"\xaa\xbb\xcc\xdd").decode()
    # The buffer is left intact (rolling window), so a later interruption can replay again.
    assert bytes(service._user_audio_preroll_buffer) == b"\xaa\xbb\xcc\xdd"


@pytest.mark.asyncio
async def test_manual_mode_interruption_with_empty_buffer_skips_replay():
    """With nothing buffered, the interruption clears and cancels but replays nothing."""
    service, recorder = _make_service(manual_turn_detection=True)

    await service._handle_interruption()

    assert recorder.kinds() == ["InputAudioBufferClearEvent", "ResponseCancelEvent"]


@pytest.mark.asyncio
async def test_manual_mode_preroll_capped_to_default_window():
    """Before VAD params are known, the pre-roll keeps DEFAULT_USER_AUDIO_PREROLL_SECS."""
    service, _ = _make_service(manual_turn_detection=True)

    # 1s at 16kHz mono / 16-bit = 32000 bytes; buffer should keep the most
    # recent 0.5s = 16000 bytes.
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=bytes(32000)))

    assert len(service._user_audio_preroll_buffer) == 16000


@pytest.mark.asyncio
async def test_manual_mode_preroll_sized_from_vad_start_secs():
    """A SpeechControlParamsFrame sizes the pre-roll to start_secs + margin."""
    service, _ = _make_service(manual_turn_detection=True)

    start_secs = 0.5
    service._handle_speech_control_params(
        SpeechControlParamsFrame(vad_params=VADParams(start_secs=start_secs))
    )
    # Send 2s of audio — comfortably more than the window — so the buffer is
    # capped to (start_secs + margin), not limited by how much we sent.
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=bytes(64000)))

    # 16kHz mono / 16-bit = 2 bytes/sample.
    expected = int(16000 * 1 * 2 * (start_secs + AUTOSIZED_USER_AUDIO_PREROLL_MARGIN_SECS))
    assert len(service._user_audio_preroll_buffer) == expected


@pytest.mark.asyncio
async def test_manual_mode_preroll_override_pins_value_and_ignores_vad_params():
    """An explicit user_audio_preroll_secs pins the pre-roll; VAD params don't resize it."""
    service, _ = _make_service(manual_turn_detection=True, preroll_secs=0.1)

    service._handle_speech_control_params(
        SpeechControlParamsFrame(vad_params=VADParams(start_secs=0.5))
    )
    # 0.1s at 16kHz mono / 16-bit = 3200 bytes.
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=bytes(32000)))

    assert len(service._user_audio_preroll_buffer) == 3200


# ---------------------------------------------------------------------------
# Server-side turn detection: no pre-roll, no clear/replay on interruption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_mode_appends_without_buffering():
    """With server-side turn detection, audio is appended but no pre-roll is kept."""
    service, recorder = _make_service(manual_turn_detection=False)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    assert recorder.kinds() == ["InputAudioBufferAppendEvent", "InputAudioBufferAppendEvent"]
    assert bytes(service._user_audio_preroll_buffer) == b""


@pytest.mark.asyncio
async def test_server_mode_interruption_does_not_clear_or_replay():
    """With server-side turn detection, an interruption doesn't touch the input buffer."""
    service, recorder = _make_service(manual_turn_detection=False)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._handle_interruption()

    assert "InputAudioBufferClearEvent" not in recorder.kinds()
    assert "ResponseCancelEvent" not in recorder.kinds()

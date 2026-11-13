#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for user-audio streaming in GeminiLiveLLMService.

These tests drive the service's audio/turn handlers directly with a fake
session and assert on the ``send_realtime_input`` calls made to the service.
They cover the server-VAD-disabled activity-window contract (audio is buffered
while the user isn't in a turn and the pre-roll is replayed after activity_start)
and the server-VAD-enabled path (audio is streamed continuously).
"""

from typing import Any

import pytest

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.google.gemini_live.llm import (
    AUTOSIZED_USER_AUDIO_PREROLL_MARGIN_SECS,
    GeminiLiveLLMService,
    GeminiVADParams,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeSession:
    """Records the keyword payloads passed to ``send_realtime_input``."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def send_realtime_input(self, **kwargs):
        self.calls.append(kwargs)

    async def send_client_content(self, **kwargs):
        self.calls.append({"client_content": kwargs})

    def audio_payloads(self) -> list[bytes]:
        return [c["audio"].data for c in self.calls if "audio" in c]

    def kinds(self) -> list[str]:
        """Ordered list of call kinds: 'audio', 'activity_start', 'activity_end'."""
        kinds = []
        for c in self.calls:
            if "audio" in c:
                kinds.append("audio")
            elif "activity_start" in c:
                kinds.append("activity_start")
            elif "activity_end" in c:
                kinds.append("activity_end")
        return kinds


def _make_service(*, vad_disabled: bool, preroll_secs: float | None = None) -> GeminiLiveLLMService:
    """Construct a service wired to a fake session. ``__init__`` does no I/O."""
    vad = GeminiVADParams(disabled=True) if vad_disabled else None
    service = GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(vad=vad),
        user_audio_preroll_secs=preroll_secs,
    )
    service._session = _FakeSession()
    service._ready_for_realtime_input = True

    async def _noop():
        pass

    # start_ttfb_metrics touches metrics machinery that needs a started
    # pipeline; stub it out so the turn handlers can run in isolation.
    service.start_ttfb_metrics = _noop  # type: ignore[method-assign]
    return service


def _audio_frame(
    *, sample_rate: int = 16000, data: bytes = b"\x01\x02" * 160
) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=data, sample_rate=sample_rate, num_channels=1)


# ---------------------------------------------------------------------------
# Manual-VAD mode: honor the activity-window contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_vad_buffers_audio_until_turn_starts():
    """While idle, audio is buffered as pre-roll and not streamed to the service."""
    service = _make_service(vad_disabled=True)
    session = service._session

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    assert session.kinds() == [], "idle audio must be buffered, not sent"
    assert bytes(service._user_audio_preroll_buffer) == b"\xaa\xbb\xcc\xdd"


@pytest.mark.asyncio
async def test_manual_vad_flushes_preroll_after_activity_start():
    """activity_start is followed by a replay of the buffered pre-roll audio."""
    service = _make_service(vad_disabled=True)
    session = service._session

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())

    # activity_start must come first, immediately followed by the pre-roll audio.
    assert session.kinds() == ["activity_start", "audio"]
    assert session.audio_payloads() == [b"\xaa\xbb\xcc\xdd"], "pre-roll must replay buffered onset"
    # Buffer is cleared so the pre-roll isn't replayed again.
    assert bytes(service._user_audio_preroll_buffer) == b""


@pytest.mark.asyncio
async def test_manual_vad_streams_audio_during_turn():
    """Once the turn has started, audio is streamed in-band."""
    service = _make_service(vad_disabled=True)
    session = service._session

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())
    # No pre-roll buffered yet, so only activity_start was sent.
    assert session.kinds() == ["activity_start"]

    await service._send_user_audio(_audio_frame(data=b"\x11\x22"))
    await service._send_user_audio(_audio_frame(data=b"\x33\x44"))

    assert session.kinds() == ["activity_start", "audio", "audio"]
    assert session.audio_payloads() == [b"\x11\x22", b"\x33\x44"]


@pytest.mark.asyncio
async def test_manual_vad_sends_activity_end_on_turn_stop():
    """Turn stop sends activity_end."""
    service = _make_service(vad_disabled=True)
    session = service._session

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())
    await service._send_user_audio(_audio_frame(data=b"\x11\x22"))
    await service._handle_user_stopped_speaking(UserStoppedSpeakingFrame())

    assert session.kinds() == ["activity_start", "audio", "activity_end"]


@pytest.mark.asyncio
async def test_manual_vad_preroll_capped_to_default_window():
    """Before VAD params are known, the pre-roll keeps DEFAULT_USER_AUDIO_PREROLL_SECS."""
    service = _make_service(vad_disabled=True)
    session = service._session

    sample_rate = 16000
    # One full second of audio at 16kHz mono / 16-bit = 32000 bytes; the buffer
    # should keep only the most recent DEFAULT_USER_AUDIO_PREROLL_SECS (0.5s) =
    # 16000 bytes.
    one_second = bytes(32000)
    await service._send_user_audio(_audio_frame(sample_rate=sample_rate, data=one_second))

    assert len(service._user_audio_preroll_buffer) == 16000

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())
    assert sum(len(p) for p in session.audio_payloads()) == 16000


@pytest.mark.asyncio
async def test_manual_vad_preroll_sized_from_vad_start_secs():
    """A SpeechControlParamsFrame sizes the pre-roll to start_secs + margin."""
    service = _make_service(vad_disabled=True)
    session = service._session

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

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())
    assert sum(len(p) for p in session.audio_payloads()) == expected


@pytest.mark.asyncio
async def test_manual_vad_preroll_override_pins_value_and_ignores_vad_params():
    """An explicit user_audio_preroll_secs pins the pre-roll; VAD params don't resize it.

    This is the escape hatch for setups using a non-default turn-start strategy,
    where the onset-to-turn-start gap isn't governed by VAD start_secs.
    """
    service = _make_service(vad_disabled=True, preroll_secs=0.1)
    session = service._session

    # The override must win over an incoming SpeechControlParamsFrame.
    service._handle_speech_control_params(
        SpeechControlParamsFrame(vad_params=VADParams(start_secs=0.5))
    )

    # 0.1s at 16kHz mono / 16-bit = int(16000 * 1 * 2 * 0.1) = 3200 bytes.
    one_second = bytes(32000)
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=one_second))

    assert len(service._user_audio_preroll_buffer) == 3200

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())
    assert sum(len(p) for p in session.audio_payloads()) == 3200


# ---------------------------------------------------------------------------
# Server-VAD mode: stream continuously (unchanged behavior)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_vad_streams_audio_when_idle():
    """With server-side VAD, audio is streamed continuously, even while idle."""
    service = _make_service(vad_disabled=False)
    session = service._session

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    assert session.kinds() == ["audio", "audio"]
    assert session.audio_payloads() == [b"\xaa\xbb", b"\xcc\xdd"]
    # No pre-roll buffer is maintained in server-VAD mode (it's never flushed).
    assert bytes(service._user_audio_preroll_buffer) == b""


@pytest.mark.asyncio
async def test_server_vad_does_not_send_activity_markers():
    """With server-side VAD, the client doesn't send activity_start/_end."""
    service = _make_service(vad_disabled=False)
    session = service._session

    await service._handle_user_started_speaking(UserStartedSpeakingFrame())
    await service._handle_user_stopped_speaking(UserStoppedSpeakingFrame())

    assert "activity_start" not in session.kinds()
    assert "activity_end" not in session.kinds()

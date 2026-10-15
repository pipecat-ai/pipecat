#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that a failing audio-context hook can't stop TTS playback.

Services override on_audio_context_completed and on_audio_context_interrupted
to message the provider, so the hooks can fail while the connection is down.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_SAMPLE_RATE = 16000


class FailingHookTTSService(TTSService):
    """HTTP-style service whose audio-context hooks raise, like a closed socket."""

    def __init__(self, delay_s: float = 0, **kwargs):
        super().__init__(push_text_frames=False, sample_rate=_SAMPLE_RATE, **kwargs)
        self._delay_s = delay_s
        self._utterances = 0

    def can_generate_metrics(self) -> bool:
        return False

    async def on_audio_context_completed(self, context_id: str):
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        raise Exception("Websocket not connected")

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self._utterances += 1
        # Each utterance's audio is tagged with its position, so tests can tell
        # which ones played.
        yield TTSAudioRawFrame(
            bytes([self._utterances, 0]) * 160, _SAMPLE_RATE, 1, context_id=context_id
        )
        if self._delay_s:
            await asyncio.sleep(self._delay_s)


def _utterances_heard(frames: list[Frame]) -> list[int]:
    return sorted({f.audio[0] for f in frames if isinstance(f, TTSAudioRawFrame)})


@pytest.mark.asyncio
async def test_failing_completed_hook_keeps_playback_running():
    service = FailingHookTTSService()

    down_frames, _ = await run_test(
        service,
        frames_to_send=[TTSSpeakFrame(text="One."), TTSSpeakFrame(text="Two.")],
    )

    assert _utterances_heard(down_frames) == [1, 2]


@pytest.mark.asyncio
async def test_failing_interrupted_hook_keeps_playback_running():
    service = FailingHookTTSService(delay_s=0.2)

    down_frames, _ = await run_test(
        service,
        frames_to_send=[
            TTSSpeakFrame(text="One."),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            TTSSpeakFrame(text="Two."),
            SleepFrame(sleep=0.3),
        ],
    )

    assert 2 in _utterances_heard(down_frames)

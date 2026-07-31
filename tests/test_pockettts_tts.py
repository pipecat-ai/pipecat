#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for PocketTTSService."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pocket_tts")

from pipecat.frames.frames import (
    AggregatedTextFrame,
    LLMAssistantPushAggregationFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

SAMPLE_RATE = 24000


def _make_mock_model():
    model = MagicMock()
    model.sample_rate = SAMPLE_RATE
    model.get_state_for_audio_prompt.return_value = {"state": "voice"}
    model.generate_audio_stream.side_effect = lambda state, text, **kwargs: iter(
        [
            torch.zeros(2400, dtype=torch.float32),
            torch.full((2400,), 0.5, dtype=torch.float32),
        ]
    )
    return model


@pytest.mark.asyncio
async def test_run_pocket_tts_success():
    """Test successful TTS generation.

    Checks frame ordering, audio conversion to int16, and that the cached
    voice state is passed with copy_state=True.
    """
    with patch("pipecat.services.pocket_tts.tts.TTSModel") as mock_model_cls:
        model = _make_mock_model()
        mock_model_cls.load_model.return_value = model

        from pipecat.services.pocket_tts.tts import PocketTTSService

        tts_service = PocketTTSService(sample_rate=SAMPLE_RATE)

        model.get_state_for_audio_prompt.assert_called_once_with("alba")

        frames_to_send = [
            TTSSpeakFrame(text="Hello world."),
        ]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
        )
        down_frames = frames_received[0]
        frame_types = [type(f) for f in down_frames]

        # Verify key frames are present
        assert AggregatedTextFrame in frame_types
        assert TTSStartedFrame in frame_types
        assert TTSStoppedFrame in frame_types
        assert TTSTextFrame in frame_types

        # Verify ordering: Started → audio/text → Stopped
        started_idx = frame_types.index(TTSStartedFrame)
        stopped_idx = frame_types.index(TTSStoppedFrame)
        text_idx = frame_types.index(TTSTextFrame)
        assert started_idx < text_idx < stopped_idx, (
            "Expected: TTSStartedFrame < TTSTextFrame < TTSStoppedFrame"
        )

        # Frames between Started and Stopped must all be audio or text. A
        # LLMAssistantPushAggregationFrame is also expected here: TTSSpeakFrame
        # defaults to append_to_context=True, so the service emits one at the end
        # of the utterance to commit the spoken text to the LLM context.
        for i in range(started_idx + 1, stopped_idx):
            assert frame_types[i] in (
                TTSAudioRawFrame,
                TTSTextFrame,
                LLMAssistantPushAggregationFrame,
            ), f"Unexpected frame type between Started and Stopped: {frame_types[i]}"

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) >= 1, "Expected at least one audio frame"
        for a_frame in audio_frames:
            assert a_frame.sample_rate == SAMPLE_RATE

        # The two mock chunks are 2400 zero samples then 2400 samples of 0.5,
        # which convert to int16 0 and 16383 respectively.
        audio = b"".join(f.audio for f in audio_frames)
        samples = torch.frombuffer(bytearray(audio), dtype=torch.int16)
        assert samples.shape[0] == 4800
        assert (samples[:2400] == 0).all()
        assert (samples[2400:] == 16383).all()

        model.generate_audio_stream.assert_called_once()
        args, kwargs = model.generate_audio_stream.call_args
        assert args[0] == {"state": "voice"}, "Expected the cached voice state"
        assert kwargs.get("copy_state") is True


@pytest.mark.asyncio
async def test_pocket_tts_voice_update():
    """Test that a runtime voice change re-derives the voice state."""
    with patch("pipecat.services.pocket_tts.tts.TTSModel") as mock_model_cls:
        model = _make_mock_model()
        mock_model_cls.load_model.return_value = model

        from pipecat.services.pocket_tts.tts import PocketTTSService

        tts_service = PocketTTSService(sample_rate=SAMPLE_RATE)

        frames_to_send = [
            TTSSpeakFrame(text="First voice."),
            SleepFrame(0.5),
            TTSUpdateSettingsFrame(delta=PocketTTSService.Settings(voice="jane")),
            TTSSpeakFrame(text="Second voice."),
            SleepFrame(0.5),
        ]

        await run_test(
            tts_service,
            frames_to_send=frames_to_send,
        )

        voices = [call.args[0] for call in model.get_state_for_audio_prompt.call_args_list]
        assert voices == ["alba", "jane"]
        assert model.generate_audio_stream.call_count == 2


if __name__ == "__main__":
    unittest.main()

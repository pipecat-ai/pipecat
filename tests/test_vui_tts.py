#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for VuiTTSService."""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vui")

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


def _make_mock_engine():
    """An Engine whose single row streams two known audio frames per turn."""
    engine = MagicMock()
    engine._loaded_ckpt = "vui-nano-1.1.safetensors"
    engine.Q = 16
    row = MagicMock()
    row.stream.side_effect = lambda text, cfg, cancel=None: iter(
        [
            torch.zeros(1, 1, 2400, dtype=torch.float32),
            torch.full((1, 1, 2400), 0.5, dtype=torch.float32),
        ]
    )
    engine.new_row.return_value = row
    return engine, row


def _patch_voice(service_cls_module, transcript="Hello."):
    """Bypass the Hub download: every voice resolves to a fixed segment."""
    from vui.engine import Segment

    return patch.object(
        service_cls_module.VuiTTSService,
        "_resolve_voice",
        lambda self, voice: Segment(transcript, torch.zeros(4, 16, dtype=torch.long)),
    )


@pytest.mark.asyncio
async def test_run_vui_tts_success():
    """Frame ordering, int16 conversion, one prefill per voice, rewind per turn."""
    import pipecat.services.vui.tts as vui_tts

    with patch.object(vui_tts, "Engine") as mock_engine_cls, _patch_voice(vui_tts):
        engine, row = _make_mock_engine()
        mock_engine_cls.return_value = engine

        tts_service = vui_tts.VuiTTSService(sample_rate=SAMPLE_RATE)
        mock_engine_cls.assert_called_once_with("vui-nano-1.1", max_rows=1)

        frames_received = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello world.")],
        )
        down_frames = frames_received[0]
        frame_types = [type(f) for f in down_frames]

        assert AggregatedTextFrame in frame_types
        assert TTSStartedFrame in frame_types
        assert TTSStoppedFrame in frame_types
        assert TTSTextFrame in frame_types

        started_idx = frame_types.index(TTSStartedFrame)
        stopped_idx = frame_types.index(TTSStoppedFrame)
        text_idx = frame_types.index(TTSTextFrame)
        assert started_idx < text_idx < stopped_idx
        for i in range(started_idx + 1, stopped_idx):
            assert frame_types[i] in (
                TTSAudioRawFrame,
                TTSTextFrame,
                LLMAssistantPushAggregationFrame,
            ), f"Unexpected frame type between Started and Stopped: {frame_types[i]}"

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) >= 1
        for a_frame in audio_frames:
            assert a_frame.sample_rate == SAMPLE_RATE

        # 2400 zero samples then 2400 samples of 0.5 -> int16 0 and 16383.
        audio = b"".join(f.audio for f in audio_frames)
        samples = torch.frombuffer(bytearray(audio), dtype=torch.int16)
        assert samples.shape[0] == 4800
        assert (samples[:2400] == 0).all()
        assert (samples[2400:] == 16383).all()

        # The voice is prefilled once; the row is rewound to it after the turn.
        row.prefill.assert_called_once()
        row.stream.assert_called_once()
        assert row.stream.call_args.args[0] == "Hello world."
        row.rewind.assert_called_once()


@pytest.mark.asyncio
async def test_vui_tts_voice_update():
    """A runtime voice change re-prefills the row on the next utterance."""
    import pipecat.services.vui.tts as vui_tts

    with patch.object(vui_tts, "Engine") as mock_engine_cls, _patch_voice(vui_tts):
        engine, row = _make_mock_engine()
        mock_engine_cls.return_value = engine

        tts_service = vui_tts.VuiTTSService(sample_rate=SAMPLE_RATE)
        frames_to_send = [
            TTSSpeakFrame(text="First voice."),
            SleepFrame(0.5),
            TTSUpdateSettingsFrame(delta=vui_tts.VuiTTSService.Settings(voice="abraham")),
            TTSSpeakFrame(text="Second voice."),
            SleepFrame(0.5),
        ]
        await run_test(tts_service, frames_to_send=frames_to_send)

        assert row.prefill.call_count == 2, "expected a re-prefill after the voice change"
        assert row.stream.call_count == 2
        assert tts_service._voice_loaded == "abraham"

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock package version check before importing pipecat (development mode)
_version_patcher = patch("importlib.metadata.version", return_value="0.0.0-dev")
_version_patcher.start()

# Mock krisp_audio before any pipecat import that loads krisp_instance / VIVA IP strategy
mock_krisp_audio = MagicMock()
mock_krisp_audio.SamplingRate.Sr8000Hz = 8000
mock_krisp_audio.SamplingRate.Sr16000Hz = 16000
mock_krisp_audio.SamplingRate.Sr24000Hz = 24000
mock_krisp_audio.SamplingRate.Sr32000Hz = 32000
mock_krisp_audio.SamplingRate.Sr44100Hz = 44100
mock_krisp_audio.SamplingRate.Sr48000Hz = 48000
mock_krisp_audio.FrameDuration.Fd10ms = "10ms"
mock_krisp_audio.FrameDuration.Fd15ms = "15ms"
mock_krisp_audio.FrameDuration.Fd20ms = "20ms"
mock_krisp_audio.FrameDuration.Fd30ms = "30ms"
mock_krisp_audio.FrameDuration.Fd32ms = "32ms"

sys.modules["krisp_audio"] = mock_krisp_audio

mock_pipecat_krisp = MagicMock()
sys.modules["pipecat_ai_krisp"] = mock_pipecat_krisp
sys.modules["pipecat_ai_krisp.audio"] = MagicMock()
sys.modules["pipecat_ai_krisp.audio.krisp_processor"] = MagicMock()

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.krisp_viva_ip_user_turn_start_strategy import (
    KrispVivaIPUserTurnStartStrategy,
)

STRATEGY_MODULE = "pipecat.turns.user_start.krisp_viva_ip_user_turn_start_strategy"


def _int16_silence(num_samples: int) -> bytes:
    return np.zeros(num_samples, dtype=np.int16).tobytes()


class TestKrispVivaIPUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    """Tests for KrispVivaIPUserTurnStartStrategy with mocked krisp_audio."""

    def setUp(self):
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".kef", delete=False)
        self.temp_model_file.write(b"dummy")
        self.temp_model_file.close()
        self.model_path = self.temp_model_file.name

        self.mock_krisp_audio = mock_krisp_audio
        self.mock_krisp_audio.reset_mock()
        self.mock_krisp_audio.ModelInfo.reset_mock()
        self.mock_krisp_audio.IpSessionConfig.reset_mock()
        self.mock_krisp_audio.IpFloat.reset_mock()

        self.mock_model_info = MagicMock()
        self.mock_krisp_audio.ModelInfo.return_value = self.mock_model_info

        self.mock_ip_cfg = MagicMock()
        self.mock_krisp_audio.IpSessionConfig.return_value = self.mock_ip_cfg

        self.mock_ip_session = MagicMock()
        self.mock_krisp_audio.IpFloat.create.return_value = self.mock_ip_session

        self.krisp_patch = patch(f"{STRATEGY_MODULE}.krisp_audio", self.mock_krisp_audio)
        self.krisp_patch.start()

        self.sdk_patcher = patch(f"{STRATEGY_MODULE}.KrispVivaSDKManager")
        self.mock_sdk_manager = self.sdk_patcher.start()
        self.mock_sdk_manager.acquire = MagicMock()
        self.mock_sdk_manager.release = MagicMock()

    def tearDown(self):
        self.krisp_patch.stop()
        self.sdk_patcher.stop()
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def _make_strategy(self, *, threshold: float = 0.5, frame_duration_ms: int = 20):
        return KrispVivaIPUserTurnStartStrategy(
            model_path=self.model_path,
            threshold=threshold,
            frame_duration_ms=frame_duration_ms,
            api_key="test-key",
        )

    def _audio_frame(self, sample_rate: int = 16000, frame_duration_ms: int = 20):
        samples = int(sample_rate * frame_duration_ms / 1000)
        return InputAudioRawFrame(
            audio=_int16_silence(samples),
            sample_rate=sample_rate,
            num_channels=1,
        )

    async def test_interruption_detected_emits_turn_and_stop(self):
        self.mock_ip_session.process = MagicMock(return_value=0.87)

        strategy = self._make_strategy(threshold=0.5)
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(VADUserStartedSpeakingFrame())
            result = await strategy.process_frame(self._audio_frame())

            self.assertTrue(fired)
            self.assertEqual(result, ProcessFrameResult.STOP)
            self.mock_ip_session.process.assert_called()
        finally:
            await strategy.cleanup()

    async def test_backchannel_suppressed_no_event_continue(self):
        self.mock_ip_session.process = MagicMock(return_value=0.23)

        strategy = self._make_strategy(threshold=0.5)
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(VADUserStartedSpeakingFrame())
            result = await strategy.process_frame(self._audio_frame())

            self.assertFalse(fired)
            self.assertEqual(result, ProcessFrameResult.CONTINUE)
        finally:
            await strategy.cleanup()

    async def test_reset_on_vad_stopped_clears_state(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.mock_ip_session.process.reset_mock()

            await strategy.process_frame(VADUserStoppedSpeakingFrame())
            result = await strategy.process_frame(self._audio_frame())

            self.assertEqual(result, ProcessFrameResult.CONTINUE)
            # process() is still called (continuous state), but with vad_flag=False
            self.mock_ip_session.process.assert_called()
            args = self.mock_ip_session.process.call_args[0]
            self.assertFalse(args[1])  # vad_flag should be False
        finally:
            await strategy.cleanup()

    async def test_reset_on_bot_stopped_clears_state(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.mock_ip_session.process.reset_mock()

            await strategy.process_frame(BotStoppedSpeakingFrame())
            result = await strategy.process_frame(self._audio_frame())

            self.assertEqual(result, ProcessFrameResult.CONTINUE)
            # process() is still called (continuous state), but with vad_flag=False
            self.mock_ip_session.process.assert_called()
            args = self.mock_ip_session.process.call_args[0]
            self.assertFalse(args[1])  # vad_flag should be False
        finally:
            await strategy.cleanup()

    async def test_no_op_before_vad_start(self):
        self.mock_ip_session.process = MagicMock(return_value=0.99)

        strategy = self._make_strategy()
        try:
            result = await strategy.process_frame(self._audio_frame())
            self.assertEqual(result, ProcessFrameResult.CONTINUE)
            # process() is called (continuous state) even before VAD start,
            # but _speech_active=False prevents triggering despite high prob
            self.mock_ip_session.process.assert_called()
            args = self.mock_ip_session.process.call_args[0]
            self.assertFalse(args[1])  # vad_flag should be False
        finally:
            await strategy.cleanup()

    async def test_decision_sticks_no_double_trigger(self):
        self.mock_ip_session.process = MagicMock(return_value=0.9)

        strategy = self._make_strategy(threshold=0.5)
        try:
            count = 0

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal count
                count += 1

            await strategy.process_frame(VADUserStartedSpeakingFrame())
            r1 = await strategy.process_frame(self._audio_frame())
            r2 = await strategy.process_frame(self._audio_frame())

            self.assertEqual(r1, ProcessFrameResult.STOP)
            self.assertEqual(r2, ProcessFrameResult.CONTINUE)
            self.assertEqual(count, 1)
        finally:
            await strategy.cleanup()

    async def test_unrelated_frames_continue(self):
        strategy = self._make_strategy()
        try:
            r1 = await strategy.process_frame(BotStartedSpeakingFrame())
            r2 = await strategy.process_frame(
                TranscriptionFrame(text="hi", user_id="", timestamp="")
            )
            self.assertEqual(r1, ProcessFrameResult.CONTINUE)
            self.assertEqual(r2, ProcessFrameResult.CONTINUE)
        finally:
            await strategy.cleanup()


if __name__ == "__main__":
    unittest.main()

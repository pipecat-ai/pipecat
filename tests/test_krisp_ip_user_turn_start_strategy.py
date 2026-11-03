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

# The version patch is scoped to just the import so it doesn't leak across the
# test session and corrupt importlib.metadata.version for other tests
# (e.g. transformers' import-time dependency checks).
with patch("importlib.metadata.version", return_value="0.0.0-dev"):
    from pipecat.audio import krisp_instance
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        InputAudioRawFrame,
        TranscriptionFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.turns.user_start import BaseUserTurnStartStrategy
    from pipecat.turns.user_start.krisp_viva_ip_user_turn_start_strategy import (
        KrispVivaIPUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
    from pipecat.turns.user_turn_controller import UserTurnController
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.utils.asyncio.task_manager import TaskManager
    from tests.frame_processor_helpers import frame_processor_setup

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

        self.ip_accepts_bot_speaking_patcher = patch(
            f"{STRATEGY_MODULE}.krisp_sdk_ip_accepts_bot_speaking"
        )
        self.mock_ip_accepts_bot_speaking = self.ip_accepts_bot_speaking_patcher.start()
        self.mock_ip_accepts_bot_speaking.return_value = True

        self.sdk_patcher = patch(f"{STRATEGY_MODULE}.KrispVivaSDKManager")
        self.mock_sdk_manager = self.sdk_patcher.start()
        self.mock_sdk_manager.acquire = MagicMock()
        self.mock_sdk_manager.release = MagicMock()

    def tearDown(self):
        self.krisp_patch.stop()
        self.ip_accepts_bot_speaking_patcher.stop()
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

    def _audio_frame(
        self, sample_rate: int = 16000, frame_duration_ms: int = 20, num_samples: int | None = None
    ):
        samples = (
            num_samples if num_samples is not None else int(sample_rate * frame_duration_ms / 1000)
        )
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

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())

            self.assertTrue(fired)
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

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())

            self.assertFalse(fired)
        finally:
            await strategy.cleanup()

    async def test_reset_on_vad_stopped_clears_state(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.mock_ip_session.process.reset_mock()

            await strategy.process_frame(VADUserStoppedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())

            # process() is still called (continuous state), but with speech_active=False
            self.mock_ip_session.process.assert_called()
            args = self.mock_ip_session.process.call_args[0]
            self.assertFalse(args[1])  # speech_active should be False
        finally:
            await strategy.cleanup()

    async def test_reset_on_bot_stopped_clears_state(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.mock_ip_session.process.reset_mock()

            await strategy.process_frame(BotStoppedSpeakingFrame())
            # The user turn controller resets the strategy once the turn starts.
            await strategy.handle_user_turn_started()
            await strategy.process_frame(self._audio_frame())

            # process() is still called (continuous state), but with speech_active=False
            self.mock_ip_session.process.assert_called()
            args = self.mock_ip_session.process.call_args[0]
            self.assertFalse(args[1])  # speech_active should be False
        finally:
            await strategy.cleanup()

    async def test_no_op_before_vad_start(self):
        self.mock_ip_session.process = MagicMock(return_value=0.99)

        strategy = self._make_strategy()
        try:
            await strategy.process_frame(self._audio_frame())
            # process() is called (continuous state) even before VAD start,
            # but _speech_active=False prevents triggering despite high prob
            self.mock_ip_session.process.assert_called()
            args = self.mock_ip_session.process.call_args[0]
            self.assertFalse(args[1])  # speech_active should be False
        finally:
            await strategy.cleanup()

    async def test_bot_speaking_flag_is_forwarded_to_model(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy()
        try:
            await strategy.process_frame(self._audio_frame())
            self.assertFalse(self.mock_ip_session.process.call_args[0][2])

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.assertTrue(self.mock_ip_session.process.call_args[0][2])

            await strategy.process_frame(BotStoppedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.assertFalse(self.mock_ip_session.process.call_args[0][2])
        finally:
            await strategy.cleanup()

    async def test_vad_start_while_bot_silent_triggers_turn_immediately(self):
        strategy = self._make_strategy()
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(VADUserStartedSpeakingFrame())

            self.assertTrue(fired)
            self.mock_ip_session.process.assert_not_called()
        finally:
            await strategy.cleanup()

    async def test_silent_bot_turn_does_not_trigger_twice(self):
        self.mock_ip_session.process = MagicMock(return_value=0.99)

        strategy = self._make_strategy()
        try:
            count = 0

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal count
                count += 1

            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())

            self.assertEqual(count, 1)
        finally:
            await strategy.cleanup()

    async def test_old_ip_api_calls_process_without_bot_speaking(self):
        self.mock_ip_accepts_bot_speaking.return_value = False
        self.mock_ip_session.process = MagicMock(return_value=0.0)

        strategy = self._make_strategy()
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())

            args = self.mock_ip_session.process.call_args.args
            self.assertEqual(len(args), 2)
            self.assertTrue(args[1])
        finally:
            await strategy.cleanup()

    async def test_old_ip_api_silent_bot_still_classifies_with_model(self):
        self.mock_ip_accepts_bot_speaking.return_value = False
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(VADUserStartedSpeakingFrame())
            self.assertFalse(fired)
            await strategy.process_frame(self._audio_frame())
            self.assertFalse(fired)

            self.mock_ip_session.process = MagicMock(return_value=0.9)
            await strategy.process_frame(self._audio_frame())
            self.assertTrue(fired)
        finally:
            await strategy.cleanup()

    async def test_old_ip_api_bot_stop_only_resets(self):
        self.mock_ip_accepts_bot_speaking.return_value = False
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy()
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(BotStoppedSpeakingFrame())
            self.assertFalse(fired)
        finally:
            await strategy.cleanup()

    async def test_new_ip_api_version_gate(self):
        for version, expected in [((1, 13, 3), False), ((1, 14, 0), True), ((2, 0, 0), True)]:
            with self.subTest(version=version):
                sdk_version = MagicMock(major=version[0], minor=version[1], patch=version[2])
                with patch.object(
                    krisp_instance.krisp_audio, "getVersion", return_value=sdk_version
                ):
                    self.assertEqual(krisp_instance.krisp_sdk_ip_accepts_bot_speaking(), expected)

    async def test_bot_stops_while_user_still_speaking_triggers_turn(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy()
        try:
            count = 0

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal count
                count += 1

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            self.assertEqual(count, 0)

            await strategy.process_frame(BotStoppedSpeakingFrame())

            self.assertEqual(count, 1)
        finally:
            await strategy.cleanup()

    async def test_bot_stopped_reaches_strategies_after_it(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)
        seen = []

        class RecordingStrategy(BaseUserTurnStartStrategy):
            async def process_frame(self, frame):
                seen.append(frame)

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[self._make_strategy(), RecordingStrategy()],
                stop=[ExternalUserTurnStopStrategy()],
            )
        )
        await controller.setup(frame_processor_setup(TaskManager()))
        await controller.start()
        try:
            turns = 0

            @controller.event_handler("on_user_turn_started")
            async def on_user_turn_started(controller, strategy, params):
                nonlocal turns
                turns += 1

            await controller.process_frame(BotStartedSpeakingFrame())
            await controller.process_frame(VADUserStartedSpeakingFrame())
            await controller.process_frame(self._audio_frame())
            bot_stopped = BotStoppedSpeakingFrame()
            await controller.process_frame(bot_stopped)

            self.assertEqual(turns, 1)
            self.assertIn(bot_stopped, seen)
        finally:
            await controller.cleanup()

    async def test_bot_stops_after_interruption_does_not_trigger_twice(self):
        self.mock_ip_session.process = MagicMock(return_value=0.9)

        strategy = self._make_strategy()
        try:
            count = 0

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal count
                count += 1

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            await strategy.process_frame(BotStoppedSpeakingFrame())

            self.assertEqual(count, 1)
        finally:
            await strategy.cleanup()

    async def test_bot_stops_without_user_speech_does_not_trigger(self):
        strategy = self._make_strategy()
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(BotStoppedSpeakingFrame())
            self.assertFalse(fired)
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

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())
            await strategy.process_frame(self._audio_frame())

            self.assertEqual(count, 1)
        finally:
            await strategy.cleanup()

    async def test_reset_clears_audio_buffer(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            # Feed a partial frame (smaller than samples_per_frame) so it stays in buffer
            await strategy.process_frame(self._audio_frame(num_samples=10))
            self.assertGreater(len(strategy._audio_buffer), 0)

            await strategy.process_frame(VADUserStoppedSpeakingFrame())
            self.assertEqual(len(strategy._audio_buffer), 0)
        finally:
            await strategy.cleanup()

    async def test_unrelated_frames_continue(self):
        strategy = self._make_strategy()
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(TranscriptionFrame(text="hi", user_id="", timestamp=""))
        finally:
            await strategy.cleanup()

    async def test_handle_user_turn_started_clears_state(self):
        self.mock_ip_session.process = MagicMock(return_value=0.1)

        strategy = self._make_strategy(threshold=0.5)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame(num_samples=10))
            self.assertTrue(strategy._speech_active)
            self.assertGreater(len(strategy._audio_buffer), 0)

            await strategy.handle_user_turn_started()

            self.assertFalse(strategy._speech_active)
            self.assertEqual(len(strategy._audio_buffer), 0)
            self.assertFalse(strategy._decision_made)
        finally:
            await strategy.cleanup()

    async def test_cleanup_releases_sdk(self):
        strategy = self._make_strategy()

        await strategy.cleanup()

        self.mock_sdk_manager.release.assert_called_once()
        self.assertIsNone(strategy._ip_session)
        self.assertFalse(strategy._sdk_acquired)

    def test_init_raises_if_no_model_path(self):
        # Cleared, since the constructor falls back to KRISP_VIVA_IP_MODEL_PATH
        # and an ambient value would satisfy it.
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                KrispVivaIPUserTurnStartStrategy(api_key="test-key")

    def test_init_raises_if_wrong_extension(self):
        with self.assertRaises(ValueError):
            KrispVivaIPUserTurnStartStrategy(model_path="/some/model.bin", api_key="test-key")

    def test_init_raises_if_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            KrispVivaIPUserTurnStartStrategy(
                model_path="/nonexistent/model.kef", api_key="test-key"
            )

    def test_init_raises_if_sdk_fails(self):
        self.mock_sdk_manager.acquire.side_effect = RuntimeError("SDK error")
        with self.assertRaises(RuntimeError):
            KrispVivaIPUserTurnStartStrategy(model_path=self.model_path, api_key="test-key")

    def test_init_uses_env_var_for_model_path(self):
        with patch.dict(os.environ, {"KRISP_VIVA_IP_MODEL_PATH": self.model_path}):
            strategy = KrispVivaIPUserTurnStartStrategy(api_key="test-key")
            self.assertEqual(strategy._model_path, self.model_path)

    async def test_vad_stopped_when_speech_inactive_is_no_op(self):
        strategy = self._make_strategy()
        try:
            await strategy.process_frame(VADUserStoppedSpeakingFrame())
            self.assertFalse(strategy._speech_active)
        finally:
            await strategy.cleanup()

    async def test_interruption_at_exact_threshold_triggers(self):
        threshold = 0.5
        self.mock_ip_session.process = MagicMock(return_value=threshold)

        strategy = self._make_strategy(threshold=threshold)
        try:
            fired = False

            @strategy.event_handler("on_user_turn_started")
            async def on_user_turn_started(strategy, params):
                nonlocal fired
                fired = True

            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            await strategy.process_frame(self._audio_frame())

            self.assertTrue(fired)
        finally:
            await strategy.cleanup()


if __name__ == "__main__":
    unittest.main()

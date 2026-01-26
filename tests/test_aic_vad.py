#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

# Check if aic_sdk is available
try:
    import aic_sdk

    HAS_AIC_SDK = True
except ImportError:
    HAS_AIC_SDK = False


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICVADAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test suite for AICVADAnalyzer using real aic_sdk."""

    @classmethod
    def setUpClass(cls):
        """Import AICVADAnalyzer after confirming aic_sdk is available."""
        from pipecat.audio.vad.aic_vad import AICVADAnalyzer

        cls.AICVADAnalyzer = AICVADAnalyzer

    def test_initialization_without_factory(self):
        """Test analyzer initialization without a factory."""
        analyzer = self.AICVADAnalyzer()

        self.assertIsNone(analyzer._vad_context_factory)
        self.assertIsNone(analyzer._vad_ctx)
        # Fixed params should be set
        self.assertEqual(analyzer._params.confidence, 0.5)
        self.assertEqual(analyzer._params.start_secs, 0.0)
        self.assertEqual(analyzer._params.stop_secs, 0.0)
        self.assertEqual(analyzer._params.min_volume, 0.0)

    def test_initialization_with_factory(self):
        """Test analyzer initialization with a factory."""
        # Create a mock VAD context for testing
        mock_vad_ctx = MockVadContext()
        factory = lambda: mock_vad_ctx
        analyzer = self.AICVADAnalyzer(vad_context_factory=factory)

        self.assertIsNotNone(analyzer._vad_context_factory)

    def test_initialization_with_vad_params(self):
        """Test analyzer initialization with VAD parameters."""
        analyzer = self.AICVADAnalyzer(
            speech_hold_duration=0.1,
            minimum_speech_duration=0.05,
            sensitivity=8.0,
        )

        self.assertEqual(analyzer._pending_speech_hold_duration, 0.1)
        self.assertEqual(analyzer._pending_minimum_speech_duration, 0.05)
        self.assertEqual(analyzer._pending_sensitivity, 8.0)

    def test_bind_vad_context_factory(self):
        """Test binding a factory post-construction."""
        mock_vad_ctx = MockVadContext()
        analyzer = self.AICVADAnalyzer()
        factory = lambda: mock_vad_ctx

        analyzer.bind_vad_context_factory(factory)

        self.assertEqual(analyzer._vad_context_factory, factory)
        # Should have attempted to initialize
        self.assertEqual(analyzer._vad_ctx, mock_vad_ctx)

    def test_bind_vad_context_factory_applies_params(self):
        """Test that binding factory applies pending VAD params."""
        mock_vad_ctx = MockVadContext()
        analyzer = self.AICVADAnalyzer(
            speech_hold_duration=0.1,
            minimum_speech_duration=0.05,
            sensitivity=8.0,
        )
        factory = lambda: mock_vad_ctx

        analyzer.bind_vad_context_factory(factory)

        # Verify parameters were applied
        self.assertIn(
            (aic_sdk.VadParameter.SpeechHoldDuration, 0.1),
            mock_vad_ctx.parameters_set,
        )
        self.assertIn(
            (aic_sdk.VadParameter.MinimumSpeechDuration, 0.05),
            mock_vad_ctx.parameters_set,
        )
        self.assertIn(
            (aic_sdk.VadParameter.Sensitivity, 8.0),
            mock_vad_ctx.parameters_set,
        )

    def test_set_sample_rate(self):
        """Test setting sample rate."""
        analyzer = self.AICVADAnalyzer()

        analyzer.set_sample_rate(16000)

        self.assertEqual(analyzer._sample_rate, 16000)

    def test_set_sample_rate_with_init_sample_rate(self):
        """Test that init_sample_rate takes precedence."""
        # Create analyzer and manually set _init_sample_rate
        analyzer = self.AICVADAnalyzer()
        analyzer._init_sample_rate = 48000

        analyzer.set_sample_rate(16000)

        # init_sample_rate should take precedence
        self.assertEqual(analyzer._sample_rate, 48000)

    def test_set_sample_rate_triggers_context_init(self):
        """Test that set_sample_rate attempts context initialization."""
        mock_vad_ctx = MockVadContext()
        factory = lambda: mock_vad_ctx
        analyzer = self.AICVADAnalyzer(vad_context_factory=factory)

        analyzer.set_sample_rate(16000)

        self.assertEqual(analyzer._vad_ctx, mock_vad_ctx)

    def test_num_frames_required_with_sample_rate(self):
        """Test num_frames_required returns correct value."""
        analyzer = self.AICVADAnalyzer()
        analyzer.set_sample_rate(16000)

        frames = analyzer.num_frames_required()

        # 10ms at 16kHz = 160 frames
        self.assertEqual(frames, 160)

    def test_num_frames_required_different_sample_rates(self):
        """Test num_frames_required for different sample rates."""
        analyzer = self.AICVADAnalyzer()

        test_cases = [
            (8000, 80),  # 10ms at 8kHz
            (16000, 160),  # 10ms at 16kHz
            (24000, 240),  # 10ms at 24kHz
            (48000, 480),  # 10ms at 48kHz
        ]

        for sample_rate, expected_frames in test_cases:
            analyzer.set_sample_rate(sample_rate)
            frames = analyzer.num_frames_required()
            self.assertEqual(frames, expected_frames, f"Failed for {sample_rate}Hz")

    def test_num_frames_required_no_sample_rate(self):
        """Test num_frames_required returns default when no sample rate."""
        analyzer = self.AICVADAnalyzer()

        frames = analyzer.num_frames_required()

        # Default is 160
        self.assertEqual(frames, 160)

    def test_voice_confidence_no_context(self):
        """Test voice_confidence returns 0.0 when no context."""
        analyzer = self.AICVADAnalyzer()

        confidence = analyzer.voice_confidence(b"\x00" * 320)

        self.assertEqual(confidence, 0.0)

    def test_voice_confidence_speech_detected(self):
        """Test voice_confidence returns 1.0 when speech detected."""
        mock_vad_ctx = MockVadContext(speech_detected=True)
        factory = lambda: mock_vad_ctx
        analyzer = self.AICVADAnalyzer(vad_context_factory=factory)
        analyzer.set_sample_rate(16000)

        confidence = analyzer.voice_confidence(b"\x00" * 320)

        self.assertEqual(confidence, 1.0)

    def test_voice_confidence_no_speech(self):
        """Test voice_confidence returns 0.0 when no speech."""
        mock_vad_ctx = MockVadContext(speech_detected=False)
        factory = lambda: mock_vad_ctx
        analyzer = self.AICVADAnalyzer(vad_context_factory=factory)
        analyzer.set_sample_rate(16000)

        confidence = analyzer.voice_confidence(b"\x00" * 320)

        self.assertEqual(confidence, 0.0)

    def test_voice_confidence_handles_exception(self):
        """Test voice_confidence handles exceptions gracefully."""
        mock_vad_ctx = MockVadContext(raise_on_detect=True)
        factory = lambda: mock_vad_ctx
        analyzer = self.AICVADAnalyzer(vad_context_factory=factory)
        analyzer.set_sample_rate(16000)

        confidence = analyzer.voice_confidence(b"\x00" * 320)

        self.assertEqual(confidence, 0.0)

    def test_lazy_initialization(self):
        """Test that VAD context is lazily initialized."""
        call_count = 0
        mock_vad_ctx = MockVadContext()

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return mock_vad_ctx

        analyzer = self.AICVADAnalyzer(vad_context_factory=counting_factory)

        # Factory not called yet
        self.assertEqual(call_count, 0)

        # First call to voice_confidence triggers initialization
        analyzer.voice_confidence(b"\x00" * 320)
        self.assertEqual(call_count, 1)

        # Subsequent calls don't re-initialize
        analyzer.voice_confidence(b"\x00" * 320)
        analyzer.voice_confidence(b"\x00" * 320)
        self.assertEqual(call_count, 1)

    def test_deferred_initialization_on_factory_failure(self):
        """Test that initialization is deferred when factory fails."""
        call_count = 0
        mock_vad_ctx = MockVadContext(speech_detected=True)

        def failing_then_succeeding_factory():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Not ready yet")
            return mock_vad_ctx

        analyzer = self.AICVADAnalyzer(vad_context_factory=failing_then_succeeding_factory)

        # First two calls fail, should return 0.0
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

        # Third call succeeds
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 1.0)

    def test_apply_vad_params_deferred_on_failure(self):
        """Test that VAD param application handles exceptions."""
        mock_vad_ctx = MockVadContext(raise_on_set_param=True)
        factory = lambda: mock_vad_ctx

        analyzer = self.AICVADAnalyzer(
            vad_context_factory=factory,
            speech_hold_duration=0.1,
        )

        # Should not raise, just log debug message
        analyzer.bind_vad_context_factory(factory)

        # Context should still be set despite param failure
        self.assertEqual(analyzer._vad_ctx, mock_vad_ctx)

    def test_apply_vad_params_only_set_values(self):
        """Test that only specified VAD params are applied."""
        mock_vad_ctx = MockVadContext()
        factory = lambda: mock_vad_ctx
        analyzer = self.AICVADAnalyzer(
            vad_context_factory=factory,
            speech_hold_duration=0.1,
            # minimum_speech_duration and sensitivity not set
        )

        analyzer.bind_vad_context_factory(factory)

        # Only SpeechHoldDuration should be set
        self.assertEqual(len(mock_vad_ctx.parameters_set), 1)
        self.assertIn(
            (aic_sdk.VadParameter.SpeechHoldDuration, 0.1),
            mock_vad_ctx.parameters_set,
        )

    def test_fixed_vad_params(self):
        """Test that VAD uses fixed parameters."""
        analyzer = self.AICVADAnalyzer()

        # These are the fixed params for AIC VAD
        self.assertEqual(analyzer._params.confidence, 0.5)
        self.assertEqual(analyzer._params.start_secs, 0.0)
        self.assertEqual(analyzer._params.stop_secs, 0.0)
        self.assertEqual(analyzer._params.min_volume, 0.0)


class MockVadContext:
    """A lightweight mock for AIC VadContext that mimics real behavior."""

    def __init__(
        self,
        speech_detected: bool = False,
        raise_on_detect: bool = False,
        raise_on_set_param: bool = False,
    ):
        self.speech_detected = speech_detected
        self.raise_on_detect = raise_on_detect
        self.raise_on_set_param = raise_on_set_param
        self.parameters_set: list[tuple] = []

    def is_speech_detected(self) -> bool:
        if self.raise_on_detect:
            raise RuntimeError("VAD error")
        return self.speech_detected

    def set_parameter(self, param, value):
        if self.raise_on_set_param:
            raise RuntimeError("Param error")
        self.parameters_set.append((param, value))


if __name__ == "__main__":
    unittest.main()

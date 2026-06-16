#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Check if aic_sdk is available
aic_sdk: Any
try:
    import aic_sdk

    HAS_AIC_SDK = True
except ImportError:
    aic_sdk = None
    HAS_AIC_SDK = False

from tests.aic_mocks import MockModel, MockProcessorSync  # noqa: E402

# Module path for patching
AIC_QUAIL_VAD_MODULE = "pipecat.audio.vad.aic_quail_vad"


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICQuailVADAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test suite for AICQuailVADAnalyzer using mocked aic_sdk types."""

    @classmethod
    def setUpClass(cls):
        from pipecat.audio.vad.aic_quail_vad import (
            DEFAULT_QUAIL_VAD_MODEL_ID,
            AICQuailVADAnalyzer,
        )

        cls.AICQuailVADAnalyzer = AICQuailVADAnalyzer
        cls.DEFAULT_QUAIL_VAD_MODEL_ID = DEFAULT_QUAIL_VAD_MODEL_ID

    def setUp(self):
        self.mock_model = MockModel(model_id="quail-vad-2.0-xxs-16khz", optimal_num_frames=160)
        self.mock_processor = MockProcessorSync()

    def _create_analyzer(self, **kwargs):
        """Construct the analyzer with all SDK touchpoints mocked.

        Returns the constructed analyzer plus the patched mock-class objects so
        tests can assert on download/from_file/Processor call shapes that
        happened during ``__init__``.
        """
        analyzer_kwargs = {"license_key": "test-key"}
        analyzer_kwargs.update(kwargs)
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id") as mock_sdk_id,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Model") as mock_model_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=self.mock_processor
            ) as mock_processor_cls,
        ):
            mock_model_cls.from_file.return_value = self.mock_model
            mock_model_cls.download.return_value = "/tmp/test.aicmodel"
            mock_config_cls.return_value = MagicMock()
            analyzer = self.AICQuailVADAnalyzer(**analyzer_kwargs)
        return analyzer, {
            "Model": mock_model_cls,
            "Processor": mock_processor_cls,
            "ProcessorConfig": mock_config_cls,
            "set_sdk_id": mock_sdk_id,
        }

    def _initialize_at(self, analyzer, sample_rate: int = 16000):
        """Drive ``set_sample_rate`` with Processor + ProcessorConfig patched.

        Model patching is unnecessary because the analyzer already holds a
        reference to the mocked model from ``__init__``.
        """
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=self.mock_processor
            ) as mock_processor_cls,
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(sample_rate)
            return mock_processor_cls

    # --- Construction --------------------------------------------------------

    def test_initialization_requires_model_id_or_path(self):
        """Construction fails when both model_id and model_path are None."""
        with patch(f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id"):
            with self.assertRaises(ValueError) as ctx:
                self.AICQuailVADAnalyzer(license_key="test-key", model_id=None, model_path=None)
        self.assertIn("model_id", str(ctx.exception))
        self.assertIn("model_path", str(ctx.exception))

    def test_validation_runs_before_set_sdk_id(self):
        """Invalid kwargs raise before mutating global telemetry state."""
        with patch(f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id") as mock_sdk_id:
            with self.assertRaises(ValueError):
                self.AICQuailVADAnalyzer(license_key="test-key", model_id=None, model_path=None)
        mock_sdk_id.assert_not_called()

    def test_init_shuts_down_executor_on_eager_load_failure(self):
        """If Model.download raises during __init__, the base executor's shutdown is called.

        We patch ThreadPoolExecutor at the source so the base-class constructor
        gets back a real mock instance whose ``shutdown`` is observable; the
        previous version patched the in-class helper, which couldn't catch a
        regression where the helper became a no-op.
        """
        from concurrent.futures import ThreadPoolExecutor

        mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id"),
            patch(f"{AIC_QUAIL_VAD_MODULE}.Model") as mock_model_cls,
            patch(
                "pipecat.audio.vad.vad_analyzer.ThreadPoolExecutor",
                return_value=mock_executor_instance,
            ),
        ):
            mock_model_cls.download.side_effect = RuntimeError("CDN unreachable")
            with self.assertRaises(RuntimeError):
                self.AICQuailVADAnalyzer(license_key="test-key")
        mock_executor_instance.shutdown.assert_called_once_with(wait=False)

    def test_init_tolerates_executor_shutdown_failure(self):
        """If executor.shutdown itself raises during eager-load cleanup, the
        original construction error still propagates."""
        from concurrent.futures import ThreadPoolExecutor

        mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
        mock_executor_instance.shutdown.side_effect = RuntimeError("shutdown nope")
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id"),
            patch(f"{AIC_QUAIL_VAD_MODULE}.Model") as mock_model_cls,
            patch(
                "pipecat.audio.vad.vad_analyzer.ThreadPoolExecutor",
                return_value=mock_executor_instance,
            ),
        ):
            mock_model_cls.download.side_effect = RuntimeError("CDN unreachable")
            with self.assertRaises(RuntimeError) as ctx:
                self.AICQuailVADAnalyzer(license_key="test-key")
        # The original error (CDN unreachable) must propagate — not the shutdown error.
        self.assertIn("CDN unreachable", str(ctx.exception))

    def test_init_shuts_down_executor_on_set_sdk_id_failure(self):
        """set_sdk_id is now inside the eager-load try/except so its failure
        also triggers the executor shutdown."""
        from concurrent.futures import ThreadPoolExecutor

        mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
        with (
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id",
                side_effect=RuntimeError("telemetry registration failed"),
            ),
            patch(f"{AIC_QUAIL_VAD_MODULE}.Model"),
            patch(
                "pipecat.audio.vad.vad_analyzer.ThreadPoolExecutor",
                return_value=mock_executor_instance,
            ),
        ):
            with self.assertRaises(RuntimeError):
                self.AICQuailVADAnalyzer(license_key="test-key")
        mock_executor_instance.shutdown.assert_called_once_with(wait=False)

    def test_init_shuts_down_executor_on_processor_init_failure(self):
        """Processor() failing during eager init (sample_rate passed to __init__)
        triggers the same executor-shutdown cleanup as earlier failure modes."""
        from concurrent.futures import ThreadPoolExecutor

        mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.set_sdk_id"),
            patch(f"{AIC_QUAIL_VAD_MODULE}.Model") as mock_model_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.Processor",
                side_effect=RuntimeError("license expired"),
            ),
            patch(
                "pipecat.audio.vad.vad_analyzer.ThreadPoolExecutor",
                return_value=mock_executor_instance,
            ),
        ):
            mock_model_cls.from_file.return_value = self.mock_model
            mock_model_cls.download.return_value = "/tmp/test.aicmodel"
            mock_config_cls.return_value = MagicMock()
            with self.assertRaises(RuntimeError):
                self.AICQuailVADAnalyzer(license_key="test-key", sample_rate=16000)
        mock_executor_instance.shutdown.assert_called_once_with(wait=False)

    def test_default_model_id(self):
        """Default model_id is the published standalone Quail VAD."""
        analyzer, _ = self._create_analyzer()
        self.assertEqual(analyzer._model_id, "quail-vad-2.0-xxs-16khz")
        self.assertEqual(analyzer._model_id, self.DEFAULT_QUAIL_VAD_MODEL_ID)

    def test_default_download_dir(self):
        """Default model_download_dir lives under the user's cache."""
        analyzer, _ = self._create_analyzer()
        expected = Path.home() / ".cache" / "pipecat" / "aic-models"
        self.assertEqual(analyzer._model_download_dir, expected)

    def test_custom_download_dir(self):
        """Caller-supplied model_download_dir is honored."""
        custom = Path("/tmp/custom-cache")
        analyzer, _ = self._create_analyzer(model_download_dir=custom)
        self.assertEqual(analyzer._model_download_dir, custom)

    def test_pending_vad_params_stored(self):
        """Constructor stashes optional VAD knobs for later application."""
        analyzer, _ = self._create_analyzer(
            speech_hold_duration=0.08,
            minimum_speech_duration=0.05,
            sensitivity=0.7,
        )
        self.assertEqual(analyzer._pending_speech_hold_duration, 0.08)
        self.assertEqual(analyzer._pending_minimum_speech_duration, 0.05)
        self.assertEqual(analyzer._pending_sensitivity, 0.7)

    def test_construction_eagerly_loads_model_via_model_id(self):
        """__init__ downloads and loads the model so cold-start happens off-hot-path."""
        _, mocks = self._create_analyzer()
        mocks["Model"].download.assert_called_once_with(
            "quail-vad-2.0-xxs-16khz",
            str(Path.home() / ".cache" / "pipecat" / "aic-models"),
        )
        mocks["Model"].from_file.assert_called_once_with("/tmp/test.aicmodel")

    def test_construction_eagerly_loads_model_via_model_path(self):
        """model_path skips the download step entirely."""
        _, mocks = self._create_analyzer(model_id=None, model_path=Path("/tmp/custom.aicmodel"))
        mocks["Model"].download.assert_not_called()
        mocks["Model"].from_file.assert_called_once_with("/tmp/custom.aicmodel")

    def test_eager_init_when_sample_rate_supplied(self):
        """sample_rate in __init__ triggers immediate processor construction."""
        analyzer, mocks = self._create_analyzer(sample_rate=16000)
        self.assertEqual(analyzer._frames_per_block, 160)
        self.assertIsNotNone(analyzer._in_f32)
        self.assertEqual(analyzer._in_f32.shape, (1, 160))
        mocks["Processor"].assert_called_once()

    # --- set_sample_rate -----------------------------------------------------

    def test_set_sample_rate_creates_processor(self):
        """set_sample_rate after construction creates the processor."""
        analyzer, _ = self._create_analyzer()
        mock_processor_cls = self._initialize_at(analyzer, 16000)
        mock_processor_cls.assert_called_once()
        self.assertEqual(analyzer.sample_rate, 16000)
        self.assertEqual(analyzer._frames_per_block, 160)

    def test_set_sample_rate_does_not_reload_model(self):
        """A second set_sample_rate call must not re-call Model.from_file/download."""
        analyzer, mocks = self._create_analyzer()
        # __init__ already loaded the model once.
        self.assertEqual(mocks["Model"].from_file.call_count, 1)
        # set_sample_rate within a fresh patch should not touch Model at all.
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.Model") as fresh_model_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=self.mock_processor),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)
            fresh_model_cls.from_file.assert_not_called()
            fresh_model_cls.download.assert_not_called()

    def test_set_sample_rate_processor_init_failure_propagates(self):
        """Processor() raising at init propagates so the pipeline crashes loudly."""
        analyzer, _ = self._create_analyzer()
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.Processor",
                side_effect=RuntimeError("license expired"),
            ),
        ):
            mock_config_cls.return_value = MagicMock()
            with self.assertRaises(RuntimeError):
                analyzer.set_sample_rate(16000)

    def test_set_sample_rate_rolls_back_state_on_processor_init_failure(self):
        """If Processor() raises mid-set_sample_rate, the previous state is restored.

        Regression guard against a half-initialized analyzer with new
        frames_per_block but no working processor.
        """
        analyzer, _ = self._create_analyzer()
        # Successfully initialize at 16000 first.
        self._initialize_at(analyzer, 16000)
        old_processor = analyzer._processor
        old_vad_ctx = analyzer._vad_ctx
        old_frames = analyzer._frames_per_block
        old_in_f32 = analyzer._in_f32

        # Now fail a re-init at a different rate. State must be restored.
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.Processor",
                side_effect=RuntimeError("license expired"),
            ),
        ):
            mock_config_cls.return_value = MagicMock()
            with self.assertRaises(RuntimeError):
                analyzer.set_sample_rate(8000)

        self.assertIs(analyzer._processor, old_processor)
        self.assertIs(analyzer._vad_ctx, old_vad_ctx)
        self.assertEqual(analyzer._frames_per_block, old_frames)
        self.assertIs(analyzer._in_f32, old_in_f32)

    def test_set_sample_rate_failure_preserves_old_processor_vad_state(self):
        """Rollback must not silently reset the old processor's VAD state.

        Regression guard for the round-2 bug where the old-processor reset()
        ran before the new Processor() was constructed.
        """
        analyzer, _ = self._create_analyzer()
        first_processor = MockProcessorSync()
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=first_processor),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)

        # Fail the next set_sample_rate at Processor construction.
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_QUAIL_VAD_MODULE}.Processor",
                side_effect=RuntimeError("license expired"),
            ),
        ):
            mock_config_cls.return_value = MagicMock()
            with self.assertRaises(RuntimeError):
                analyzer.set_sample_rate(8000)

        # The old processor must NOT have been reset — otherwise the rollback
        # restores a wiped processor instead of the working one.
        self.assertFalse(first_processor.processor_ctx.reset_called)

    def test_set_sample_rate_reinit_resets_old_processor(self):
        """A second set_sample_rate calls reset() on the previous processor."""
        analyzer, _ = self._create_analyzer()
        # First init at 16000.
        first_processor = MockProcessorSync()
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=first_processor),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)

        self.assertFalse(first_processor.processor_ctx.reset_called)

        # Second init triggers reset on the first processor.
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=MockProcessorSync()),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)

        self.assertTrue(first_processor.processor_ctx.reset_called)

    def test_set_sample_rate_reinit_tolerates_old_processor_reset_failure(self):
        """Old processor's reset() failing during re-init is logged, not raised."""
        analyzer, _ = self._create_analyzer()
        # First init.
        first_processor = MockProcessorSync()
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=first_processor),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)

        # Make the old processor's reset fail. Re-init must still succeed.
        first_processor.processor_ctx.reset = MagicMock(side_effect=RuntimeError("flaky"))
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=MockProcessorSync()),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)  # must not raise

    def test_set_sample_rate_uses_correct_frames_per_block_when_model_optimal_differs(self):
        """num_frames_required reflects model.get_optimal_num_frames after init.

        Regression guard for the ordering bug where super().set_sample_rate ran
        before _initialize_processor — base sized internal buffers against the
        160-fallback even when the model wanted a different window.
        """
        self.mock_model._optimal_num_frames = 240  # e.g. 16 kHz Quail VAD model
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.assertEqual(analyzer._frames_per_block, 240)
        self.assertEqual(analyzer.num_frames_required(), 240)
        # Base class also stores its own _vad_frames; verify it matches.
        self.assertEqual(analyzer._vad_frames, 240)

    # --- num_frames_required branches ----------------------------------------

    def test_num_frames_required_before_init(self):
        """No sample rate and no init: returns the safe 160 fallback."""
        analyzer, _ = self._create_analyzer(sample_rate=None)
        self.assertEqual(analyzer.num_frames_required(), 160)

    def test_num_frames_required_after_init(self):
        """Post-init: returns the model's optimal frame count."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.assertEqual(analyzer.num_frames_required(), 160)

    def test_num_frames_required_defensive_middle_branch(self):
        """Defensive branch: sample_rate set but frames-per-block still 0.

        Not reachable in normal flow after the set_sample_rate reorder, but
        the fallback exists in case subclasses set _sample_rate directly.
        """
        analyzer, _ = self._create_analyzer(sample_rate=None)
        analyzer._sample_rate = 24000  # Simulate base setting rate without _initialize_processor
        self.assertEqual(analyzer._frames_per_block, 0)
        self.assertEqual(analyzer.num_frames_required(), 240)  # 24000 * 0.01

    # --- voice_confidence ----------------------------------------------------

    def test_voice_confidence_before_init_returns_zero(self):
        """No processor yet → no confidence."""
        analyzer, _ = self._create_analyzer()
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

    def test_voice_confidence_reports_speech(self):
        """When VAD says speech, return 1.0."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.mock_processor.vad_ctx.speech_detected = True
        # 10 ms at 16 kHz int16 → 320 bytes.
        confidence = analyzer.voice_confidence(b"\x00" * 320)
        self.assertEqual(confidence, 1.0)
        self.assertEqual(len(self.mock_processor.process_calls), 1)
        self.assertEqual(self.mock_processor.process_calls[0].shape, (1, 160))

    def test_voice_confidence_reports_silence(self):
        """When VAD says no speech, return 0.0."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.mock_processor.vad_ctx.speech_detected = False
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

    def test_voice_confidence_swallows_sdk_errors(self):
        """Exceptions from processor.process() return 0.0 (pipeline stays alive)."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.mock_processor.process = MagicMock(side_effect=RuntimeError("boom"))
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

    def test_voice_confidence_swallows_is_speech_detected_errors(self):
        """is_speech_detected() raising after process() succeeds returns 0.0
        and re-arms the error latch (distinct path from process() failure)."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        # process() succeeds; the failure happens in is_speech_detected().
        self.mock_processor.vad_ctx.raise_on_detect = True
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)
        self.assertTrue(analyzer._inference_error_logged)

    def test_voice_confidence_logs_inference_error_once(self):
        """Persistent SDK errors log at ERROR once, then go silent."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.mock_processor.process = MagicMock(side_effect=RuntimeError("boom"))

        self.assertFalse(analyzer._inference_error_logged)
        analyzer.voice_confidence(b"\x00" * 320)
        self.assertTrue(analyzer._inference_error_logged)
        # Second call: still returns 0.0, no second ERROR-level log emission.
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)
        self.assertTrue(analyzer._inference_error_logged)

    def test_voice_confidence_resets_error_latch_on_success(self):
        """A successful inference re-arms the error latch so fresh errors after a
        recovery surface at ERROR level rather than being buried at DEBUG.
        """
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        # First call fails — sets the latch.
        self.mock_processor.process = MagicMock(side_effect=RuntimeError("transient"))
        analyzer.voice_confidence(b"\x00" * 320)
        self.assertTrue(analyzer._inference_error_logged)
        # Recovery: a successful call clears the latch.
        self.mock_processor.process = MagicMock(return_value=None)
        analyzer.voice_confidence(b"\x00" * 320)
        self.assertFalse(analyzer._inference_error_logged)

    def test_voice_confidence_rejects_wrong_buffer_size(self):
        """Buffers not matching frames_per_block * 2 bytes return 0.0 without calling process."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        # 161 frames * 2 bytes = 322 bytes (off by one frame).
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 322), 0.0)
        self.assertTrue(analyzer._buffer_size_warning_logged)
        self.assertEqual(len(self.mock_processor.process_calls), 0)

    def test_buffer_size_warning_latch_resets_on_reinit(self):
        """A successful set_sample_rate reset re-arms the buffer-size warning latch."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        # Trip the latch with a bad-size call.
        analyzer.voice_confidence(b"\x00" * 322)
        self.assertTrue(analyzer._buffer_size_warning_logged)
        # Re-init at the same rate. The latch must clear so the next bad-size
        # call at the new configuration logs again.
        with (
            patch(f"{AIC_QUAIL_VAD_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_QUAIL_VAD_MODULE}.Processor", return_value=MockProcessorSync()),
        ):
            mock_config_cls.return_value = MagicMock()
            analyzer.set_sample_rate(16000)
        self.assertFalse(analyzer._buffer_size_warning_logged)

    # --- VAD parameters ------------------------------------------------------

    def test_vad_params_applied_to_vad_context(self):
        """Constructor-supplied tuning knobs are pushed to VadContext."""
        analyzer, _ = self._create_analyzer(
            speech_hold_duration=0.08,
            minimum_speech_duration=0.05,
            sensitivity=0.7,
        )
        self._initialize_at(analyzer, 16000)
        params = self.mock_processor.vad_ctx.parameters_set
        self.assertEqual(len(params), 3)
        self.assertEqual([v for _, v in params], [0.08, 0.05, 0.7])

    def test_vad_parameter_first_failure_does_not_drop_subsequent(self):
        """Per-parameter try/except means one failure doesn't silently drop others."""
        analyzer, _ = self._create_analyzer(
            speech_hold_duration=0.08,
            minimum_speech_duration=0.05,
            sensitivity=0.7,
        )
        self._initialize_at(analyzer, 16000)
        # Configure VadContext to fail on the first parameter only.
        call_log = []

        def selective_set(param, value):
            call_log.append((param, value))
            if len(call_log) == 1:
                raise RuntimeError("first param rejected")

        analyzer._vad_ctx.set_parameter = selective_set
        analyzer._apply_vad_parameters()
        # All three params should have been attempted despite the first failure.
        self.assertEqual(len(call_log), 3)

    def test_vad_parameter_application_swallows_errors(self):
        """A failing set_parameter call is logged, not re-raised."""
        analyzer, _ = self._create_analyzer(sensitivity=0.7)
        self._initialize_at(analyzer, 16000)
        analyzer._vad_ctx.set_parameter = MagicMock(side_effect=RuntimeError("boom"))
        analyzer._apply_vad_parameters()  # must not raise

    def test_apply_vad_parameters_noop_without_context(self):
        """_apply_vad_parameters returns early when there is no VadContext."""
        analyzer, _ = self._create_analyzer(sensitivity=0.7)
        analyzer._vad_ctx = None
        analyzer._apply_vad_parameters()  # must not raise

    # --- Cleanup -------------------------------------------------------------

    async def test_cleanup_releases_resources(self):
        """cleanup() resets the processor context and nils out state."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.assertIsNotNone(analyzer._processor)
        await analyzer.cleanup()
        self.assertIsNone(analyzer._processor)
        self.assertIsNone(analyzer._vad_ctx)
        self.assertIsNone(analyzer._model)
        self.assertIsNone(analyzer._in_f32)
        self.assertEqual(analyzer._frames_per_block, 0)
        self.assertTrue(self.mock_processor.processor_ctx.reset_called)

    async def test_cleanup_tolerates_reset_failure(self):
        """cleanup() logs and continues if ProcessorContext.reset raises."""
        analyzer, _ = self._create_analyzer()
        self._initialize_at(analyzer, 16000)
        self.mock_processor.processor_ctx.reset = MagicMock(side_effect=RuntimeError("nope"))
        await analyzer.cleanup()  # must not raise
        self.assertIsNone(analyzer._processor)

    async def test_cleanup_without_init_is_safe(self):
        """cleanup() can be called before set_sample_rate."""
        analyzer, _ = self._create_analyzer()
        await analyzer.cleanup()  # must not raise


if __name__ == "__main__":
    unittest.main()

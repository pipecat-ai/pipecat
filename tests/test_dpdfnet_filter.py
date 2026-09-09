#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DPDFNetFilter.

These patch ``StreamEnhancer`` and ``MODEL_REGISTRY`` at their import site in
``pipecat.audio.filters.dpdfnet_filter``, so the suite runs whether or not the
``dpdfnet`` extra is installed and never downloads model weights.
"""

import os
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np

from pipecat.audio.filters.dpdfnet_filter import DPDFNetFilter
from pipecat.frames.frames import FilterEnableFrame
from tests.dpdfnet_mocks import (
    MOCK_MODEL_REGISTRY,
    ExplodingStreamEnhancer,
    FailingStreamEnhancer,
    MockStreamEnhancer,
)

FILTER_MODULE = "pipecat.audio.filters.dpdfnet_filter"


def tone_bytes(sample_rate: int, duration: float = 0.02, frequency: float = 440.0) -> bytes:
    """Build a PCM16 sine tone of the given duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    samples = np.sin(2 * np.pi * frequency * t) * 0.5
    return (samples * 32767.0).astype(np.int16).tobytes()


def patch_dpdfnet(enhancer_cls=MockStreamEnhancer, sample_rate=16000):
    """Patch the enhancer class and model registry at the filter's import site."""
    factory = partial(enhancer_cls, sample_rate=sample_rate)
    return patch.multiple(
        FILTER_MODULE,
        StreamEnhancer=factory,
        MODEL_REGISTRY=MOCK_MODEL_REGISTRY,
    )


class TestDPDFNetFilter(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_model_raises_in_constructor(self):
        """An unknown model name fails fast, at pipeline construction time."""
        with patch(f"{FILTER_MODULE}.MODEL_REGISTRY", MOCK_MODEL_REGISTRY):
            with self.assertRaises(ValueError):
                DPDFNetFilter(model="not-a-real-model")

    async def test_passthrough_before_start(self):
        """Before start(), audio is returned untouched."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            audio = tone_bytes(16000)
            self.assertEqual(await filter.filter(audio), audio)

    async def test_passthrough_when_disabled(self):
        """A disabled filter returns its input unmodified, not merely non-empty."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)
            await filter.process_frame(FilterEnableFrame(enable=False))

            audio = tone_bytes(16000)
            self.assertEqual(await filter.filter(audio), audio)

    async def test_buffering_returns_empty_then_audio(self):
        """The causal window means the first short chunk yields nothing."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            # 5 ms, well under the 20 ms analysis window.
            self.assertEqual(await filter.filter(tone_bytes(16000, duration=0.005)), b"")
            # Another 20 ms pushes it past a full window.
            self.assertTrue(len(await filter.filter(tone_bytes(16000, duration=0.020))) > 0)

    async def test_enhancement_applied(self):
        """Audio round-trips through the model with the expected scaling."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            audio = tone_bytes(16000, duration=0.100)
            out = await filter.filter(audio)

            self.assertTrue(len(out) > 0)
            self.assertNotEqual(out, audio)

            in_rms = np.sqrt(np.mean(np.frombuffer(audio, dtype=np.int16).astype(float) ** 2))
            out_rms = np.sqrt(np.mean(np.frombuffer(out, dtype=np.int16).astype(float) ** 2))
            # The mock applies a fixed 0.5 gain; allow for the partial first window.
            self.assertAlmostEqual(out_rms / in_rms, 0.5, delta=0.05)

    async def test_no_resamplers_when_rate_matches(self):
        """A transport already at the model's native rate skips resampling."""
        with patch_dpdfnet(sample_rate=16000):
            filter = DPDFNetFilter(model="dpdfnet2")
            await filter.start(sample_rate=16000)

            self.assertIsNone(filter._resampler_in)
            self.assertIsNone(filter._resampler_out)

    async def test_resampling_roundtrip_preserves_tone(self):
        """8 kHz transport into a 16 kHz model round-trips at the transport rate."""
        with patch_dpdfnet(sample_rate=16000):
            filter = DPDFNetFilter(model="dpdfnet2")
            await filter.start(sample_rate=8000)

            self.assertIsNotNone(filter._resampler_in)
            self.assertIsNotNone(filter._resampler_out)
            # Two distinct instances: one stream resampler cannot serve both
            # directions.
            self.assertIsNot(filter._resampler_in, filter._resampler_out)

            out = b""
            for _ in range(50):  # 1 second in 20 ms chunks
                out += await filter.filter(tone_bytes(8000, duration=0.020))

            # Output is at the transport rate, within a couple of hundred ms of
            # the input length.
            out_samples = len(out) // 2
            self.assertAlmostEqual(out_samples / 8000, 1.0, delta=0.2)

            # And it is still a 440 Hz tone.
            mid = np.frombuffer(out, dtype=np.int16).astype(np.float32)[2000:6000]
            spectrum = np.abs(np.fft.rfft(mid))
            peak_hz = np.fft.rfftfreq(len(mid), 1 / 8000)[np.argmax(spectrum)]
            self.assertAlmostEqual(peak_hz, 440.0, delta=50.0)

    async def test_model_always_called_at_native_rate(self):
        """The model is fed at its native rate, never handed the transport rate.

        Passing the transport rate would make dpdfnet resample each chunk
        independently with librosa, which has no history across chunks. We
        resample with a streaming resampler instead, so this must stay None.
        """
        with patch_dpdfnet(sample_rate=16000):
            filter = DPDFNetFilter(model="dpdfnet2")
            await filter.start(sample_rate=8000)

            for _ in range(10):
                await filter.filter(tone_bytes(8000, duration=0.020))

            self.assertTrue(filter._enhancer.rates_seen)
            self.assertTrue(all(rate is None for rate in filter._enhancer.rates_seen))

    async def test_model_load_failure_degrades_to_passthrough(self):
        """A failed download leaves the pipeline running on unfiltered audio."""
        with patch_dpdfnet(enhancer_cls=FailingStreamEnhancer):
            filter = DPDFNetFilter()
            # Must not raise: base_input calls start() without a try/except.
            await filter.start(sample_rate=16000)

            self.assertFalse(filter._dpdfnet_ready)
            audio = tone_bytes(16000)
            self.assertEqual(await filter.filter(audio), audio)

    async def test_dpdfnet_not_installed_degrades_to_passthrough(self):
        """With dpdfnet absent the filter is inert rather than fatal."""
        with patch.multiple(FILTER_MODULE, StreamEnhancer=None, MODEL_REGISTRY={}):
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            self.assertFalse(filter._dpdfnet_ready)
            audio = tone_bytes(16000)
            self.assertEqual(await filter.filter(audio), audio)

    async def test_inference_exception_degrades_to_passthrough(self):
        """An exception inside the model never reaches the transport's audio task."""
        with patch_dpdfnet(enhancer_cls=ExplodingStreamEnhancer):
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            audio = tone_bytes(16000, duration=0.100)
            self.assertEqual(await filter.filter(audio), audio)

    async def test_stop_is_idempotent(self):
        """stop() runs from the transport's stop, cancel and cleanup paths."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            await filter.stop()
            await filter.stop()
            await filter.stop()

            self.assertIsNone(filter._executor)
            self.assertIsNone(filter._enhancer)
            self.assertFalse(filter._dpdfnet_ready)

    async def test_filter_after_stop(self):
        """Audio arriving after teardown passes through instead of raising."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)
            await filter.stop()

            audio = tone_bytes(16000)
            self.assertEqual(await filter.filter(audio), audio)

    async def test_empty_input_returns_empty(self):
        """Empty input yields empty output, which the transport skips."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            self.assertEqual(await filter.filter(b""), b"")

    async def test_odd_length_input_carries_trailing_byte(self):
        """A misaligned chunk carries its trailing byte instead of dropping it.

        Dropping it would desync every later sample by one byte.
        """
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)

            chunk = tone_bytes(16000, duration=0.040)[:641]  # 1280 bytes, sliced to odd
            await filter.filter(chunk)
            self.assertEqual(filter._partial_byte, chunk[-1:])

            await filter.filter(chunk)
            # 641 + 641 = 1282 bytes = 641 whole samples, with no byte lost.
            self.assertEqual(filter._enhancer.samples_seen, 641)

    async def test_reset_on_reenable_only_on_transition(self):
        """Re-enabling resets stale recurrent state; a redundant enable does not."""
        with patch_dpdfnet():
            filter = DPDFNetFilter()
            await filter.start(sample_rate=16000)
            baseline = filter._enhancer.reset_count

            await filter.process_frame(FilterEnableFrame(enable=False))
            await filter.process_frame(FilterEnableFrame(enable=True))
            await filter.filter(tone_bytes(16000, duration=0.100))
            self.assertEqual(filter._enhancer.reset_count, baseline + 1)

            # Already enabled: no reset, so no hole punched in the audio.
            await filter.process_frame(FilterEnableFrame(enable=True))
            await filter.filter(tone_bytes(16000, duration=0.100))
            self.assertEqual(filter._enhancer.reset_count, baseline + 1)


@unittest.skipUnless(
    os.environ.get("PIPECAT_DPDFNET_ONNX_PATH"),
    "set PIPECAT_DPDFNET_ONNX_PATH to run against a real DPDFNet model",
)
class TestDPDFNetFilterIntegration(unittest.IsolatedAsyncioTestCase):
    """Opt-in checks against a real ONNX model. Never runs in CI."""

    async def test_real_model_enhances_audio(self):
        """A real model loads from disk and returns audio at the transport rate."""
        filter = DPDFNetFilter(
            model=os.environ.get("PIPECAT_DPDFNET_MODEL", "dpdfnet2"),
            onnx_path=os.environ["PIPECAT_DPDFNET_ONNX_PATH"],
        )
        await filter.start(sample_rate=16000)
        self.assertTrue(filter._dpdfnet_ready)

        out = b""
        for _ in range(25):  # 500 ms in 20 ms chunks
            out += await filter.filter(tone_bytes(16000, duration=0.020))

        self.assertTrue(len(out) > 0)
        await filter.stop()

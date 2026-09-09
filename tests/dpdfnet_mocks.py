#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared dpdfnet test mocks for the DPDFNet test suite.

Importing in: ``tests/test_dpdfnet_filter.py``. These stand in for
``dpdfnet.stream.StreamEnhancer`` so the suite runs without the ``dpdfnet``
extra installed and without downloading model weights in CI. Keep the buffering
behavior aligned with the live 0.6.0 surface so the suite stays representative.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MockModelInfo:
    """Stand-in for ``dpdfnet.models.ModelInfo``."""

    name: str
    sample_rate: int
    frame_ms: float = 20.0


MOCK_MODEL_REGISTRY = {
    "baseline": MockModelInfo("baseline", 16000),
    "dpdfnet2": MockModelInfo("dpdfnet2", 16000),
    "dpdfnet2_8khz": MockModelInfo("dpdfnet2_8khz", 8000),
    "dpdfnet2_48khz_hr": MockModelInfo("dpdfnet2_48khz_hr", 48000),
}


class MockStreamEnhancer:
    """Stand-in for ``dpdfnet.stream.StreamEnhancer``.

    Reproduces the buffering contract rather than the model: returns an empty
    array until one full analysis window has accumulated, then emits hop-aligned
    output. Applies a fixed 0.5 gain so tests can tell filtered audio apart from
    pass-through.
    """

    def __init__(self, model="dpdfnet2", onnx_path=None, verbose=False, *, sample_rate=16000):
        """Record construction args and size the window from the sample rate."""
        self.model = model
        self.onnx_path = onnx_path
        self.verbose = verbose
        self.model_sample_rate = sample_rate
        # dpdfnet uses a 20 ms window with a hop of half that.
        self._win_len = int(sample_rate * 0.020)
        self._hop = self._win_len // 2
        self.reset_count = 0
        self.rates_seen = []
        self.samples_seen = 0
        self.reset()

    def reset(self):
        """Drop the recurrent state and the input buffer."""
        self._buf = np.zeros(0, dtype=np.float32)
        self.reset_count += 1

    def process(self, chunk, sample_rate=None):
        """Buffer a chunk and emit whatever complete hops are available."""
        self.rates_seen.append(sample_rate)
        chunk = np.asarray(chunk, dtype=np.float32)
        self.samples_seen += chunk.size
        self._buf = np.concatenate([self._buf, chunk])

        out = []
        while len(self._buf) >= self._win_len:
            out.append(self._buf[: self._hop] * 0.5)
            self._buf = self._buf[self._hop :]

        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class FailingStreamEnhancer:
    """A ``StreamEnhancer`` whose construction fails, as a failed download does."""

    def __init__(self, *args, **kwargs):
        """Raise as ``resolve_model()`` does when weights cannot be fetched."""
        raise RuntimeError("failed to download model weights")


class ExplodingStreamEnhancer(MockStreamEnhancer):
    """A ``StreamEnhancer`` that constructs cleanly but fails during inference."""

    def process(self, chunk, sample_rate=None):
        """Raise on every call, to exercise the pass-through recovery path."""
        raise RuntimeError("inference failed")

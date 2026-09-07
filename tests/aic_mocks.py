#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared aic_sdk test mocks for the AIC test suite.

Importing in: ``tests/test_aic_filter.py`` and ``tests/test_aic_quail_vad.py``.
Keep behavior aligned with the live ``aic_sdk`` 3.0 surface so the suite stays
representative.
"""

from typing import Any

import numpy as np


class MockVadContext:
    """Stand-in for ``aic_sdk.VadContext``."""

    def __init__(
        self,
        speech_detected: bool = False,
        raw_probability: float = 0.0,
        raise_on_detect: bool = False,
        raise_on_set_param: bool = False,
        prediction_delay: int = 0,
    ) -> None:
        self.speech_detected = speech_detected
        self.raw_probability = raw_probability
        # raise_on_detect drives both query paths so error tests can target
        # whichever the code under test calls (is_speech_detected /
        # raw_vad_probability).
        self.raise_on_detect = raise_on_detect
        self.raise_on_set_param = raise_on_set_param
        self.prediction_delay = prediction_delay
        self.parameters_set: list[tuple] = []
        self.reset_called = False

    def is_speech_detected(self) -> bool:
        if self.raise_on_detect:
            raise RuntimeError("VAD error")
        return self.speech_detected

    def raw_vad_probability(self) -> float:
        if self.raise_on_detect:
            raise RuntimeError("VAD error")
        return self.raw_probability

    def set_parameter(self, param: Any, value: float) -> None:
        if self.raise_on_set_param:
            raise RuntimeError("Param error")
        self.parameters_set.append((param, value))

    def get_prediction_delay(self) -> int:
        return self.prediction_delay

    def reset(self) -> None:
        self.reset_called = True


class MockProcessorContext:
    """Stand-in for ``aic_sdk.ProcessorContext``."""

    def __init__(self) -> None:
        self.parameters_set: list[tuple] = []
        self.reset_called = False
        self._audio_delay = 0

    def get_audio_delay(self) -> int:
        return self._audio_delay

    def set_parameter(self, param: Any, value: float) -> None:
        self.parameters_set.append((param, value))

    def reset(self) -> None:
        self.reset_called = True


class MockProcessorAsync:
    """Stand-in for ``aic_sdk.ProcessorAsync`` used by :class:`AICFilter`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.processor_ctx = MockProcessorContext()
        self.process_calls: list[np.ndarray] = []
        self.terminated = False

    def get_context(self) -> MockProcessorContext:
        return self.processor_ctx

    async def process_async(self, audio_array: np.ndarray) -> np.ndarray:
        self.process_calls.append(audio_array.copy())
        return audio_array.copy()

    async def terminate_session_async(self) -> None:
        self.terminated = True


class MockVadSync:
    """Stand-in for ``aic_sdk.Vad`` used by :class:`AICQuailVADAnalyzer`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.vad_ctx = MockVadContext()
        self.process_calls: list[np.ndarray] = []
        self.terminated = False

    def get_context(self) -> MockVadContext:
        return self.vad_ctx

    def process(self, audio: np.ndarray) -> None:
        self.process_calls.append(audio.copy())

    def terminate_session(self) -> None:
        self.terminated = True


class MockModel:
    """Stand-in for ``aic_sdk.Model``.

    ``optimal_block_size`` is configurable so tests can exercise paths where
    the model's optimal block size differs from the 10 ms / 160-sample fallback.
    """

    def __init__(self, model_id: str = "test-model", optimal_block_size: int = 160) -> None:
        self._model_id = model_id
        self._optimal_block_size = optimal_block_size
        self._optimal_sample_rate = 16000

    def get_id(self) -> str:
        return self._model_id

    def get_optimal_block_size(self, sample_rate: int) -> int:
        return self._optimal_block_size

    def get_optimal_sample_rate(self) -> int:
        return self._optimal_sample_rate

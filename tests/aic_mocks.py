#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared aic_sdk test mocks for the AIC test suite.

Importing in: ``tests/test_aic_filter.py``, ``tests/test_aic_vad.py``,
``tests/test_aic_quail_vad.py``. Keep behavior aligned with the live
``aic_sdk`` 2.5.0 surface so the suite stays representative.
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
    ) -> None:
        self.speech_detected = speech_detected
        self.raw_probability = raw_probability
        # raise_on_detect drives both query paths so error tests can target
        # whichever the code under test calls (is_speech_detected /
        # raw_vad_probability).
        self.raise_on_detect = raise_on_detect
        self.raise_on_set_param = raise_on_set_param
        self.parameters_set: list[tuple] = []

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


class MockProcessorContext:
    """Stand-in for ``aic_sdk.ProcessorContext`` (sync surface)."""

    def __init__(self) -> None:
        self.parameters_set: list[tuple] = []
        self.reset_called = False
        self._output_delay = 0

    def get_output_delay(self) -> int:
        return self._output_delay

    def set_parameter(self, param: Any, value: float) -> None:
        self.parameters_set.append((param, value))

    def reset(self) -> None:
        self.reset_called = True


class MockProcessorAsync:
    """Stand-in for ``aic_sdk.ProcessorAsync`` used by :class:`AICFilter`.

    In aic-sdk 2.3.0 the sync ``Processor`` and async ``ProcessorAsync`` both
    expose sync ``get_*_context()`` (the async getters were reverted before
    release). This mock matches that final shape.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.processor_ctx = MockProcessorContext()
        self.vad_ctx = MockVadContext()

    def get_processor_context(self) -> MockProcessorContext:
        return self.processor_ctx

    def get_vad_context(self) -> MockVadContext:
        return self.vad_ctx

    async def process_async(self, audio_array: np.ndarray) -> np.ndarray:
        return audio_array.copy()


class MockProcessorSync:
    """Stand-in for ``aic_sdk.Processor`` used by :class:`AICQuailVADAnalyzer`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.vad_ctx = MockVadContext()
        self.processor_ctx = MockProcessorContext()
        self.process_calls: list[np.ndarray] = []

    def get_vad_context(self) -> MockVadContext:
        return self.vad_ctx

    def get_processor_context(self) -> MockProcessorContext:
        return self.processor_ctx

    def process(self, audio: np.ndarray) -> np.ndarray:
        self.process_calls.append(audio.copy())
        return audio.copy()


class MockModel:
    """Stand-in for ``aic_sdk.Model``.

    ``optimal_num_frames`` is configurable so tests can exercise paths where
    the model's optimal frame count differs from the 10 ms / 160-frame fallback.
    """

    def __init__(self, model_id: str = "test-model", optimal_num_frames: int = 160) -> None:
        self._model_id = model_id
        self._optimal_num_frames = optimal_num_frames
        self._optimal_sample_rate = 16000

    def get_id(self) -> str:
        return self._model_id

    def get_optimal_num_frames(self, sample_rate: int) -> int:
        return self._optimal_num_frames

    def get_optimal_sample_rate(self) -> int:
        return self._optimal_sample_rate

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


class MockAnalysisResult:
    """Stand-in for ``aic_sdk.AnalysisResult`` (Tyto)."""

    def __init__(
        self,
        risk_score: float = 0.0,
        speaker_reverb: float = 0.0,
        speaker_loudness: float = 0.0,
        interfering_speech: float = 0.0,
        codec_degradation: float = 0.0,
        noise: float = 0.0,
        packet_loss: float = 0.0,
    ) -> None:
        self.risk_score = risk_score
        self.speaker_reverb = speaker_reverb
        self.speaker_loudness = speaker_loudness
        self.interfering_speech = interfering_speech
        self.codec_degradation = codec_degradation
        self.noise = noise
        self.packet_loss = packet_loss


class MockCollector:
    """Stand-in for ``aic_sdk.Collector`` (Tyto).

    Records ``initialize`` configs and buffered arrays so tests can assert on the
    audio tap. ``raise_on_buffer`` exercises the buffering error path.
    """

    def __init__(self, raise_on_buffer: bool = False) -> None:
        self.raise_on_buffer = raise_on_buffer
        self.initialized_with: list[Any] = []
        self.buffer_calls: list[np.ndarray] = []

    def initialize(self, config: Any) -> None:
        self.initialized_with.append(config)

    def buffer(self, audio: np.ndarray) -> None:
        if self.raise_on_buffer:
            raise RuntimeError("buffer error")
        self.buffer_calls.append(audio.copy())


class MockAnalyzer:
    """Stand-in for ``aic_sdk.Analyzer`` (Tyto).

    Returns ``result`` from ``analyze_buffered`` (default all-zeros);
    ``raise_on_analyze`` exercises the analysis error path.
    """

    def __init__(
        self,
        result: MockAnalysisResult | None = None,
        raise_on_analyze: bool = False,
    ) -> None:
        self.result = result or MockAnalysisResult()
        self.raise_on_analyze = raise_on_analyze
        self.analyze_calls = 0

    def analyze_buffered(self) -> MockAnalysisResult:
        self.analyze_calls += 1
        if self.raise_on_analyze:
            raise RuntimeError("analyze error")
        return self.result

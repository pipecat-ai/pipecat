#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.metrics.metrics import MetricsData, SmartTurnMetricsData

# Default timing parameters
STOP_SECS = 3
PRE_SPEECH_MS = 0
MAX_DURATION_SECONDS = 8  # Max allowed segment duration
USE_ONLY_LAST_VAD_SEGMENT = True


class SmartTurnParams(BaseModel):
    stop_secs: float = STOP_SECS
    pre_speech_ms: float = PRE_SPEECH_MS
    max_duration_secs: float = MAX_DURATION_SECONDS
    # not exposing this for now yet until the model can handle it.
    # use_only_last_vad_segment: bool = USE_ONLY_LAST_VAD_SEGMENT


class SmartTurnTimeoutException(Exception):
    pass


class BaseSmartTurn(BaseTurnAnalyzer):
    def __init__(
        self, *, sample_rate: Optional[int] = None, params: Optional[SmartTurnParams] = None
    ):
        super().__init__(sample_rate=sample_rate)
        self._params = params or SmartTurnParams()
        # Configuration
        self._stop_ms = self._params.stop_secs * 1000  # silence threshold in ms
        # Inference state
        self._audio_buffer = []
        self._speech_triggered = False
        self._silence_ms = 0
        self._speech_start_time = 0

    @property
    def speech_triggered(self) -> bool:
        return self._speech_triggered

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        # Convert raw audio to float32 format and append to the buffer
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.append((time.time(), audio_float32))

        state = EndOfTurnState.INCOMPLETE

        if is_speech:
            # Reset silence tracking on speech
            self._silence_ms = 0
            self._speech_triggered = True
            if self._speech_start_time == 0:
                self._speech_start_time = time.time()
        else:
            if self._speech_triggered:
                chunk_duration_ms = len(audio_int16) / (self._sample_rate / 1000)
                self._silence_ms += chunk_duration_ms
                # If silence exceeds threshold, mark end of turn
                if self._silence_ms >= self._stop_ms:
                    logger.debug(
                        f"End of Turn complete due to stop_secs. Silence in ms: {self._silence_ms}"
                    )
                    state = EndOfTurnState.COMPLETE
                    self._clear(state)
            else:
                # Trim buffer to prevent unbounded growth before speech
                max_buffer_time = (
                    (self._params.pre_speech_ms / 1000)
                    + self._params.stop_secs
                    + self._params.max_duration_secs
                )
                while (
                    self._audio_buffer and self._audio_buffer[0][0] < time.time() - max_buffer_time
                ):
                    self._audio_buffer.pop(0)

        return state

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        state, result = await self._process_speech_segment(self._audio_buffer)
        if state == EndOfTurnState.COMPLETE or USE_ONLY_LAST_VAD_SEGMENT:
            self._clear(state)
        logger.debug(f"End of Turn result: {state}")
        return state, result

    def clear(self):
        self._clear(EndOfTurnState.COMPLETE)

    def _clear(self, turn_state: EndOfTurnState):
        # If the state is still incomplete, keep the _speech_triggered as True
        self._speech_triggered = turn_state == EndOfTurnState.INCOMPLETE
        self._audio_buffer = []
        self._speech_start_time = 0
        self._silence_ms = 0

    async def _process_speech_segment(
        self, audio_buffer
    ) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        state = EndOfTurnState.INCOMPLETE

        if not audio_buffer:
            return state, None

        # Extract recent audio segment for prediction
        start_time = self._speech_start_time - (self._params.pre_speech_ms / 1000)
        start_index = 0
        for i, (t, _) in enumerate(audio_buffer):
            if t >= start_time:
                start_index = i
                break

        end_index = len(audio_buffer) - 1

        # Extract the audio segment
        segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index : end_index + 1]]
        segment_audio = np.concatenate(segment_audio_chunks)

        # Limit maximum duration
        max_samples = int(self._params.max_duration_secs * self.sample_rate)
        if len(segment_audio) > max_samples:
            # slices the array to keep the last max_samples samples, discarding the earlier part.
            segment_audio = segment_audio[-max_samples:]

        result_data = None

        if len(segment_audio) > 0:
            start_time = time.perf_counter()
            try:
                result = await self._predict_endpoint(segment_audio)
                state = (
                    EndOfTurnState.COMPLETE
                    if result["prediction"] == 1
                    else EndOfTurnState.INCOMPLETE
                )
                end_time = time.perf_counter()

                # Calculate processing time
                e2e_processing_time_ms = (end_time - start_time) * 1000

                # Extract metrics from the nested structure
                metrics = result.get("metrics", {})
                inference_time = metrics.get("inference_time", 0)
                total_time = metrics.get("total_time", 0)

                # Prepare the result data
                result_data = SmartTurnMetricsData(
                    processor="BaseSmartTurn",
                    is_complete=result["prediction"] == 1,
                    probability=result["probability"],
                    inference_time_ms=inference_time * 1000,
                    server_total_time_ms=total_time * 1000,
                    e2e_processing_time_ms=e2e_processing_time_ms,
                )

                logger.trace(
                    f"Prediction: {'Complete' if result_data.is_complete else 'Incomplete'}"
                )
                logger.trace(f"Probability of complete: {result_data.probability:.4f}")
                logger.trace(f"Inference time: {result_data.inference_time_ms:.2f}ms")
                logger.trace(f"Server total time: {result_data.server_total_time_ms:.2f}ms")
                logger.trace(f"E2E processing time: {result_data.e2e_processing_time_ms:.2f}ms")
            except SmartTurnTimeoutException:
                logger.debug(
                    f"End of Turn complete due to stop_secs. Silence in ms: {self._silence_ms}"
                )
                state = EndOfTurnState.COMPLETE

        else:
            logger.trace(f"params: {self._params}, stop_ms: {self._stop_ms}")
            logger.trace("Captured empty audio segment, skipping prediction.")

        return state, result_data

    @abstractmethod
    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Abstract method to predict if a turn has ended based on audio.

        Args:
            audio_array: Float32 numpy array of audio samples at 16kHz.

        Returns:
            Dictionary with:
              - prediction: 1 if turn is complete, else 0
              - probability: Confidence of the prediction
        """
        pass

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel


class EndOfTurnState(Enum):
    COMPLETE = 1
    INCOMPLETE = 2


STOP_SECS = 1
PRE_SPEECH_MS = 200
MAX_DURATION_SECONDS = 8  # Maximum duration for the smart turn model


class SmartTurnParams(BaseModel):
    stop_secs: float = STOP_SECS
    pre_speech_ms: float = PRE_SPEECH_MS
    max_duration_secs: float = MAX_DURATION_SECONDS


class BaseSmartTurn(ABC):
    def __init__(self, *, sample_rate: Optional[int] = None, params: SmartTurnParams = SmartTurnParams()):
        self._init_sample_rate = sample_rate
        self._params = params
        # settings variables
        self._sample_rate = 0
        self._chunk_size_ms = 0
        self._stop_ms = self._params.stop_secs * 1000
        # inference variables
        self._audio_buffer = []
        self._speech_triggered = False
        self._silence_frames = 0
        self._speech_start_time = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_sample_rate(self, sample_rate: int):
        self._sample_rate = self._init_sample_rate or sample_rate

    @property
    def chunk_size_ms(self) -> int:
        return self._chunk_size_ms

    def set_chunk_size_ms(self, chunk_size_ms: int):
        self._chunk_size_ms = chunk_size_ms

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        # Divide by 32768 because we have signed 16-bit data.
        audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0

        state = EndOfTurnState.INCOMPLETE
        if is_speech:
            self._silence_frames = 0
            self._speech_triggered = True
            if self._speech_start_time is None:
                self._speech_start_time = time.time()
            self._audio_buffer.append((time.time(), audio_float32))
        else:
            if self._speech_triggered:
                self._audio_buffer.append((time.time(), audio_float32))
                self._silence_frames += 1
                if self._silence_frames * self._chunk_size_ms >= self._stop_ms:
                    logger.debug("End of Turn complete due to stop_secs.")
                    state = EndOfTurnState.COMPLETE
                    self._clear()
            else:
                # Keep buffering some silence before potential speech starts
                self._audio_buffer.append((time.time(), audio_float32))
                # Keep the buffer size reasonable, assuming CHUNK is small
                max_buffer_time = (
                    self._params.pre_speech_ms + self._stop_ms
                ) / 1000 + self._params.max_duration_secs  # Some extra buffer
                while (
                    self._audio_buffer and self._audio_buffer[0][0] < time.time() - max_buffer_time
                ):
                    self._audio_buffer.pop(0)

        return state

    def analyze_end_of_turn(self) -> EndOfTurnState:
        logger.debug("Analyzing End of Turn...")
        state = self._process_speech_segment(self._audio_buffer)
        if state == EndOfTurnState.COMPLETE:
            self._clear()

        logger.debug(f"End of Turn result: {state}")
        return state

    def _clear(self):
        self._speech_triggered = False
        self._audio_buffer = []
        self._speech_start_time = None
        self._silence_frames = 0

    def _process_speech_segment(self, audio_buffer) -> EndOfTurnState:
        state = EndOfTurnState.INCOMPLETE

        if not audio_buffer:
            return state

        # Find start and end indices for the segment
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

        # Remove (self._stop_ms - 200)ms from the end of the segment
        samples_to_remove = int((self._stop_ms - 200) / 1000 * self.sample_rate)
        segment_audio = segment_audio[:-samples_to_remove]

        # Limit maximum duration
        if len(segment_audio) / self.sample_rate > self._params.max_duration_secs:
            segment_audio = segment_audio[: int(self._params.max_duration_secs * self.sample_rate)]

        # No resampling needed as both recording and prediction use 16000 Hz
        segment_audio_resampled = segment_audio

        if len(segment_audio_resampled) > 0:
            # Call the new predict_endpoint function with the audio data
            start_time = time.perf_counter()

            result = self._predict_endpoint(segment_audio_resampled)

            state = (
                EndOfTurnState.COMPLETE if result["prediction"] == 1 else EndOfTurnState.INCOMPLETE
            )

            end_time = time.perf_counter()

            logger.debug("--------")
            logger.debug(f"Prediction: {'Complete' if result['prediction'] == 1 else 'Incomplete'}")
            logger.debug(f"Probability of complete: {result['probability']:.4f}")
            logger.debug(f"Prediction took {(end_time - start_time) * 1000:.2f}ms seconds")
        else:
            logger.debug("Captured empty audio segment, skipping prediction.")

        return state

    @abstractmethod
    def _predict_endpoint(self, buffer: np.ndarray) -> Dict[str, any]:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            audio_array: Numpy array containing audio samples at 16kHz

        Returns:
            Dictionary containing prediction results:
            - prediction: 1 for complete, 0 for incomplete
            - probability: Probability of completion class
        """
        pass

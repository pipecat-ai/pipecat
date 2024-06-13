#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import abstractmethod
from enum import Enum

from pydantic.main import BaseModel

from pipecat.utils.audio import calculate_audio_volume, exp_smoothing


class VADState(Enum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.8
    min_volume: float = 0.6


class VADAnalyzer:

    def __init__(self, sample_rate: int, num_channels: int, params: VADParams):
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * num_channels * 2

        vad_frames_per_sec = self._vad_frames / self._sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

        self._vad_buffer = b""

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

    @property
    def sample_rate(self):
        return self._sample_rate

    @abstractmethod
    def num_frames_required(self) -> int:
        pass

    @abstractmethod
    def voice_confidence(self, buffer) -> float:
        pass

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = calculate_audio_volume(audio, self._sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    def analyze_audio(self, buffer) -> VADState:
        self._vad_buffer += buffer

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        audio_frames = self._vad_buffer[:num_required_bytes]
        self._vad_buffer = self._vad_buffer[num_required_bytes:]

        confidence = self.voice_confidence(audio_frames)

        volume = self._get_smoothed_volume(audio_frames)
        self._prev_volume = volume

        speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

        if speaking:
            match self._vad_state:
                case VADState.QUIET:
                    self._vad_state = VADState.STARTING
                    self._vad_starting_count = 1
                case VADState.STARTING:
                    self._vad_starting_count += 1
                case VADState.STOPPING:
                    self._vad_state = VADState.SPEAKING
                    self._vad_stopping_count = 0
        else:
            match self._vad_state:
                case VADState.STARTING:
                    self._vad_state = VADState.QUIET
                    self._vad_starting_count = 0
                case VADState.SPEAKING:
                    self._vad_state = VADState.STOPPING
                    self._vad_stopping_count = 1
                case VADState.STOPPING:
                    self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

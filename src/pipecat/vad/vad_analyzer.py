#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import abstractmethod
from enum import Enum


class VADState(Enum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADAnalyzer:

    def __init__(
            self,
            sample_rate,
            num_channels,
            vad_confidence=0.5,
            vad_start_s=0.2,
            vad_stop_s=0.8):
        self._sample_rate = sample_rate
        self._vad_confidence = vad_confidence
        self._vad_start_s = vad_start_s
        self._vad_stop_s = vad_stop_s
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * num_channels * 2

        vad_frame_s = self._vad_frames / self._sample_rate

        self._vad_start_frames = round(self._vad_start_s / vad_frame_s)
        self._vad_stop_frames = round(self._vad_stop_s / vad_frame_s)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

        self._vad_buffer = b""

    @property
    def sample_rate(self):
        return self._sample_rate

    @abstractmethod
    def num_frames_required(self) -> int:
        pass

    @abstractmethod
    def voice_confidence(self, buffer) -> float:
        pass

    def analyze_audio(self, buffer) -> VADState:
        self._vad_buffer += buffer

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        audio_frames = self._vad_buffer[:num_required_bytes]
        self._vad_buffer = self._vad_buffer[num_required_bytes:]

        confidence = self.voice_confidence(audio_frames)
        speaking = confidence >= self._vad_confidence

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

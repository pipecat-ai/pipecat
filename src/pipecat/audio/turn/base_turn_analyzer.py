#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class EndOfTurnState(Enum):
    COMPLETE = 1
    INCOMPLETE = 2


class BaseEndOfTurnAnalyzer(ABC):
    def __init__(self, *, sample_rate: Optional[int] = None):
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._chunk_size_ms = 0

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

    @abstractmethod
    def append_audio(self, buffer: bytes, is_speech: bool):
        pass

    @abstractmethod
    def analyze_end_of_turn(self) -> EndOfTurnState:
        pass

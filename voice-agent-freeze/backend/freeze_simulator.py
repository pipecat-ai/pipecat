"""Simulate a mid-call bot freeze by dropping TTS audio after N LLM turns."""

from __future__ import annotations

import time

from loguru import logger
from pipecat.frames.frames import (Frame, LLMFullResponseEndFrame,
                                   TTSAudioRawFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FreezeSimulator(FrameProcessor):
    """Drop ``TTSAudioRawFrame`` after ``freeze_after_turn`` completed LLM responses."""

    def __init__(self, freeze_after_turn: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_after_turn = freeze_after_turn
        self._turn_count = 0
        self._frozen = False

        self.freeze_start_ts: float | None = None
        self.freeze_end_ts: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # count completed LLM turns
        if isinstance(frame, LLMFullResponseEndFrame):
            self._turn_count += 1
            if self._turn_count >= self._freeze_after_turn and not self._frozen:
                self._frozen = True
                self.freeze_start_ts = time.time()
                logger.info("FREEZE TRIGGERED after turn {}", self._turn_count)

        # Drop audio frames during freeze
        if isinstance(frame, TTSAudioRawFrame) and self._frozen:
            return  # drop all audio frames during freeze window

        await self.push_frame(frame, direction)

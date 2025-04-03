#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import STTService


class TranscriptionLogObserver(BaseObserver):
    """Observer to log transcription activity to the console.

    Logs all frame instances (only from STT service) of:

    - TranscriptionFrame
    - InterimTranscriptionFrame

    This allows you to track when the LLM starts responding, what it generates,
    and when it finishes.

    """

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        if not isinstance(src, STTService):
            return

        time_sec = timestamp / 1_000_000_000

        arrow = "→"

        if isinstance(frame, TranscriptionFrame):
            logger.debug(
                f"💬 {src} {arrow} TRANSCRIPTION: {frame.text!r} from {frame.user_id!r} at {time_sec:.2f}s"
            )
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug(
                f"💬 {src} {arrow} INTERIM TRANSCRIPTION: {frame.text!r} from {frame.user_id!r} at {time_sec:.2f}s"
            )

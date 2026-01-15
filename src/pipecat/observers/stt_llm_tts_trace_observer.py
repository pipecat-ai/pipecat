#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM logging observer for Pipecat."""

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_output import BaseOutputTransport


class STTLLMTTSTraceObserver(BaseObserver):
    """Observer to basic STT, LLM, & TTS activity to the console.

    Logs all frame instances of:

    - UserStartedSpeakingFrame
    - TranscriptionFrame

    - LLMFullResponseStartFrame
    - LLMFullResponseEndFrame

    - TTSStartedFrame
    - TTSStoppedFrame
    - BotStartedSpeakingFrame
    - BotStoppedSpeakingFrame
    """

    def __init__(self):
        """Initialize frame start times to calculate span times."""
        super().__init__()
        self._last_user_started_speaking_frame_time = 0
        self._last_transcription_frame_time = 0

        self._last_llm_response_start_frame_time = 0

        self._last_tts_started_frame_time = 0
        self._last_tts_stopped_frame_time = 0
        self._last_bot_started_speaking_frame_time = 0

        self._arrow = "→"

    async def on_push_frame(self, data: FramePushed):
        """Handle frame push events and log STT, LLM, & TTS activities.

        Args:
            data: The frame push event data containing source, destination,
                  frame, direction, and timestamp information.
        """
        src = data.source
        dst = data.destination
        frame = data.frame
        direction = data.direction
        timestamp = data.timestamp

        time_sec = timestamp / 1_000_000_000

        if isinstance(src, BaseOutputTransport):
            # Trace STT
            if isinstance(frame, UserStartedSpeakingFrame):
                self._handle_UserStartedSpeakingFrame(src, dst, frame, time_sec)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._handle_UserStoppedSpeakingFrame(src, dst, frame, time_sec)
            # TTS
            if isinstance(dst, TTSService):
                if isinstance(frame, BotStartedSpeakingFrame):
                    self._handle_BotStartedSpeakingFrame(src, dst, frame, time_sec)
                elif isinstance(frame, BotStoppedSpeakingFrame):
                    self._handle_BotStoppedSpeakingFrame(src, dst, frame, time_sec)

        # STT
        elif isinstance(src, STTService):
            if isinstance(frame, TranscriptionFrame):
                self._handle_TranscriptionFrame(src, dst, frame, time_sec)

        # Trace LLM
        elif isinstance(src, LLMService):
            if isinstance(frame, LLMFullResponseStartFrame):
                self._handle_LLMFullResponseStartFrame(src, dst, frame, time_sec)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._handle_LLMFullResponseEndFrame(src, dst, frame, time_sec)

        # Trace TTS
        elif isinstance(src, TTSService):
            if isinstance(frame, TTSStartedFrame):
                self._handle_TTSStartedFrame(src, dst, frame, time_sec)
            elif isinstance(frame, TTSStoppedFrame):
                self._handle_TTSStoppedFrame(src, dst, frame, time_sec)

    # STT frame handlers
    def _handle_UserStartedSpeakingFrame(self, src, dst, frame, time_sec):
        self._last_user_started_speaking_frame_time = time_sec
        logger.debug(f"🙂🟢 UserStartedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_UserStoppedSpeakingFrame(self, src, dst, frame, time_sec):
        logger.debug(f"🙂🔴 UserStoppedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_TranscriptionFrame(self, src, dst, frame, time_sec):
        self._last_transcription_frame_time = time_sec
        logger.debug(f"🙂📝 TranscriptionFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

        if 0 != self._last_user_started_speaking_frame_time:
            stt_generation_time = time_sec - self._last_user_started_speaking_frame_time
            self._last_user_started_speaking_frame_time = 0
            logger.info(f"📝⏰ STT span: {stt_generation_time:.4f}s")

    # LLM frame handlers
    def _handle_LLMFullResponseStartFrame(self, src, dst, frame, time_sec):
        self._last_llm_response_start_frame_time = time_sec
        logger.debug(
            f"🧠🟢 LLMFullResponseStartFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s"
        )

    def _handle_LLMFullResponseEndFrame(self, src, dst, frame, time_sec):
        logger.debug(f"🧠🔴 LLMFullResponseEndFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")
        llm_time = time_sec - self._last_llm_response_start_frame_time
        logger.info(f"🧠⏰ LLM span: {llm_time:.4f}s")

    # TTS frame handlers
    def _handle_TTSStartedFrame(self, src, dst, frame, time_sec):
        self._last_tts_started_frame_time = time_sec
        logger.debug(f"📢🟢 TTSStartedFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_TTSStoppedFrame(self, src, dst, frame, time_sec):
        self._last_tts_stopped_frame_time = time_sec
        tts_time = time_sec - self._last_tts_started_frame_time
        logger.debug(f"📢🔴 TTSStoppedFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")
        logger.info(f"📢⏰ TTS generation span: {tts_time:.4f}s")

    def _handle_BotStartedSpeakingFrame(self, src, dst, frame, time_sec):
        self._last_bot_started_speaking_frame_time = time_sec
        tts_time = time_sec - self._last_tts_started_frame_time
        logger.debug(f"🤖🟢 BotStartedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")
        logger.info(f"📢⏰ TTS to first speech span: {tts_time:.4f}s")

    def _handle_BotStoppedSpeakingFrame(self, src, dst, frame, time_sec):
        logger.debug(f"🤖🔴 BotStoppedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

import time
from typing import Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotTextCollectorProcessor(FrameProcessor):
    """Collects bot/LLM text output for the transcript UI.
    
    Captures:
    - LLM text chunks (LLMTextFrame)
    - Bot speaking events for timing
    """
    
    def __init__(
        self,
        on_bot_transcript: Optional[Callable[[str, float], None]] = None,
        on_latency: Optional[Callable[[float, float], None]] = None,
        on_transcript_end: Optional[Callable[[float], None]] = None,
        **kwargs
    ):
        """Initialize the bot text collector.
        
        Args:
            on_bot_transcript: Callback(text, start_time) for bot speech.
            on_latency: Callback(user_stop_time, bot_start_time) for latency.
            on_transcript_end: Callback(end_time) when transcript ends.
        """
        super().__init__(**kwargs)
        self._on_bot_transcript = on_bot_transcript
        self._on_latency = on_latency
        self._on_transcript_end = on_transcript_end
        
        self._current_bot_text = ""
        self._bot_start_time: Optional[float] = None
        self._user_stopped_time: Optional[float] = None
    
    def set_user_stopped_time(self, stop_time: float):
        """Set when the user stopped speaking (for latency calculation)."""
        self._user_stopped_time = stop_time
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to collect bot text and timing."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMTextFrame):
            if frame.text:
                self._current_bot_text += frame.text
                logger.debug(f"Got LLM text chunk: '{frame.text}'")
        
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_start_time = time.time()
            
            if self._user_stopped_time and self._on_latency:
                self._on_latency(self._user_stopped_time, self._bot_start_time)
                self._user_stopped_time = None
        
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._current_bot_text and self._on_bot_transcript:
                self._on_bot_transcript(
                    self._current_bot_text.strip(),
                    self._bot_start_time or time.time()
                )
            
            if self._on_transcript_end:
                self._on_transcript_end(time.time())
            
            self._current_bot_text = ""
            self._bot_start_time = None
        
        await self.push_frame(frame, direction)

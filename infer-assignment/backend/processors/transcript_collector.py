import time
from typing import Callable, Optional, TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from processors.bot_text_collector import BotTextCollectorProcessor


class TranscriptCollectorProcessor(FrameProcessor):
    """Collects user transcripts and timing information for the recording UI.
    
    Captures:
    - User transcriptions with timestamps (from STT)
    - User speech start/stop events (from VAD)
    
    Note: Bot transcripts are handled by BotTextCollectorProcessor.
    """
    
    def __init__(
        self,
        on_user_transcript: Optional[Callable[[str, float], None]] = None,
        on_transcript_end: Optional[Callable[[float], None]] = None,
        bot_collector: Optional["BotTextCollectorProcessor"] = None,
        **kwargs
    ):
        """Initialize the user transcript collector.
        
        Args:
            on_user_transcript: Callback(text, start_time) for user speech.
            on_transcript_end: Callback(end_time) when a transcript segment ends.
            bot_collector: Reference to bot collector for latency coordination.
        """
        super().__init__(**kwargs)
        self._on_user_transcript = on_user_transcript
        self._on_transcript_end = on_transcript_end
        self._bot_collector = bot_collector
        
        # State tracking
        self._user_speaking = False
        self._user_stopped_time: Optional[float] = None
        self._current_user_text = ""
        self._user_start_time: Optional[float] = None
        self._transcript_emitted = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to collect user transcripts and timing."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._user_start_time = time.time()
            self._current_user_text = ""
            self._transcript_emitted = False
            logger.debug("User started speaking")
        
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._user_stopped_time = time.time()
            
            if self._on_transcript_end:
                self._on_transcript_end(self._user_stopped_time)
            
            if self._bot_collector:
                self._bot_collector.set_user_stopped_time(self._user_stopped_time)
            
            logger.debug("User stopped speaking")
        
        elif isinstance(frame, TranscriptionFrame):
            if frame.text:
                self._current_user_text = frame.text
                
                if self._on_user_transcript and not getattr(self, '_transcript_emitted', False):
                    self._on_user_transcript(
                        self._current_user_text.strip(),
                        self._user_start_time or time.time()
                    )
                    self._transcript_emitted = True
        
        await self.push_frame(frame, direction)

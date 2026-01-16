import asyncio
import time
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FreezeSimulatorProcessor(FrameProcessor):
    """Simulates a bot freeze by blocking TTS audio frames.
    
    When triggered, this processor will stop passing audio frames for a
    specified duration, simulating a freeze condition that can be detected
    and visualized in the recording UI.
    
    The freeze has two modes:
    - If triggered while audio is flowing: starts immediately
    - If triggered during silence: waits for audio to start, then freezes
    """
    
    def __init__(
        self,
        freeze_duration_seconds: float = 3.0,
        freeze_callback=None,
        **kwargs
    ):
        """Initialize the freeze simulator.
        
        Args:
            freeze_duration_seconds: How long to freeze when triggered.
            freeze_callback: Callback function(start_time, end_time) when freeze ends.
        """
        super().__init__(**kwargs)
        self._freeze_duration = freeze_duration_seconds
        self._freeze_callback = freeze_callback
        
        # Freeze state
        self._freeze_pending = False  # Waiting for audio to start
        self._freeze_active = False   # Actually blocking audio
        self._freeze_start_time: Optional[float] = None
        self._freeze_end_time: Optional[float] = None
        self._unfreeze_task: Optional[asyncio.Task] = None
        
        # Track if we're in a TTS sequence
        self._in_tts_sequence = False
        self._frames_blocked = 0
    
    @property
    def is_frozen(self) -> bool:
        """Check if freeze is currently active or pending."""
        return self._freeze_active or self._freeze_pending
    
    async def trigger_freeze(self):
        """Trigger a freeze for the configured duration.
        
        Records the freeze start time immediately when triggered.
        If no audio is currently flowing, waits until audio starts before blocking frames.
        """
        if self._freeze_active or self._freeze_pending:
            return
        
        # Always record the start time when the freeze is triggered
        self._freeze_start_time = time.time()
        
        if self._in_tts_sequence:
            await self._start_freeze()
        else:
            self._freeze_pending = True
    
    async def _start_freeze(self):
        """Actually start the freeze (called when audio is available)."""
        self._freeze_pending = False
        self._freeze_active = True
        self._frames_blocked = 0
        
        # Schedule unfreeze
        self._unfreeze_task = asyncio.create_task(self._unfreeze_after_duration())
    
    async def _unfreeze_after_duration(self):
        """Automatically unfreeze after the configured duration."""
        await asyncio.sleep(self._freeze_duration)
        await self._end_freeze()
    
    async def _end_freeze(self):
        """End the freeze and record the event."""
        if not self._freeze_active:
            return
        
        # Calculate freeze end time based on start time + duration
        # This ensures the recorded duration matches the configured duration
        self._freeze_end_time = self._freeze_start_time + self._freeze_duration if self._freeze_start_time else time.time()
        self._freeze_active = False
        
        if self._freeze_callback and self._freeze_start_time and self._freeze_end_time:
            self._freeze_callback(self._freeze_start_time, self._freeze_end_time)
        
        self._freeze_start_time = None
        self._freeze_end_time = None
        self._frames_blocked = 0
        self._unfreeze_task = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, blocking TTS audio during freeze."""
        await super().process_frame(frame, direction)
        
        # Track TTS sequence
        if isinstance(frame, TTSStartedFrame):
            logger.debug(f"TTSStartedFrame received, freeze_pending={self._freeze_pending}, freeze_active={self._freeze_active}")
            self._in_tts_sequence = True
            if self._freeze_pending:
                await self._start_freeze()
        elif isinstance(frame, TTSStoppedFrame):
            self._in_tts_sequence = False
        
        # Block TTS audio frames during freeze
        if isinstance(frame, TTSAudioRawFrame):
            if self._freeze_active:
                self._frames_blocked += 1
                return
            elif self._freeze_pending:
                await self._start_freeze()
                self._frames_blocked += 1
                return
        
        await self.push_frame(frame, direction)

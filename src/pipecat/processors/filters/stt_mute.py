from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    STTMuteFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import STTService


class STTMuteStrategy(Enum):
    NEVER = "never"  # Never mute
    FIRST_SPEECH = "first_speech"  # Mute only during first bot speech
    ALWAYS = "always"  # Mute during all bot speech
    CUSTOM = "custom"  # Allow custom logic via callback


@dataclass
class STTMuteConfig:
    """Configuration for STTMuteProcessor"""

    strategy: STTMuteStrategy = STTMuteStrategy.NEVER
    # Optional callback for custom muting logic
    should_mute_callback: Optional[Callable[["STTMuteProcessor"], bool]] = None


class STTMuteProcessor(FrameProcessor):
    """A general-purpose processor that handles STT muting and interruption control.

    This processor combines the concepts of STT muting and interruption control,
    treating them as a single coordinated feature. When STT is muted, interruptions
    are automatically disabled.
    """

    def __init__(self, stt_service: STTService, config: STTMuteConfig = STTMuteConfig(), **kwargs):
        super().__init__(**kwargs)
        self._stt_service = stt_service
        self._config = config
        self._first_speech_handled = False
        self._bot_is_speaking = False

    @property
    def is_muted(self) -> bool:
        """Returns whether STT is currently muted."""
        return self._stt_service.is_muted

    async def _handle_mute_state(self, should_mute: bool):
        """Handles both STT muting and interruption control."""
        if should_mute != self.is_muted:
            logger.info(f"STT {'muting' if should_mute else 'unmuting'}")
            await self.push_frame(STTMuteFrame(muted=should_mute))

    def _should_mute(self) -> bool:
        """Determines if STT should be muted based on current state and strategy."""
        if not self._bot_is_speaking:
            return False

        if self._config.strategy == STTMuteStrategy.ALWAYS:
            return True
        elif (
            self._config.strategy == STTMuteStrategy.FIRST_SPEECH and not self._first_speech_handled
        ):
            self._first_speech_handled = True
            return True
        elif self._config.strategy == STTMuteStrategy.CUSTOM and self._config.should_mute_callback:
            return self._config.should_mute_callback(self)

        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Handle bot speaking state changes
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_is_speaking = True
            await self._handle_mute_state(self._should_mute())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_speaking = False
            await self._handle_mute_state(self._should_mute())

        # Handle frame propagation
        if isinstance(frame, (StartInterruptionFrame, StopInterruptionFrame)):
            # Only pass interruption frames when not muted
            if not self.is_muted:
                await self.push_frame(frame, direction)
            else:
                logger.debug("Interruption frame suppressed - STT currently muted")
        else:
            # Pass all other frames through
            await self.push_frame(frame, direction)

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speech-to-text (STT) muting control module.

This module provides functionality to control STT muting based on different strategies,
such as during function calls, bot speech, or custom conditions. It helps manage when
the STT service should be active or inactive during a conversation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    STTMuteFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import STTService


class STTMuteStrategy(Enum):
    """Strategies determining when STT should be muted.

    Attributes:
        FIRST_SPEECH: Mute only during first bot speech
        FUNCTION_CALL: Mute during function calls
        ALWAYS: Mute during all bot speech
        CUSTOM: Allow custom logic via callback
    """

    FIRST_SPEECH = "first_speech"  # Mute only during first bot speech
    FUNCTION_CALL = "function_call"  # Mute during function calls
    ALWAYS = "always"  # Mute during all bot speech
    CUSTOM = "custom"  # Allow custom logic via callback


@dataclass
class STTMuteConfig:
    """Configuration for STT muting behavior.

    Args:
        strategies: Set of muting strategies to apply
        should_mute_callback: Optional callback for custom muting logic.
            Only required when using STTMuteStrategy.CUSTOM
    """

    strategies: set[STTMuteStrategy]
    # Optional callback for custom muting logic
    should_mute_callback: Optional[Callable[["STTMuteFilter"], Awaitable[bool]]] = None


class STTMuteFilter(FrameProcessor):
    """A processor that handles STT muting and interruption control.

    This processor combines STT muting and interruption control as a coordinated
    feature. When STT is muted, interruptions are automatically disabled.

    Args:
        stt_service: Service handling speech-to-text functionality
        config: Configuration specifying muting strategies
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(self, stt_service: STTService, config: STTMuteConfig, **kwargs):
        super().__init__(**kwargs)
        self._stt_service = stt_service
        self._config = config
        self._first_speech_handled = False
        self._bot_is_speaking = False
        self._function_call_in_progress = False

    @property
    def is_muted(self) -> bool:
        """Returns whether STT is currently muted."""
        return self._stt_service.is_muted

    async def _handle_mute_state(self, should_mute: bool):
        """Handles both STT muting and interruption control."""
        if should_mute != self.is_muted:
            logger.debug(f"STT {'muting' if should_mute else 'unmuting'}")
            await self.push_frame(STTMuteFrame(mute=should_mute))

    async def _should_mute(self) -> bool:
        """Determines if STT should be muted based on current state and strategy."""
        for strategy in self._config.strategies:
            match strategy:
                case STTMuteStrategy.FUNCTION_CALL:
                    if self._function_call_in_progress:
                        return True

                case STTMuteStrategy.ALWAYS:
                    if self._bot_is_speaking:
                        return True

                case STTMuteStrategy.FIRST_SPEECH:
                    if self._bot_is_speaking and not self._first_speech_handled:
                        self._first_speech_handled = True
                        return True

                case STTMuteStrategy.CUSTOM:
                    if self._bot_is_speaking and self._config.should_mute_callback:
                        should_mute = await self._config.should_mute_callback(self)
                        if should_mute:
                            return True

        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames and manages muting state."""
        # Handle function call state changes
        if isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = True
            await self._handle_mute_state(await self._should_mute())
        elif isinstance(frame, FunctionCallResultFrame):
            self._function_call_in_progress = False
            await self._handle_mute_state(await self._should_mute())
        # Handle bot speaking state changes
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_is_speaking = True
            await self._handle_mute_state(await self._should_mute())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_speaking = False
            await self._handle_mute_state(await self._should_mute())

        # Handle frame propagation
        if isinstance(
            frame,
            (
                StartInterruptionFrame,
                StopInterruptionFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
            ),
        ):
            # Only pass VAD-related frames when not muted
            if not self.is_muted:
                await self.push_frame(frame, direction)
            else:
                logger.debug(f"{frame.__class__.__name__} suppressed - STT currently muted")
        else:
            # Pass all other frames through
            await self.push_frame(frame, direction)

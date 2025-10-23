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
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    StartFrame,
    STTMuteFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class STTMuteStrategy(Enum):
    """Strategies determining when STT should be muted.

    Each strategy defines different conditions under which speech-to-text
    processing should be temporarily disabled to prevent unwanted audio
    processing during specific conversation states.

    Parameters:
        FIRST_SPEECH: Mute STT until the first bot speech is detected.
        MUTE_UNTIL_FIRST_BOT_COMPLETE: Mute STT until the first bot completes speaking,
        regardless of whether it is the first speech.
        FUNCTION_CALL: Mute STT during function calls to prevent interruptions.
        ALWAYS: Always mute STT when the bot is speaking.
        CUSTOM: Use a custom callback to determine muting logic dynamically.
    """

    FIRST_SPEECH = "first_speech"
    MUTE_UNTIL_FIRST_BOT_COMPLETE = "mute_until_first_bot_complete"
    FUNCTION_CALL = "function_call"
    ALWAYS = "always"
    CUSTOM = "custom"


@dataclass
class STTMuteConfig:
    """Configuration for STT muting behavior.

    Defines which muting strategies to apply and provides optional custom
    callback for advanced muting logic. Multiple strategies can be combined
    to create sophisticated muting behavior.

    Parameters:
        strategies: Set of muting strategies to apply simultaneously.
        should_mute_callback: Optional callback for custom muting logic.
            Only required when using STTMuteStrategy.CUSTOM. Called with
            the STTMuteFilter instance to determine muting state.

    Note:
        MUTE_UNTIL_FIRST_BOT_COMPLETE and FIRST_SPEECH strategies should not be used together
        as they handle the first bot speech differently.
    """

    strategies: set[STTMuteStrategy]
    should_mute_callback: Optional[Callable[["STTMuteFilter"], Awaitable[bool]]] = None

    def __post_init__(self):
        """Validate configuration after initialization.

        Raises:
            ValueError: If incompatible strategies are used together.
        """
        if (
            STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE in self.strategies
            and STTMuteStrategy.FIRST_SPEECH in self.strategies
        ):
            raise ValueError(
                "MUTE_UNTIL_FIRST_BOT_COMPLETE and FIRST_SPEECH strategies should not be used together"
            )


class STTMuteFilter(FrameProcessor):
    """A processor that handles STT muting and interruption control.

    This processor combines STT muting and interruption control as a coordinated
    feature. When STT is muted, interruptions are automatically disabled by
    suppressing VAD-related frames. This prevents unwanted speech detection
    during bot speech, function calls, or other specified conditions.
    """

    def __init__(self, *, config: STTMuteConfig, **kwargs):
        """Initialize the STT mute filter.

        Args:
            config: Configuration specifying muting strategies and behavior.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._config = config
        self._first_speech_handled = False
        self._bot_is_speaking = False
        self._function_call_in_progress = False
        self._is_muted = False  # Initialize as unmuted, will set state on StartFrame if needed

    @property
    def is_muted(self) -> bool:
        """Check if STT is currently muted.

        Returns:
            True if STT is currently muted and audio frames are being suppressed.
        """
        return self._is_muted

    async def _handle_mute_state(self, should_mute: bool):
        """Handle STT muting and interruption control state changes."""
        if should_mute != self.is_muted:
            logger.debug(f"STTMuteFilter {'muting' if should_mute else 'unmuting'}")
            self._is_muted = should_mute
            await self.push_frame(STTMuteFrame(mute=should_mute), FrameDirection.UPSTREAM)
            await self.push_frame(STTMuteFrame(mute=should_mute), FrameDirection.DOWNSTREAM)

    async def _should_mute(self) -> bool:
        """Determine if STT should be muted based on current state and strategies."""
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

                case STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE:
                    if not self._first_speech_handled:
                        return True

                case STTMuteStrategy.CUSTOM:
                    if self._bot_is_speaking and self._config.should_mute_callback:
                        should_mute = await self._config.should_mute_callback(self)
                        if should_mute:
                            return True

        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage muting state.

        Monitors conversation state through frame types and applies muting
        strategies accordingly. Suppresses VAD-related frames when muted
        while allowing other frames to pass through.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Determine if we need to change mute state based on frame type
        should_mute = None

        # Process frames to determine mute state
        if isinstance(frame, StartFrame):
            should_mute = await self._should_mute()
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = True
            should_mute = await self._should_mute()
        elif isinstance(frame, FunctionCallResultFrame):
            self._function_call_in_progress = False
            should_mute = await self._should_mute()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_is_speaking = True
            should_mute = await self._should_mute()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_speaking = False
            if not self._first_speech_handled:
                self._first_speech_handled = True
            should_mute = await self._should_mute()

        # Then push the original frame
        if isinstance(
            frame,
            (
                InterruptionFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
                InputAudioRawFrame,
                InterimTranscriptionFrame,
                TranscriptionFrame,
            ),
        ):
            # Only pass VAD-related frames when not muted
            if not self.is_muted:
                await self.push_frame(frame, direction)
            else:
                logger.trace(f"{frame.__class__.__name__} suppressed - STT currently muted")
        else:
            # Pass all other frames through
            await self.push_frame(frame, direction)

        # Finally handle mute state change if needed
        if should_mute is not None and should_mute != self.is_muted:
            await self._handle_mute_state(should_mute)

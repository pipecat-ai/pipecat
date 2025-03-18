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
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.stt_mute_filter import STTMuteFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import detect_voice_mail


class LLMMuteStrategy(Enum):
    """Strategies determining when LLM should be muted.

    Attributes:
        FIRST_SPEECH: Mute only during first detected bot speech
        MUTE_UNTIL_FIRST_BOT_COMPLETE: Start muted and remain muted until first bot speech completes
        FUNCTION_CALL: Mute during function calls
        ALWAYS: Mute during all bot speech
        CUSTOM: Allow custom logic via callback
    """

    FIRST_SPEECH = "first_speech"
    MUTE_UNTIL_FIRST_BOT_COMPLETE = "mute_until_first_bot_complete"
    FUNCTION_CALL = "function_call"
    ALWAYS = "always"
    CUSTOM = "custom"


@dataclass
class LLMMuteConfig:
    """Configuration for LLM muting behavior.

    Args:
        strategies: Set of muting strategies to apply
        should_mute_callback: Optional callback for custom muting logic.
            Only required when using LLMMuteStrategy.CUSTOM

    Note:
        MUTE_UNTIL_FIRST_BOT_COMPLETE and FIRST_SPEECH strategies should not be used together
        as they handle the first bot speech differently.
    """

    strategies: set[LLMMuteStrategy]
    # Optional callback for custom muting logic
    should_mute_callback: Optional[Callable[["LLMMuteFilter"], Awaitable[bool]]] = None

    def __post_init__(self):
        if (
            LLMMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE in self.strategies
            and LLMMuteStrategy.FIRST_SPEECH in self.strategies
        ):
            raise ValueError(
                "MUTE_UNTIL_FIRST_BOT_COMPLETE and FIRST_SPEECH strategies should not be used together"
            )


class LLMMuteFilter(FrameProcessor):
    def __init__(self, config: LLMMuteConfig, **kwargs):
        super().__init__(**kwargs)
        self._is_muted = False
        self._user_tx = []
        self._config = config
        self._first_speech_handled = False
        self._bot_is_speaking = False
        self._function_call_in_progress = False
        self._is_muted = False

    @property
    def is_muted(self) -> bool:
        """Returns whether LLM generation is currently muted."""
        return self._is_muted

    async def _handle_mute_state(self, should_mute: bool):
        """Handles both LLM generation mute and interruption control."""
        self._is_muted = should_mute

    # helper function to add user tx to user_words
    async def _add_user_tx(self, frame: TranscriptionFrame):
        self._user_tx.append(frame.text)

        logger.info(f"User words: {self._user_tx=}")

    @staticmethod
    def _detect_voice_mail(frame: Frame) -> bool:
        """Detect voice mail from TranscriptionFrame"""
        if not isinstance(frame, TranscriptionFrame):
            return False
        voice_mail_detected = detect_voice_mail(frame.text)

        logger.info(f"Voice mail detected: {voice_mail_detected=} for {frame=}")
        return voice_mail_detected

    async def _should_mute(self) -> bool:
        """Determines if LLM should be muted based on current state and strategy."""
        for strategy in self._config.strategies:
            match strategy:
                case LLMMuteStrategy.FUNCTION_CALL:
                    if self._function_call_in_progress:
                        return True

                case LLMMuteStrategy.ALWAYS:
                    if self._bot_is_speaking:
                        return True

                case LLMMuteStrategy.FIRST_SPEECH:
                    if self._bot_is_speaking and not self._first_speech_handled:
                        self._first_speech_handled = True
                        return True

                case LLMMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE:
                    if not self._first_speech_handled:
                        return True

                case LLMMuteStrategy.CUSTOM:
                    if self._bot_is_speaking and self._config.should_mute_callback:
                        should_mute = await self._config.should_mute_callback(self)
                        if should_mute:
                            return True

        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames and manages muting state."""
        await super().process_frame(frame, direction)

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
                TranscriptionFrame,
                InterimTranscriptionFrame,
            ),
        ):
            # Only pass VAD-related frames when not muted
            if not self.is_muted or self._detect_voice_mail(frame):
                if isinstance(frame, TranscriptionFrame):
                    await self._add_user_tx(frame)
                await self.push_frame(frame, direction)
            else:
                logger.debug(
                    f"LLMMuteFilter: {frame.__class__.__name__} suppressed - STT currently muted"
                )
        else:
            # Pass all other frames through
            await self.push_frame(frame, direction)

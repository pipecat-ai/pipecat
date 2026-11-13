#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared behavior of the Haru full-duplex speech-to-speech services."""

import asyncio
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    SpeechOutputAudioRawFrame,
    TextType,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

DEFAULT_MODEL = "higurashi-ai/Haru-4B"

#: Sample rate of the model's audio, in and out.
HARU_SAMPLE_RATE = 24000

#: Samples in one model frame: 133 ms. The model takes and produces whole frames.
HARU_FRAME_SAMPLES = 3200

_FRAME_BYTES = HARU_FRAME_SAMPLES * 2

# Quiet time that ends the model's turn. Text arrives on 133 ms frame
# boundaries, so this has to clear ordinary gaps within speech.
TURN_GAP_SECS = 0.8

# Session options that both the in-process engines and the server take.
_SESSION_OPTIONS = (
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "guidance_scale",
    "eop_bias",
    "speak_first",
)


@dataclass
class HaruLLMSettings(LLMSettings):
    """Settings for the Haru services.

    The model fixes all of them when a conversation starts. ``None`` leaves
    an option at the model's own default.

    Parameters:
        guidance_scale: Audio guidance, from 1 to 2. Higher follows the voice
            reference more closely.
        eop_bias: Added to the logit of the model's "start speaking" token.
            Above 0 the model starts speaking sooner.
        speak_first: Whether the model opens the conversation.
    """

    guidance_scale: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eop_bias: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speak_first: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _TurnGrouper:
    """The model's in-progress turn, assembled from per-frame text.

    Parameters:
        gap_secs: Quiet time that ends the turn.
        open: Whether a turn is currently open.
        timer: The task that closes the turn once ``gap_secs`` elapses.
        lock: Held while the turn opens or closes. The timer runs in its own
            task, so this is what keeps one turn's end frames from landing
            among the next turn's start frames.
    """

    gap_secs: float
    open: bool = False
    timer: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class HaruBaseLLMService(LLMService):
    """Base class for the Haru full-duplex speech-to-speech services.

    Haru listens and speaks at the same time. Every 133 ms it takes one frame
    of the user's audio and produces one frame of its own audio, together
    with the text of what it is saying. The model decides on its own when to
    speak, when to backchannel and when to yield, so there is no client-side
    turn detection and nothing triggers a response: the pipeline streams
    audio in and plays audio out.

    This class regroups input audio into model frames, turns the model's
    frames into pipeline frames and assembles its text into assistant turns.
    Subclasses provide the model: :meth:`_open_session` starts a conversation
    and :meth:`_send_user_frame` feeds it.

    The conversation starts on the first ``LLMContextFrame`` (typically queued
    as an ``LLMRunFrame``). ``Settings.system_instruction`` is the model's
    character card. The model takes no conversation history and no tools, so
    the context's messages are not sent to it.

    No ``InterruptionFrame`` is broadcast: the model handles being talked
    over itself. It produces no transcript of the user either, so the user
    aggregator records user turns only if another processor transcribes them.
    The assistant aggregator records the model's side from its text.

    Event handlers available:

    - on_session_started: Called once the model has run its prompt and is
      listening.
    """

    Settings = HaruLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        voice_reference: str | os.PathLike | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Haru service.

        Args:
            voice_reference: Path of an audio clip of the voice the model
                should speak with: one speaker, 3 seconds to 3 minutes of
                clean speech. Haru ships without voices.
            settings: Service settings. All of them are fixed once the
                conversation has started.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        default_settings = self.Settings(
            model=DEFAULT_MODEL,
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            guidance_scale=None,
            eop_bias=None,
            speak_first=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._voice_reference = voice_reference

        self._session_requested = False
        self._session_started = False

        self._resampler = create_stream_resampler()
        self._input_buffer = bytearray()
        self._warned_audio_dropped = False

        self._assistant_turn = _TurnGrouper(gap_secs=TURN_GAP_SECS)

        self._register_event_handler("on_session_started")

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Recommend turn strategies that never interrupt the model.

        The model produces no user transcript and proposes no user turns, so
        the strategies are external ones that another processor may drive.
        """
        return LLMServiceMetadataFrame(
            service_name=self.name,
            user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False),
        )

    #
    # lifecycle
    #

    async def stop(self, frame: EndFrame):
        """Close the model's open turn and end the conversation.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._close_assistant_turn()
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """End the conversation immediately.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._close_session()

    async def _update_settings(self, delta):
        """Store a settings delta.

        The model fixes its settings when a conversation starts, so a change
        is not applied to the running one.
        """
        changed = await super()._update_settings(delta)
        self._warn_unhandled_updated_settings(changed)
        return changed

    #
    # frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)

        await self.push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        """Start the conversation from the first context.

        Later contexts carry what the aggregators recorded from this
        conversation, which the model already has.
        """
        if self._session_requested:
            return
        self._session_requested = True
        instructions = assert_given(self._settings.system_instruction) or ""
        logger.debug(f"{self}: starting conversation")
        await self._open_session(instructions)

    def _session_options(self) -> dict[str, Any]:
        """The session options that are set, by the name the model gives them."""
        options = {name: getattr(self._settings, name) for name in _SESSION_OPTIONS}
        return {name: value for name, value in options.items() if value is not None}

    async def _session_ready(self):
        """Mark the conversation started. Called by subclasses once the model is listening."""
        self._session_started = True
        logger.info(
            f"{self}: conversation started. The model handles being interrupted by itself; "
            "no InterruptionFrame is broadcast."
        )
        await self._call_event_handler("on_session_started")

    @abstractmethod
    async def _open_session(self, instructions: str):
        """Start a conversation with the given character card.

        Must call :meth:`_session_ready` once the model is listening.

        Args:
            instructions: The model's character card.
        """
        ...

    @abstractmethod
    async def _send_user_frame(self, audio: np.ndarray):
        """Give the model one frame of the user's audio.

        Args:
            audio: ``HARU_FRAME_SAMPLES`` float32 samples at 24 kHz.
        """
        ...

    async def _close_session(self):
        """End the conversation and release what it holds."""
        self._session_started = False
        self._input_buffer.clear()

    #
    # audio
    #

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        if not self._session_started:
            if not self._warned_audio_dropped:
                self._warned_audio_dropped = True
                logger.debug(f"{self}: dropping input audio until the conversation has started")
            return
        audio = frame.audio
        if frame.sample_rate != HARU_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, frame.sample_rate, HARU_SAMPLE_RATE)
        self._input_buffer.extend(audio)
        while len(self._input_buffer) >= _FRAME_BYTES:
            pcm = bytes(self._input_buffer[:_FRAME_BYTES])
            del self._input_buffer[:_FRAME_BYTES]
            await self._send_user_frame(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)

    async def _push_model_audio(self, audio: bytes):
        """Push one frame of the model's audio.

        The model produces audio continuously at real-time pace, silence
        included, so this is a speech stream rather than TTS output: the
        output transport derives ``BotStartedSpeakingFrame`` and
        ``BotStoppedSpeakingFrame`` from the audio itself.

        Args:
            audio: 16-bit mono PCM at 24 kHz.
        """
        await self.push_frame(
            SpeechOutputAudioRawFrame(audio=audio, sample_rate=HARU_SAMPLE_RATE, num_channels=1)
        )

    #
    # assistant turns
    #

    async def _push_model_text(self, text: str):
        """Push the text of what the model is saying, opening a turn if none is open.

        Args:
            text: The new text of a frame. Carries its own leading spaces.
        """
        turn = self._assistant_turn
        # Whitespace alone says nothing: it can't open a turn, or the context
        # would record an empty assistant message.
        if not text or not (turn.open or text.strip()):
            return
        async with turn.lock:
            if not turn.open:
                turn.open = True
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TTSStartedFrame())

            # RTVI relies on LLMTextFrames for its "bot-llm-text" event, but the
            # assistant aggregator records the TTSTextFrame, so the LLMTextFrame
            # must not also be appended to the context.
            llm_text_frame = LLMTextFrame(text)
            llm_text_frame.append_to_context = False
            await self.push_frame(llm_text_frame)

            tts_text_frame = TTSTextFrame(text, text_type=TextType.SENTENCE)
            tts_text_frame.includes_inter_frame_spaces = True
            await self.push_frame(tts_text_frame)

            if turn.timer is not None:
                await self.cancel_task(turn.timer)
            turn.timer = self.create_task(self._close_turn_after_gap(), "turn-gap")

    async def _close_turn_after_gap(self):
        turn = self._assistant_turn
        await asyncio.sleep(turn.gap_secs)
        async with turn.lock:
            # Running inside the turn's own timer, so the timer must not cancel itself.
            turn.timer = None
            await self._end_assistant_turn()

    async def _close_assistant_turn(self):
        """Close the model's open turn, if any, so response frames stay balanced."""
        turn = self._assistant_turn
        async with turn.lock:
            if turn.timer is not None:
                await self.cancel_task(turn.timer)
                turn.timer = None
            await self._end_assistant_turn()

    async def _end_assistant_turn(self):
        """Emit the open turn's end frames, with the turn's lock held."""
        turn = self._assistant_turn
        if not turn.open:
            return
        turn.open = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to 16-bit little-endian PCM.

    Args:
        audio: Float samples.

    Returns:
        The samples as 16-bit PCM bytes.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2").tobytes()

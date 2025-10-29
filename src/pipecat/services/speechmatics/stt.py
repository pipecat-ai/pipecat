#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics STT service integration."""

import asyncio
import datetime
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from speechmatics.rt import (
        AsyncClient,
        AudioEncoding,
        AudioFormat,
        ClientMessageType,
        ConversationConfig,
        OperatingPoint,
        ServerMessageType,
        TranscriptionConfig,
        __version__,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Speechmatics, you need to `pip install pipecat-ai[speechmatics]`."
    )
    raise Exception(f"Missing module: {e}")


class EndOfUtteranceMode(str, Enum):
    """End of turn delay options for transcription."""

    NONE = "none"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class DiarizationFocusMode(str, Enum):
    """Speaker focus mode for diarization."""

    RETAIN = "retain"
    IGNORE = "ignore"


@dataclass
class AdditionalVocabEntry:
    """Additional vocabulary entry.

    Parameters:
        content: The word to add to the dictionary.
        sounds_like: Similar words to the word.
    """

    content: str
    sounds_like: list[str] = field(default_factory=list)


@dataclass
class DiarizationKnownSpeaker:
    """Known speakers for speaker diarization.

    Parameters:
        label: The label of the speaker.
        speaker_identifiers: One or more data strings for the speaker.
    """

    label: str
    speaker_identifiers: list[str]


@dataclass
class SpeechFragment:
    """Fragment of an utterance.

    Parameters:
        start_time: Start time of the fragment in seconds (from session start).
        end_time: End time of the fragment in seconds (from session start).
        language: Language of the fragment. Defaults to `Language.EN`.
        is_eos: Whether the fragment is the end of a sentence. Defaults to `False`.
        is_final: Whether the fragment is the final fragment. Defaults to `False`.
        is_disfluency: Whether the fragment is a disfluency. Defaults to `False`.
        is_punctuation: Whether the fragment is a punctuation. Defaults to `False`.
        attaches_to: Whether the fragment attaches to the previous or next fragment (punctuation). Defaults to empty string.
        content: Content of the fragment. Defaults to empty string.
        speaker: Speaker of the fragment (if diarization is enabled). Defaults to `None`.
        confidence: Confidence of the fragment (0.0 to 1.0). Defaults to `1.0`.
        result: Raw result of the fragment from the TTS.
    """

    start_time: float
    end_time: float
    language: Language = Language.EN
    is_eos: bool = False
    is_final: bool = False
    is_disfluency: bool = False
    is_punctuation: bool = False
    attaches_to: str = ""
    content: str = ""
    speaker: str | None = None
    confidence: float = 1.0
    result: Any | None = None


@dataclass
class SpeakerFragments:
    """SpeechFragment items grouped by speaker_id.

    Parameters:
        speaker_id: The ID of the speaker.
        is_active: Whether the speaker is active (emits frame).
        timestamp: The timestamp of the frame.
        language: The language of the frame.
        fragments: The list of SpeechFragment items.
    """

    speaker_id: str | None = None
    is_active: bool = False
    timestamp: str | None = None
    language: Language | None = None
    fragments: list[SpeechFragment] = field(default_factory=list)

    def __str__(self):
        """Return a string representation of the object."""
        return f"SpeakerFragments(speaker_id: {self.speaker_id}, timestamp: {self.timestamp}, language: {self.language}, text: {self._format_text()})"

    def _format_text(self, format: str | None = None) -> str:
        """Wrap text with speaker ID in an optional f-string format.

        Args:
            format: Format to wrap the text with.

        Returns:
            str: The wrapped text.
        """
        # Cumulative contents
        content = ""

        # Assemble the text
        for frag in self.fragments:
            if content == "" or frag.attaches_to == "previous":
                content += frag.content
            else:
                content += " " + frag.content

        # Format the text, if format is provided
        if format is None or self.speaker_id is None:
            return content
        return format.format(**{"speaker_id": self.speaker_id, "text": content})

    def _as_frame_attributes(
        self, active_format: str | None = None, passive_format: str | None = None
    ) -> dict[str, Any]:
        """Return a dictionary of attributes for a TranscriptionFrame.

        Args:
            active_format: Format to wrap the text with.
            passive_format: Format to wrap the text with. Defaults to `active_format`.

        Returns:
            dict[str, Any]: The dictionary of attributes.
        """
        if not passive_format:
            passive_format = active_format
        return {
            "text": self._format_text(active_format if self.is_active else passive_format),
            "user_id": self.speaker_id or "",
            "timestamp": self.timestamp,
            "language": self.language,
            "result": [frag.result for frag in self.fragments],
        }


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Speechmatics STT service.

        Parameters:
            operating_point: Operating point for transcription accuracy vs. latency tradeoff. It is
                recommended to use OperatingPoint.ENHANCED for most use cases. Defaults to
                OperatingPoint.ENHANCED.

            domain: Domain for Speechmatics API. Defaults to None.

            language: Language code for transcription. Defaults to `Language.EN`.

            output_locale: Output locale for transcription, e.g. `Language.EN_GB`.
                Defaults to None.

            enable_vad: Enable VAD to trigger end of utterance detection. This should be used
                without any other VAD enabled in the agent and will emit the speaker started
                and stopped frames. Defaults to False.

            enable_partials: Enable partial transcriptions. When enabled, the STT engine will
                emit partial word frames - useful for the visualisation of real-time transcription.
                Defaults to True.

            max_delay: Maximum delay in seconds for transcription. This forces the STT engine to
                speed up the processing of transcribed words and reduces the interval between partial
                and final results. Lower values can have an impact on accuracy. Defaults to 1.0.

            end_of_utterance_silence_trigger: Maximum delay in seconds for end of utterance trigger.
                The delay is used to wait for any further transcribed words before emitting the final
                word frames. The value must be lower than max_delay.
                Defaults to 0.5.

            end_of_utterance_mode: End of utterance delay mode. When ADAPTIVE is used, the delay
                can be adjusted on the content of what the most recent speaker has said, such as
                rate of speech and whether they have any pauses or disfluencies. When FIXED is used,
                the delay is fixed to the value of `end_of_utterance_delay`. Use of NONE disables
                end of utterance detection and uses a fallback timer.
                Defaults to `EndOfUtteranceMode.FIXED`.

            additional_vocab: List of additional vocabulary entries. If you supply a list of
                additional vocabulary entries, the this will increase the weight of the words in the
                vocabulary and help the STT engine to better transcribe the words.
                Defaults to [].

            punctuation_overrides: Punctuation overrides. This allows you to override the punctuation
                in the STT engine. This is useful for languages that use different punctuation
                than English. See documentation for more information.
                Defaults to None.

            enable_diarization: Enable speaker diarization. When enabled, the STT engine will
                determine and attribute words to unique speakers. The speaker_sensitivity
                parameter can be used to adjust the sensitivity of diarization.
                Defaults to False.

            speaker_sensitivity: Diarization sensitivity. A higher value increases the sensitivity
                of diarization and helps when two or more speakers have similar voices.
                Defaults to 0.5.

            max_speakers: Maximum number of speakers to detect. This forces the STT engine to cluster
                words into a fixed number of speakers. It should not be used to limit the number of
                speakers, unless it is clear that there will only be a known number of speakers.
                Defaults to None.

            speaker_active_format: Formatter for active speaker ID. This formatter is used to format
                the text output for individual speakers and ensures that the context is clear for
                language models further down the pipeline. The attributes `text` and `speaker_id` are
                available. The system instructions for the language model may need to include any
                necessary instructions to handle the formatting.
                Example: `@{speaker_id}: {text}`.
                Defaults to transcription output.

            speaker_passive_format: Formatter for passive speaker ID. As with the
                speaker_active_format, the attributes `text` and `speaker_id` are available.
                Example: `@{speaker_id} [background]: {text}`.
                Defaults to transcription output.

            prefer_current_speaker: Prefer current speaker ID. When set to true, groups of words close
                together are given extra weight to be identified as the same speaker.
                Defaults to False.

            focus_speakers: List of speaker IDs to focus on. When enabled, only these speakers are
                emitted as finalized frames and other speakers are considered passive. Words from
                other speakers are still processed, but only emitted when a focussed speaker has
                also said new words. A list of labels (e.g. `S1`, `S2`) or identifiers of known
                speakers (e.g. `speaker_1`, `speaker_2`) can be used.
                Defaults to [].

            ignore_speakers: List of speaker IDs to ignore. When enabled, these speakers are
                excluded from the transcription and their words are not processed. Their speech
                will not trigger any VAD or end of utterance detection. By default, any speaker
                with a label starting and ending with double underscores will be excluded (e.g.
                `__ASSISTANT__`).
                Defaults to [].

            focus_mode: Speaker focus mode for diarization. When set to `DiarizationFocusMode.RETAIN`,
                the STT engine will retain words spoken by other speakers (not listed in `ignore_speakers`)
                and process them as passive speaker frames. When set to `DiarizationFocusMode.IGNORE`,
                the STT engine will ignore words spoken by other speakers and they will not be processed.
                Defaults to `DiarizationFocusMode.RETAIN`.

            known_speakers: List of known speaker labels and identifiers. If you supply a list of
                labels and identifiers for speakers, then the STT engine will use them to attribute
                any spoken words to that speaker. This is useful when you want to attribute words
                to a specific speaker, such as the assistant or a specific user. Labels and identifiers
                can be obtained from a running STT session and then used in subsequent sessions.
                Identifiers are unique to each Speechmatics account and cannot be used across accounts.
                Refer to our examples on the format of the known_speakers parameter.
                Defaults to [].

            chunk_size: Audio chunk size for streaming. Defaults to 160.
            audio_encoding: Audio encoding format. Defaults to AudioEncoding.PCM_S16LE.
        """

        # Service configuration
        operating_point: OperatingPoint = OperatingPoint.ENHANCED
        domain: str | None = None
        language: Language | str = Language.EN
        output_locale: Language | str | None = None

        # Features
        enable_vad: bool = False
        enable_partials: bool = True
        max_delay: float = 1.0
        end_of_utterance_silence_trigger: float = 0.5
        end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED
        additional_vocab: list[AdditionalVocabEntry] = []
        punctuation_overrides: dict | None = None

        # Diarization
        enable_diarization: bool = False
        speaker_sensitivity: float = 0.5
        max_speakers: int | None = None
        speaker_active_format: str = "{text}"
        speaker_passive_format: str = "{text}"
        prefer_current_speaker: bool = False
        focus_speakers: list[str] = []
        ignore_speakers: list[str] = []
        focus_mode: DiarizationFocusMode = DiarizationFocusMode.RETAIN
        known_speakers: list[DiarizationKnownSpeaker] = []

        # Audio
        chunk_size: int = 160
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE

    class UpdateParams(BaseModel):
        """Update parameters for Speechmatics STT service.

        These are the only parameters that can be changed once a session has started. If you need to
        change the language, etc., then you must create a new instance of the service.

        Parameters:
            focus_speakers: List of speaker IDs to focus on. When enabled, only these speakers are
                emitted as finalized frames and other speakers are considered passive. Words from
                other speakers are still processed, but only emitted when a focussed speaker has
                also said new words. A list of labels (e.g. `S1`, `S2`) or identifiers of known
                speakers (e.g. `speaker_1`, `speaker_2`) can be used.
                Defaults to [].

            ignore_speakers: List of speaker IDs to ignore. When enabled, these speakers are
                excluded from the transcription and their words are not processed. Their speech
                will not trigger any VAD or end of utterance detection. By default, any speaker
                with a label starting and ending with double underscores will be excluded (e.g.
                `__ASSISTANT__`).
                Defaults to [].

            focus_mode: Speaker focus mode for diarization. When set to `DiarizationFocusMode.RETAIN`,
                the STT engine will retain words spoken by other speakers (not listed in `ignore_speakers`)
                and process them as passive speaker frames. When set to `DiarizationFocusMode.IGNORE`,
                the STT engine will ignore words spoken by other speakers and they will not be processed.
                Defaults to `DiarizationFocusMode.RETAIN`.
        """

        focus_speakers: list[str] = []
        ignore_speakers: list[str] = []
        focus_mode: DiarizationFocusMode = DiarizationFocusMode.RETAIN

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int = 16000,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initialize the Speechmatics STT service.

        Args:
            api_key: Speechmatics API key for authentication. Uses environment variable
                `SPEECHMATICS_API_KEY` if not provided.
            base_url: Base URL for Speechmatics API. Uses environment variable `SPEECHMATICS_RT_URL`
                or defaults to `wss://eu2.rt.speechmatics.com/v2`.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            params: Optional[InputParams]: Input parameters for the service.
            **kwargs: Additional arguments passed to STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Service parameters
        self._api_key: str = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self._base_url: str = (
            base_url or os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2"
        )

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Default parameters
        self._params = params or SpeechmaticsSTTService.InputParams()

        # Deprecation check
        _check_deprecated_args(kwargs, self._params)

        # Complete configuration objects
        self._transcription_config: TranscriptionConfig = None
        self._process_config()

        # STT client
        self._client: AsyncClient | None = None

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []

        # Speaking states
        self._is_speaking: bool = False

        # Timing info
        self._start_time: datetime.datetime | None = None
        self._total_time: datetime.timedelta | None = None

        # Event handlers
        if self._params.enable_diarization:
            self._register_event_handler("on_speakers_result")

        # EndOfUtterance fallback timer
        self._end_of_utterance_timer: asyncio.Task | None = None

    async def start(self, frame: StartFrame):
        """Called when the new session starts."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Called when the session ends."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Called when the session is cancelled."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Adds audio to the audio buffer and yields None."""
        try:
            if self._client:
                await self._client.send_audio(audio)
            yield None
        except Exception as e:
            logger.error(f"Speechmatics error: {e}")
            yield ErrorFrame(f"Speechmatics error: {e}", fatal=False)
            await self._disconnect()

    def update_params(
        self,
        params: UpdateParams,
    ) -> None:
        """Updates the speaker configuration.

        This can update the speakers to listen to or ignore during an in-flight
        transcription. Only available if diarization is enabled.

        Args:
            params: Update parameters for the service.
        """
        # Check possible
        if not self._params.enable_diarization:
            raise ValueError("Diarization is not enabled")

        # Update the diarization configuration
        if params.focus_speakers is not None:
            self._params.focus_speakers = params.focus_speakers
        if params.ignore_speakers is not None:
            self._params.ignore_speakers = params.ignore_speakers
        if params.focus_mode is not None:
            self._params.focus_mode = params.focus_mode

    async def send_message(self, message: ClientMessageType | str, **kwargs: Any) -> None:
        """Send a message to the STT service.

        This sends a message to the STT service via the underlying transport. If the session
        is not running, this will raise an exception. Messages in the wrong format will also
        cause an error.

        Args:
            message: Message to send to the STT service.
            **kwargs: Additional arguments passed to the underlying transport.
        """
        try:
            payload = {"message": message}
            payload.update(kwargs)
            logger.debug(f"Sending message to STT: {payload}")
            asyncio.run_coroutine_threadsafe(
                self._client.send_message(payload), self.get_event_loop()
            )
        except Exception as e:
            raise RuntimeError(f"error sending message to STT: {e}")

    async def _connect(self) -> None:
        """Connect to the STT service."""
        # Create new STT RT client
        self._client = AsyncClient(
            api_key=self._api_key,
            url=_get_endpoint_url(self._base_url),
        )

        # Log the event
        logger.debug(f"{self} Connecting to Speechmatics STT service")

        # Recognition started event
        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]):
            logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Partial transcript event
        if self._params.enable_partials:

            @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def _evt_on_partial_transcript(message: dict[str, Any]):
                self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=True)

        # End of Utterance
        if self._params.end_of_utterance_mode == EndOfUtteranceMode.FIXED:

            @self._client.on(ServerMessageType.END_OF_UTTERANCE)
            def _evt_on_end_of_utterance(message: dict[str, Any]):
                logger.debug("End of utterance received from STT")
                asyncio.run_coroutine_threadsafe(
                    self._handle_end_of_utterance(), self.get_event_loop()
                )

        # Speaker Result
        if self._params.enable_diarization:

            @self._client.on(ServerMessageType.SPEAKERS_RESULT)
            def _evt_on_speakers_result(message: dict[str, Any]):
                logger.debug("Speakers result received from STT")
                asyncio.run_coroutine_threadsafe(
                    self._call_event_handler("on_speakers_result", message),
                    self.get_event_loop(),
                )

        # Start session
        try:
            await self._client.start_session(
                transcription_config=self._transcription_config,
                audio_format=AudioFormat(
                    encoding=self._params.audio_encoding,
                    sample_rate=self.sample_rate,
                    chunk_size=self._params.chunk_size,
                ),
            )
            logger.debug(f"{self} Connected to Speechmatics STT service")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} Error connecting to Speechmatics: {e}")
            self._client = None

    async def _disconnect(self) -> None:
        """Disconnect from the STT service."""
        # Disconnect the client
        logger.debug(f"{self} Disconnecting from Speechmatics STT service")
        try:
            if self._client:
                await asyncio.wait_for(self._client.close(), timeout=5.0)
                logger.debug(f"{self} Disconnected from Speechmatics STT service")
        except asyncio.TimeoutError:
            logger.warning(f"{self} Timeout while closing Speechmatics client connection")
        except Exception as e:
            logger.error(f"{self} Error closing Speechmatics client: {e}")
        finally:
            self._client = None
            await self._call_event_handler("on_disconnected")

    def _process_config(self) -> None:
        """Create a formatted STT transcription config.

        Creates a transcription config object based on the service parameters. Aligns
        with the Speechmatics RT API transcription config.
        """
        # Transcription config
        transcription_config = TranscriptionConfig(
            language=self._params.language,
            domain=self._params.domain,
            output_locale=self._params.output_locale,
            operating_point=self._params.operating_point,
            diarization="speaker" if self._params.enable_diarization else None,
            enable_partials=self._params.enable_partials,
            max_delay=self._params.max_delay,
        )

        # Additional vocab
        if self._params.additional_vocab:
            transcription_config.additional_vocab = [
                {
                    "content": e.content,
                    **({"sounds_like": e.sounds_like} if e.sounds_like else {}),
                }
                for e in self._params.additional_vocab
            ]

        # Diarization
        if self._params.enable_diarization:
            dz_cfg = {}
            if self._params.speaker_sensitivity is not None:
                dz_cfg["speaker_sensitivity"] = self._params.speaker_sensitivity
            if self._params.prefer_current_speaker is not None:
                dz_cfg["prefer_current_speaker"] = self._params.prefer_current_speaker
            if self._params.known_speakers:
                dz_cfg["speakers"] = {
                    s.label: s.speaker_identifiers for s in self._params.known_speakers
                }
            if self._params.max_speakers is not None:
                dz_cfg["max_speakers"] = self._params.max_speakers
            if dz_cfg:
                transcription_config.speaker_diarization_config = dz_cfg

        # End of Utterance (for fixed)
        if (
            self._params.end_of_utterance_silence_trigger
            and self._params.end_of_utterance_mode == EndOfUtteranceMode.FIXED
        ):
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=self._params.end_of_utterance_silence_trigger,
            )

        # Punctuation overrides
        if self._params.punctuation_overrides:
            transcription_config.punctuation_overrides = self._params.punctuation_overrides

        # Set config
        self._transcription_config = transcription_config

    def _handle_transcript(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        # Add the speech fragments
        has_changed = self._add_speech_fragments(
            message=message,
            is_final=is_final,
        )

        # Skip if unchanged
        if not has_changed:
            return

        # Set a timer for the end of utterance
        self._end_of_utterance_timer_start()

        # Send frames
        asyncio.run_coroutine_threadsafe(self._send_frames(), self.get_event_loop())

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def _end_of_utterance_timer_start(self):
        """Start the timer for the end of utterance.

        This will use the STT's `end_of_utterance_silence_trigger` value and set
        a timer to send the latest transcript to the pipeline. It is used as a
        fallback from the EnfOfUtterance messages from the STT.

        Note that the `end_of_utterance_silence_trigger` will be from when the
        last updated speech was received and this will likely be longer in
        real world time to that inside of the STT engine.
        """
        # Reset the end of utterance timer
        if self._end_of_utterance_timer is not None:
            self._end_of_utterance_timer.cancel()

        # Send after a delay
        async def send_after_delay(delay: float):
            await asyncio.sleep(delay)
            logger.debug("Fallback EndOfUtterance triggered.")
            asyncio.run_coroutine_threadsafe(self._handle_end_of_utterance(), self.get_event_loop())

        # Start the timer
        self._end_of_utterance_timer = asyncio.create_task(
            send_after_delay(self._params.end_of_utterance_silence_trigger * 2)
        )

    async def _handle_end_of_utterance(self):
        """Handle the end of utterance event.

        This will check for any running timers for end of utterance, reset them,
        and then send a finalized frame to the pipeline.
        """
        # Send the frames
        await self._send_frames(finalized=True)

        # Reset the end of utterance timer
        if self._end_of_utterance_timer:
            self._end_of_utterance_timer.cancel()
            self._end_of_utterance_timer = None

    async def _send_frames(self, finalized: bool = False) -> None:
        """Send frames to the pipeline.

        Send speech frames to the pipeline. If VAD is enabled, then this will
        also send an interruption and user started speaking frames. When the
        final transcript is received, then this will send a user stopped speaking
        and stop interruption frames.

        Args:
            finalized: Whether the data is final or partial.
        """
        # Get speech frames (InterimTranscriptionFrame)
        speech_frames = self._get_frames_from_fragments()

        # Skip if no frames
        if not speech_frames:
            return

        # Check at least one frame is active
        if not any(frame.is_active for frame in speech_frames):
            return

        # Frames to send
        downstream_frames: list[Frame] = []

        # If VAD is enabled, then send a speaking frame
        if self._params.enable_vad and not self._is_speaking:
            logger.debug("User started speaking")
            self._is_speaking = True
            await self.push_interruption_task_frame_and_wait()
            downstream_frames += [UserStartedSpeakingFrame()]

        # If final, then re-parse into TranscriptionFrame
        if finalized:
            # Reset the speech fragments
            self._speech_fragments.clear()

            # Transform frames
            downstream_frames += [
                TranscriptionFrame(
                    **frame._as_frame_attributes(
                        self._params.speaker_active_format, self._params.speaker_passive_format
                    )
                )
                for frame in speech_frames
            ]

            # Log transcript(s)
            logger.debug(f"Finalized transcript: {[f.text for f in downstream_frames]}")

        # Return as interim results (unformatted)
        else:
            downstream_frames += [
                InterimTranscriptionFrame(
                    **frame._as_frame_attributes(
                        self._params.speaker_active_format, self._params.speaker_passive_format
                    )
                )
                for frame in speech_frames
            ]

        # If VAD is enabled, then send a speaking frame
        if self._params.enable_vad and self._is_speaking and finalized:
            logger.debug("User stopped speaking")
            self._is_speaking = False
            downstream_frames += [UserStoppedSpeakingFrame()]

        # Send the DOWNSTREAM frames
        for frame in downstream_frames:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _add_speech_fragments(self, message: dict[str, Any], is_final: bool = False) -> bool:
        """Takes a new Partial or Final from the STT engine.

        Accumulates it into the _speech_data list. As new final data is added, all
        partials are removed from the list.

        Note: If a known speaker is `__[A-Z0-9_]{2,}__`, then the words are skipped,
        as this is used to protect against self-interruption by the assistant or to
        block out specific known voices.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.

        Returns:
            bool: True if the speech data was updated, False otherwise.
        """
        # Parsed new speech data from the STT engine
        fragments: list[SpeechFragment] = []

        # Current length of the speech data
        current_length = len(self._speech_fragments)

        # Iterate over the results in the payload
        for result in message.get("results", []):
            alt = result.get("alternatives", [{}])[0]
            if alt.get("content", None):
                # Create the new fragment
                fragment = SpeechFragment(
                    start_time=result.get("start_time", 0),
                    end_time=result.get("end_time", 0),
                    language=alt.get("language", Language.EN),
                    is_eos=alt.get("is_eos", False),
                    is_final=is_final,
                    attaches_to=result.get("attaches_to", ""),
                    content=alt.get("content", ""),
                    speaker=alt.get("speaker", None),
                    confidence=alt.get("confidence", 1.0),
                    result=result,
                )

                # Speaker filtering
                if fragment.speaker:
                    # Drop `__XX__` speakers
                    if re.match(r"^__[A-Z0-9_]{2,}__$", fragment.speaker):
                        continue

                    # Drop speakers not focussed on
                    if (
                        self._params.focus_mode == DiarizationFocusMode.IGNORE
                        and self._params.focus_speakers
                        and fragment.speaker not in self._params.focus_speakers
                    ):
                        continue

                    # Drop ignored speakers
                    if (
                        self._params.ignore_speakers
                        and fragment.speaker in self._params.ignore_speakers
                    ):
                        continue

                # Add the fragment
                fragments.append(fragment)

        # Remove existing partials, as new partials and finals are provided
        self._speech_fragments = [frag for frag in self._speech_fragments if frag.is_final]

        # Return if no new fragments and length of the existing data is unchanged
        if not fragments and len(self._speech_fragments) == current_length:
            return False

        # Add the fragments to the speech data
        self._speech_fragments.extend(fragments)

        # Data was updated
        return True

    def _get_frames_from_fragments(self) -> list[SpeakerFragments]:
        """Get speech data objects for the current fragment list.

        Each speech fragments is grouped by contiguous speaker and then
        returned as internal SpeakerFragments objects with the `speaker_id` field
        set to the current speaker (string). An utterance may contain speech from
        more than one speaker (e.g. S1, S2, S1, S3, ...), so they are kept
        in strict order for the context of the conversation.

        Returns:
            List[SpeakerFragments]: The list of objects.
        """
        # Speaker groups
        current_speaker: str | None = None
        speaker_groups: list[list[SpeechFragment]] = [[]]

        # Group by speakers
        for frag in self._speech_fragments:
            if frag.speaker != current_speaker:
                current_speaker = frag.speaker
                if speaker_groups[-1]:
                    speaker_groups.append([])
            speaker_groups[-1].append(frag)

        # Create SpeakerFragments objects
        speaker_fragments: list[SpeakerFragments] = []
        for group in speaker_groups:
            sd = self._get_speaker_fragments_from_fragment_group(group)
            if sd:
                speaker_fragments.append(sd)

        # Return the grouped SpeakerFragments objects
        return speaker_fragments

    def _get_speaker_fragments_from_fragment_group(
        self,
        group: list[SpeechFragment],
    ) -> SpeakerFragments | None:
        """Take a group of fragments and piece together into SpeakerFragments.

        Each fragment for a given speaker is assembled into a string,
        taking into consideration whether words are attached to the
        previous or next word (notably punctuation). This ensures that
        the text does not have extra spaces. This will also check for
        any straggling punctuation from earlier utterances that should
        be removed.

        Args:
            group: List of SpeechFragment objects.

        Returns:
            SpeakerFragments: The object for the group.
        """
        # Check for starting fragments that are attached to previous
        if group and group[0].attaches_to == "previous":
            group = group[1:]

        # Check for trailing fragments that are attached to next
        if group and group[-1].attaches_to == "next":
            group = group[:-1]

        # Check there are results
        if not group:
            return None

        # Get the timing extremes
        start_time = min(frag.start_time for frag in group)

        # Timestamp
        ts = (self._start_time + datetime.timedelta(seconds=start_time)).isoformat(
            timespec="milliseconds"
        )

        # Determine if the speaker is considered active
        is_active = True
        if self._params.enable_diarization and self._params.focus_speakers:
            is_active = group[0].speaker in self._params.focus_speakers

        # Return the SpeakerFragments object
        return SpeakerFragments(
            speaker_id=group[0].speaker,
            timestamp=ts,
            language=group[0].language,
            fragments=group,
            is_active=is_active,
        )


def _get_endpoint_url(url: str) -> str:
    """Format the endpoint URL with the SDK and app versions.

    Args:
        url: The base URL for the endpoint.

    Returns:
        str: The formatted endpoint URL.
    """
    query_params = dict()
    query_params["sm-app"] = f"pipecat/{__version__}"
    query = urlencode(query_params)

    return f"{url}?{query}"


def _language_to_speechmatics_language(language: Language) -> str:
    """Convert a Language enum to a Speechmatics language code.

    Args:
        language: The Language enum to convert.

    Returns:
        str: The Speechmatics language code, if found.
    """
    # List of supported input languages
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BA: "ba",
        Language.EU: "eu",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.YUE: "yue",
        Language.CA: "ca",
        Language.HR: "hr",
        Language.CS: "cs",
        Language.DA: "da",
        Language.NL: "nl",
        Language.EN: "en",
        Language.EO: "eo",
        Language.ET: "et",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.DE: "de",
        Language.EL: "el",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.IT: "it",
        Language.ID: "id",
        Language.GA: "ga",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.LV: "lv",
        Language.LT: "lt",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.CMN: "cmn",
        Language.MR: "mr",
        Language.MN: "mn",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.ES: "es",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UG: "ug",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.CY: "cy",
    }

    # Get the language code
    result = BASE_LANGUAGES.get(language)

    # Fail if language is not supported
    if not result:
        raise ValueError(f"Unsupported language: {language}")

    # Return the language code
    return result


def _locale_to_speechmatics_locale(language_code: str, locale: Language) -> str | None:
    """Convert a Language enum to a Speechmatics language code.

    Args:
        language_code: The language code.
        locale: The Language enum to convert.

    Returns:
        str: The Speechmatics language code, if found.
    """
    # Languages and output locales
    LOCALES = {
        "en": {
            Language.EN_GB: "en-GB",
            Language.EN_US: "en-US",
            Language.EN_AU: "en-AU",
        },
    }

    # Get the locale code
    result = LOCALES.get(language_code, {}).get(locale)

    # Fail if locale is not supported
    if not result:
        logger.warning(f"Unsupported output locale: {locale}, defaulting to {language_code}")

    # Return the locale code
    return result


def _check_deprecated_args(kwargs: dict, params: SpeechmaticsSTTService.InputParams) -> None:
    """Check arguments for deprecation and update params if necessary.

    This function will show deprecation warnings for deprecated arguments and
    migrate them to the new location in the params object. If the new location
    is None, the argument is not used.

    Args:
        kwargs: Keyword arguments passed to the constructor.
        params: Input parameters for the service.
    """

    # Show deprecation warnings
    def _deprecation_warning(old: str, new: str | None = None):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if new:
                message = f"`{old}` is deprecated, use `InputParams.{new}`"
            else:
                message = f"`{old}` is deprecated and not used"
            warnings.warn(message, DeprecationWarning)

    # List of deprecated arguments and their new location
    deprecated_args = [
        ("language", "language"),
        ("language_code", "language"),
        ("domain", "domain"),
        ("output_locale", "output_locale"),
        ("output_locale_code", "output_locale"),
        ("enable_partials", "enable_partials"),
        ("max_delay", "max_delay"),
        ("chunk_size", "chunk_size"),
        ("audio_encoding", "audio_encoding"),
        ("end_of_utterance_silence_trigger", "end_of_utterance_silence_trigger"),
        {"enable_speaker_diarization", "enable_diarization"},
        ("text_format", "speaker_active_format"),
        ("max_speakers", "max_speakers"),
        ("transcription_config", None),
    ]

    # Show warnings + migrate the arguments
    for old, new in deprecated_args:
        if old in kwargs:
            _deprecation_warning(old, new)
            if kwargs.get(old, None) is not None:
                params.__setattr__(new, kwargs[old])

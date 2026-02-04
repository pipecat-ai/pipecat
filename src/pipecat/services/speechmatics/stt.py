#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics STT service integration."""

import asyncio
import os
from enum import Enum
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from speechmatics.voice import (
        AdditionalVocabEntry,
        AgentClientMessageType,
        AgentServerMessageType,
        AudioEncoding,
        EndOfUtteranceMode,
        OperatingPoint,
        SpeakerFocusConfig,
        SpeakerFocusMode,
        SpeakerIdentifier,
        SpeechSegmentConfig,
        VoiceAgentClient,
        VoiceAgentConfig,
        VoiceAgentConfigPreset,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Speechmatics, you need to `pip install pipecat-ai[speechmatics]`."
    )
    raise Exception(f"Missing module: {e}")


load_dotenv()


class TurnDetectionMode(str, Enum):
    """Endpoint and turn detection handling mode.

    How the STT engine handles the endpointing of speech. If using Pipecat's built-in endpointing,
    then use `TurnDetectionMode.EXTERNAL` (default).

    To use the STT engine's built-in endpointing, then use `TurnDetectionMode.ADAPTIVE` for simple
    voice activity detection or `TurnDetectionMode.SMART_TURN` for more advanced ML-based
    endpointing.
    """

    FIXED = "fixed"
    EXTERNAL = "external"
    ADAPTIVE = "adaptive"
    SMART_TURN = "smart_turn"


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.
    """

    # Export related classes as class attributes
    TurnDetectionMode = TurnDetectionMode
    AudioEncoding = AudioEncoding
    OperatingPoint = OperatingPoint
    SpeakerFocusMode = SpeakerFocusMode
    SpeakerFocusConfig = SpeakerFocusConfig
    SpeakerIdentifier = SpeakerIdentifier
    AdditionalVocabEntry = AdditionalVocabEntry

    class InputParams(BaseModel):
        """Configuration parameters for Speechmatics STT service.

        Parameters:
            domain: Domain for Speechmatics API. Defaults to None.

            language: Language code for transcription. Defaults to `Language.EN`.

            turn_detection_mode: Endpoint handling, one of `TurnDetectionMode.FIXED`,
                `TurnDetectionMode.EXTERNAL`, `TurnDetectionMode.ADAPTIVE` and
                `TurnDetectionMode.SMART_TURN`. Defaults to `TurnDetectionMode.EXTERNAL`.

            speaker_active_format: Formatter for active speaker ID. This formatter is used to format
                the text output for individual speakers and ensures that the context is clear for
                language models further down the pipeline. The attributes `text` and `speaker_id` are
                available. The system instructions for the language model may need to include any
                necessary instructions to handle the formatting.
                Example: `@{speaker_id}: {text}`. Defaults to None.

            speaker_passive_format: Formatter for passive speaker ID. As with the
                speaker_active_format, the attributes `text` and `speaker_id` are available.
                Example: `@{speaker_id} [background]: {text}`. Defaults to None.

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

            focus_mode: Speaker focus mode for diarization. When set to `SpeakerFocusMode.RETAIN`,
                the STT engine will retain words spoken by other speakers (not listed in `ignore_speakers`)
                and process them as passive speaker frames. When set to `SpeakerFocusMode.IGNORE`,
                the STT engine will ignore words spoken by other speakers and they will not be processed.
                Defaults to `SpeakerFocusMode.RETAIN`.

            known_speakers: List of known speaker labels and identifiers. If you supply a list of
                labels and identifiers for speakers, then the STT engine will use them to attribute
                any spoken words to that speaker. This is useful when you want to attribute words
                to a specific speaker, such as the assistant or a specific user. Labels and identifiers
                can be obtained from a running STT session and then used in subsequent sessions.
                Identifiers are unique to each Speechmatics account and cannot be used across accounts.
                Refer to our examples on the format of the known_speakers parameter.
                Defaults to [].

            additional_vocab: List of additional vocabulary entries. If you supply a list of
                additional vocabulary entries, the this will increase the weight of the words in the
                vocabulary and help the STT engine to better transcribe the words.
                Defaults to [].

            audio_encoding: Audio encoding format. Defaults to AudioEncoding.PCM_S16LE.

            operating_point: Operating point for transcription accuracy vs. latency tradeoff. It is
                recommended to use OperatingPoint.ENHANCED for most use cases. Default to enhanced.

            max_delay: Maximum delay in seconds for transcription. This forces the STT engine to
                speed up the processing of transcribed words and reduces the interval between partial
                and final results. Lower values can have an impact on accuracy.

            end_of_utterance_silence_trigger: Maximum delay in seconds for end of utterance trigger.
                The delay is used to wait for any further transcribed words before emitting the final
                word frames. The value must be lower than max_delay.

            end_of_utterance_max_delay: Maximum delay in seconds for end of utterance delay.
                The delay is used to wait for any further transcribed words before emitting the final
                word frames. The value must be greater than end_of_utterance_silence_trigger.

            punctuation_overrides: Punctuation overrides. This allows you to override the punctuation
                in the STT engine. This is useful for languages that use different punctuation
                than English. See documentation for more information.

            include_partials: Include partial segment fragments (words) in the output of
                AddPartialSegment messages. Partial fragments from the STT will always be used for
                speaker activity detection. This setting is used only for the formatted text output
                of individual segments.

            split_sentences: Emit finalized sentences mid-turn. When enabled, as soon as a sentence
                is finalized, it will be emitted as a final segment. This is useful for applications
                that need to process sentences as they are finalized. Defaults to False.

            enable_diarization: Enable speaker diarization. When enabled, the STT engine will
                determine and attribute words to unique speakers. The speaker_sensitivity
                parameter can be used to adjust the sensitivity of diarization.

            speaker_sensitivity: Diarization sensitivity. A higher value increases the sensitivity
                of diarization and helps when two or more speakers have similar voices.

            max_speakers: Maximum number of speakers to detect. This forces the STT engine to cluster
                words into a fixed number of speakers. It should not be used to limit the number of
                speakers, unless it is clear that there will only be a known number of speakers.

            prefer_current_speaker: Prefer current speaker ID. When set to true, groups of words close
                together are given extra weight to be identified as the same speaker.

            extra_params: Extra parameters to pass to the STT engine. This is a dictionary of
                additional parameters that can be used to configure the STT engine.
                Default to None.

        """

        # Service configuration
        domain: str | None = None
        language: Language | str = Language.EN

        # Endpointing mode
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.EXTERNAL

        # Output formatting
        speaker_active_format: str | None = None
        speaker_passive_format: str | None = None

        # Speakers
        focus_speakers: list[str] = []
        ignore_speakers: list[str] = []
        focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN
        known_speakers: list[SpeakerIdentifier] = []

        # Custom dictionary
        additional_vocab: list[AdditionalVocabEntry] = []

        # Audio
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE

        # -------------------
        # Advanced features
        # -------------------

        # Features
        operating_point: OperatingPoint | None = None
        max_delay: float | None = None
        end_of_utterance_silence_trigger: float | None = None
        end_of_utterance_max_delay: float | None = None
        punctuation_overrides: dict | None = None
        include_partials: bool | None = None
        split_sentences: bool | None = None

        # Diarization
        enable_diarization: bool | None = None
        speaker_sensitivity: float | None = None
        max_speakers: int | None = None
        prefer_current_speaker: bool | None = None

        # Extra parameters
        extra_params: dict | None = None

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

            focus_mode: Speaker focus mode for diarization. When set to `SpeakerFocusMode.RETAIN`,
                the STT engine will retain words spoken by other speakers (not listed in `ignore_speakers`)
                and process them as passive speaker frames. When set to `SpeakerFocusMode.IGNORE`,
                the STT engine will ignore words spoken by other speakers and they will not be processed.
                Defaults to `SpeakerFocusMode.RETAIN`.
        """

        focus_speakers: list[str] = []
        ignore_speakers: list[str] = []
        focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int | None = None,
        params: InputParams | None = None,
        should_interrupt: bool = True,
        **kwargs,
    ):
        """Initialize the Speechmatics STT service.

        Args:
            api_key: Speechmatics API key for authentication. Uses environment variable
                `SPEECHMATICS_API_KEY` if not provided.
            base_url: Base URL for Speechmatics API. Uses environment variable `SPEECHMATICS_RT_URL`
                or defaults to `wss://eu2.rt.speechmatics.com/v2`.
            sample_rate: Optional audio sample rate in Hz.
            params: Optional[InputParams]: Input parameters for the service.
            should_interrupt: Determine whether the bot should be interrupted when Speechmatics turn_detection_mode is configured to detect user speech.
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

        # Default params
        params = params or SpeechmaticsSTTService.InputParams()
        self._should_interrupt = should_interrupt

        # Deprecation check
        self._check_deprecated_args(kwargs, params)

        # Voice agent
        self._client: VoiceAgentClient | None = None
        self._config: VoiceAgentConfig = self._prepare_config(params)

        # Outbound frame queue
        self._outbound_frames: asyncio.Queue[Frame] = asyncio.Queue()

        # Output formatting
        if params.speaker_active_format is None:
            params.speaker_active_format = (
                "@{speaker_id}: {text}" if params.enable_diarization else "{text}"
            )

        # Framework options
        self._enable_vad: bool = self._config.end_of_utterance_mode not in [
            EndOfUtteranceMode.FIXED,
            EndOfUtteranceMode.EXTERNAL,
        ]
        self._speaker_active_format: str = params.speaker_active_format
        self._speaker_passive_format: str = (
            params.speaker_passive_format or params.speaker_active_format
        )

        # Model + metrics
        self.set_model_name(self._config.operating_point.value)

        # Message queue
        self._stt_msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stt_msg_task: asyncio.Task | None = None

        # Speaking states
        self._is_speaking: bool = False
        self._bot_speaking: bool = False

        # Event handlers
        if params.enable_diarization:
            self._register_event_handler("on_speakers_result")

    # ============================================================================
    # LIFE-CYCLE / SESSION MANAGEMENT
    # ============================================================================

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

    async def _connect(self) -> None:
        """Connect to the STT service.

        - Create STT client
        - Register handlers for messages
        - Connect to the client
        - Start message processing task
        """
        # Log the event
        logger.debug(f"{self} connecting to Speechmatics STT service")

        # Update the audio sample rate
        self._config.sample_rate = self.sample_rate

        # STT client
        self._client: VoiceAgentClient = VoiceAgentClient(
            api_key=self._api_key,
            url=self._base_url,
            app=f"pipecat/{pipecat_version()}",
            config=self._config,
        )

        # Add message queue
        def add_message(message: dict[str, Any]):
            self._stt_msg_queue.put_nowait(message)

        # Add listeners
        self._client.on(AgentServerMessageType.ADD_PARTIAL_SEGMENT, add_message)
        self._client.on(AgentServerMessageType.ADD_SEGMENT, add_message)

        # Add listeners for VAD
        if self._enable_vad:
            self._client.on(AgentServerMessageType.START_OF_TURN, add_message)
            self._client.on(AgentServerMessageType.END_OF_TURN, add_message)

        # Speaker result listener
        if self._config.enable_diarization:
            self._client.on(AgentServerMessageType.SPEAKERS_RESULT, add_message)

        # Other messages for debugging
        self._client.on(AgentServerMessageType.ERROR, add_message)
        self._client.on(AgentServerMessageType.WARNING, add_message)
        self._client.on(AgentServerMessageType.INFO, add_message)
        self._client.on(AgentServerMessageType.END_OF_TURN_PREDICTION, add_message)
        self._client.on(AgentServerMessageType.END_OF_UTTERANCE, add_message)

        # Connect to the client
        try:
            await self._client.connect()
            logger.debug(f"{self} connected")
        except Exception as e:
            self._client = None
            await self.push_error(error_msg=f"Error connecting to STT service: {e}", exception=e)

        # Start message processing task
        if not self._stt_msg_task:
            self._stt_msg_task = self.create_task(self._process_stt_messages())

    async def _disconnect(self) -> None:
        """Disconnect from the STT service.

        - Cancel message processing task
        - Disconnect the client
        - Emit on_disconnected event handler for clients
        """
        # Cancel the message processing task
        if self._stt_msg_task:
            await self.cancel_task(self._stt_msg_task)
            self._stt_msg_task = None

        # Disconnect the client
        logger.debug(f"{self} disconnecting from Speechmatics STT service")
        try:
            if self._client:
                await self._client.disconnect()
        except asyncio.TimeoutError:
            logger.warning(f"{self} timeout while closing Speechmatics client connection")
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Speechmatics client: {e}", exception=e)
        finally:
            self._client = None
            await self._call_event_handler("on_disconnected")

    async def _process_stt_messages(self) -> None:
        """Process messages from the STT client.

        Messages from the STT client are processed in a separate task to avoid blocking the main
        thread. They are handled in strict order in which they are received.
        """
        try:
            while True:
                message = await self._stt_msg_queue.get()
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    def _prepare_config(self, params: InputParams) -> VoiceAgentConfig:
        """Parse the InputParams into VoiceAgentConfig."""
        # Preset
        config = VoiceAgentConfigPreset.load(params.turn_detection_mode.value)

        # Language + domain
        config.language = self._language_to_speechmatics_language(params.language)
        config.domain = params.domain
        config.output_locale = self._locale_to_speechmatics_locale(config.language, params.language)

        # Speaker config
        config.speaker_config = SpeakerFocusConfig(
            focus_speakers=params.focus_speakers,
            ignore_speakers=params.ignore_speakers,
            focus_mode=params.focus_mode,
        )
        config.known_speakers = params.known_speakers

        # Custom dictionary
        config.additional_vocab = params.additional_vocab

        # Advanced parameters
        for param in [
            "operating_point",
            "max_delay",
            "end_of_utterance_silence_trigger",
            "end_of_utterance_max_delay",
            "punctuation_overrides",
            "include_partials",
            "split_sentences",
            "enable_diarization",
            "speaker_sensitivity",
            "max_speakers",
            "prefer_current_speaker",
        ]:
            if getattr(params, param) is not None:
                setattr(config, param, getattr(params, param))

        # Extra parameters
        if isinstance(params.extra_params, dict):
            for key, value in params.extra_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Enable sentences
        config.speech_segment_config = SpeechSegmentConfig(
            emit_sentences=params.split_sentences or False
        )

        # Return the complete config
        return config

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
        if not self._config.enable_diarization:
            raise ValueError("Diarization is not enabled")

        # Update the existing diarization configuration
        if params.focus_speakers is not None:
            self._config.speaker_config.focus_speakers = params.focus_speakers
        if params.ignore_speakers is not None:
            self._config.speaker_config.ignore_speakers = params.ignore_speakers
        if params.focus_mode is not None:
            self._config.speaker_config.focus_mode = params.focus_mode

        # Send the update
        if self._client:
            self._client.update_diarization_config(self._config.speaker_config)

    # ============================================================================
    # HANDLE ENGINE MESSAGES
    # ============================================================================

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle a message from the STT client."""
        event = message.get("message", "")

        # Handle events
        match event:
            case AgentServerMessageType.ADD_PARTIAL_SEGMENT:
                await self._handle_partial_segment(message)
            case AgentServerMessageType.ADD_SEGMENT:
                await self._handle_segment(message)
            case AgentServerMessageType.START_OF_TURN:
                await self._handle_start_of_turn(message)
            case AgentServerMessageType.END_OF_TURN:
                await self._handle_end_of_turn(message)
            case AgentServerMessageType.SPEAKERS_RESULT:
                await self._handle_speakers_result(message)
            case _:
                logger.debug(f"{self} {event} -> {message}")

    async def _handle_partial_segment(self, message: dict[str, Any]) -> None:
        """Handle AddPartialSegment events.

        AddPartialSegment events are triggered by Speechmatics STT when it detects a
        partial segment of speech. These events provide the partial transcript for
        the current speaking turn.

        Args:
            message: the message payload.
        """
        # Handle segments
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            await self._send_frames(segments)

    async def _handle_segment(self, message: dict[str, Any]) -> None:
        """Handle AddSegment events.

        AddSegment events are triggered by Speechmatics STT when it detects a
        final segment of speech. These events provide the final transcript for
        the current speaking turn.

        Args:
            message: the message payload.
        """
        # Handle segments
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            await self._send_frames(segments, finalized=True)

    async def _handle_start_of_turn(self, message: dict[str, Any]) -> None:
        """Handle StartOfTurn events.

        When Speechmatics STT detects the start of a new speaking turn, a StartOfTurn
        event is triggered. This triggers bot interruption to stop any ongoing speech
        synthesis and signals the start of user speech detection.

        The service will:
        - Send a BotInterruptionFrame upstream to stop bot speech
        - Send a UserStartedSpeakingFrame downstream to notify other components
        - Start metrics collection for measuring response times

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} StartOfTurn received")
        # await self.start_processing_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.push_interruption_task_frame_and_wait()

    async def _handle_end_of_turn(self, message: dict[str, Any]) -> None:
        """Handle EndOfTurn events.

        EndOfTurn events are triggered by Speechmatics STT when it concludes a
        speaking turn. This occurs either due to silence or reaching the
        end-of-turn confidence thresholds. These events provide the final
        transcript for the completed turn.

        The service will:
        - Stop processing metrics collection
        - Send a UserStoppedSpeakingFrame to signal turn completion

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} EndOfTurn received")
        # await self.stop_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    async def _handle_speakers_result(self, message: dict[str, Any]) -> None:
        """Handle SpeakersResult events.

        SpeakersResult events are triggered by Speechmatics STT when it provides
        speaker information for the current speaking turn.

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} speakers result received from STT")
        await self._call_event_handler("on_speakers_result", message)

    # ============================================================================
    # SEND FRAMES TO PIPELINE
    # ============================================================================

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for VAD and metrics handling.

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        # Forward to parent
        await super().process_frame(frame, direction)

        # Track the bot
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        # Force finalization
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._enable_vad:
                logger.warning(
                    f"{self} VADUserStoppedSpeakingFrame received but internal VAD is being used"
                )
            elif not self._enable_vad and self._client is not None:
                self.request_finalize()
                self._client.finalize()

    async def _send_frames(self, segments: list[dict[str, Any]], finalized: bool = False) -> None:
        """Send frames to the pipeline.

        Args:
            segments: The segments to send.
            finalized: Whether the data is final or partial.
        """
        # Skip if no frames
        if not segments:
            return

        # Frames to send
        frames: list[Frame] = []

        # Create frame from segment
        def attr_from_segment(segment: dict[str, Any]) -> dict[str, Any]:
            # Formats the output text based on the speaker and defined formats from the config.
            text = (
                self._speaker_active_format
                if segment.get("is_active", True)
                else self._speaker_passive_format
            ).format(
                **{
                    "speaker_id": segment.get("speaker_id", "UU"),
                    "text": segment.get("text", ""),
                    "ts": segment.get("timestamp"),
                    "lang": segment.get("language"),
                }
            )

            # Return the attributes for the frame
            return {
                "text": text,
                "user_id": segment.get("speaker_id") or "",
                "timestamp": segment.get("timestamp"),
                "language": segment.get("language"),
                "result": segment.get("results", []),
            }

        # If final, then re-parse into TranscriptionFrame
        if finalized:
            # Do any segments have `is_eou` set to True?
            if (
                any(segment.get("is_eou", False) for segment in segments)
                and self._finalize_requested
            ):
                self.confirm_finalize()

            # Add the finalized frames
            frames += [TranscriptionFrame(**attr_from_segment(segment)) for segment in segments]

            # Handle the text (for metrics reporting)
            finalized_text = "|".join([s["text"] for s in segments])
            await self._handle_transcription(
                finalized_text, is_final=True, language=segments[0]["language"]
            )

            # Log the frames
            logger.debug(f"{self} finalized transcript: {[f.text for f in frames]}")

        # Return as interim results (unformatted)
        else:
            # Add the interim frames
            frames += [
                InterimTranscriptionFrame(**attr_from_segment(segment)) for segment in segments
            ]

            # Log the frames
            logger.debug(f"{self} interim transcript: {[f.text for f in frames]}")

        # Send the frames
        for frame in frames:
            await self.push_frame(frame)

    # ============================================================================
    # PUBLIC FUNCTIONS
    # ============================================================================

    async def send_message(self, message: AgentClientMessageType | str, **kwargs: Any) -> None:
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
            logger.debug(f"{self} sending message to STT: {payload}")
            self.create_task(self._client.send_message(payload))
        except Exception as e:
            raise RuntimeError(f"{self} error sending message to STT: {e}")

    # ============================================================================
    # METRICS
    # ============================================================================

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechmatics STT supports generation of metrics.
        """
        return True

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Adds audio to the audio buffer and yields None."""
        try:
            if self._client:
                await self._client.send_audio(audio)
            yield None
        except Exception as e:
            yield ErrorFrame(f"Speechmatics error: {e}")
            await self._disconnect()

    # ============================================================================
    # HELPERS
    # ============================================================================

    def _language_to_speechmatics_language(self, language: Language) -> str:
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
        result = resolve_language(language, BASE_LANGUAGES, use_base_code=True)

        # Fail if language is not supported
        if not result:
            raise ValueError(f"Unsupported language: {language}")

        # Return the language code
        return result

    def _locale_to_speechmatics_locale(self, base_code: str, locale: Language) -> str | None:
        """Convert a Language enum to a Speechmatics language / locale code.

        Args:
            base_code: The language code.
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

        # Ensure language code is in the map
        if "-" not in str(locale) or base_code not in LOCALES:
            return None

        # Get the locale code
        result = LOCALES.get(base_code).get(locale, None)

        # Fail if locale is not supported
        if not result:
            logger.warning(f"{self} Unsupported output locale: {locale}, defaulting to {base_code}")

        # Return the locale code
        return result

    def _check_deprecated_args(self, kwargs: dict, params: InputParams) -> None:
        """Check arguments for deprecation and update params if necessary.

        This function will show deprecation warnings for deprecated arguments and
        migrate them to the new location in the params object. If the new location
        is None, the argument is not used.

        Args:
            kwargs: Keyword arguments passed to the constructor.
            params: Input parameters for the service.
        """

        # Show deprecation warnings
        def _deprecation_warning(old: str, new: str | None = None) -> None:
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
            ("output_locale", None),
            ("output_locale_code", None),
            ("enable_partials", None),
            ("max_delay", "max_delay"),
            ("chunk_size", None),
            ("audio_encoding", "audio_encoding"),
            ("end_of_utterance_silence_trigger", "end_of_utterance_silence_trigger"),
            {"enable_speaker_diarization", "enable_diarization"},
            ("text_format", "speaker_active_format"),
            ("max_speakers", "max_speakers"),
            ("transcription_config", None),
            ("enable_vad", None),
            ("end_of_utterance_mode", None),
        ]

        # Show warnings + migrate the arguments
        for old, new in deprecated_args:
            if old in kwargs:
                _deprecation_warning(old, new)
                if kwargs.get(old, None) is not None:
                    params.__setattr__(new, kwargs[old])

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini Live API service implementation.

This module provides real-time conversational AI capabilities using Google's
Gemini Live API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.
"""

import asyncio
import base64
import io
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    InputTextRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.frames import LLMSearchOrigin, LLMSearchResponseFrame, LLMSearchResult
from pipecat.services.google.utils import update_google_client_http_options
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.string import match_endofsentence
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_gemini_live, traced_stt

from .file_api import GeminiFileAPI

try:
    from google.genai import Client
    from google.genai.live import AsyncSession
    from google.genai.types import (
        ActivityEnd,
        ActivityStart,
        AudioTranscriptionConfig,
        AutomaticActivityDetection,
        Blob,
        ContextWindowCompressionConfig,
        EndSensitivity,
        FunctionResponse,
        GenerationConfig,
        GroundingMetadata,
        HistoryConfig,
        HttpOptions,
        LiveConnectConfig,
        LiveServerMessage,
        MediaResolution,
        Modality,
        ProactivityConfig,
        RealtimeInputConfig,
        SessionResumptionConfig,
        SlidingWindow,
        SpeechConfig,
        StartSensitivity,
        ThinkingConfig,
        VoiceConfig,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


# Connection management constants
MAX_CONSECUTIVE_FAILURES = 3
CONNECTION_ESTABLISHED_THRESHOLD = 10.0  # seconds


def language_to_gemini_language(language: Language) -> str | None:
    """Maps a Language enum value to a Gemini Live supported language code.

    Source:
    https://ai.google.dev/api/generate-content#MediaResolution

    Args:
        language: The language enum value to convert.

    Returns:
        The Gemini language code string, or None if the language is not supported.
    """
    LANGUAGE_MAP = {
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN_IN: "bn-IN",
        # Chinese (Mandarin)
        Language.CMN: "cmn-CN",
        Language.CMN_CN: "cmn-CN",
        Language.ZH: "cmn-CN",  # Map general Chinese to Mandarin for Gemini
        Language.ZH_CN: "cmn-CN",  # Map Simplified Chinese to Mandarin for Gemini
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # English
        Language.EN: "en-US",  # Default to US English (though not explicitly listed in supported codes)
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain Spanish
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        # French
        Language.FR: "fr-FR",  # Default to France French
        Language.FR_FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_NL: "nl-NL",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese (Brazil)
        Language.PT_BR: "pt-BR",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class GeminiModalities(StrEnum):
    """Supported modalities for Gemini Live.

    Parameters:
        TEXT: Text responses.
        AUDIO: Audio responses.
    """

    TEXT = "TEXT"
    AUDIO = "AUDIO"


class GeminiMediaResolution(StrEnum):
    """Media resolution options for Gemini Live.

    Parameters:
        UNSPECIFIED: Use default resolution setting.
        LOW: Low resolution with 64 tokens.
        MEDIUM: Medium resolution with 256 tokens.
        HIGH: High resolution with zoomed reframing and 256 tokens.
    """

    UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"  # Use default
    LOW = "MEDIA_RESOLUTION_LOW"  # 64 tokens
    MEDIUM = "MEDIA_RESOLUTION_MEDIUM"  # 256 tokens
    HIGH = "MEDIA_RESOLUTION_HIGH"  # Zoomed reframing with 256 tokens


class GeminiVADParams(BaseModel):
    """Voice Activity Detection parameters for Gemini Live.

    Parameters:
        disabled: Whether to disable VAD. Defaults to None (server-side VAD is enabled).
        start_sensitivity: Sensitivity for speech start detection. Defaults to None.
        end_sensitivity: Sensitivity for speech end detection. Defaults to None.
        prefix_padding_ms: Prefix padding in milliseconds. Defaults to None.
        silence_duration_ms: Silence duration threshold in milliseconds. Defaults to None.
    """

    disabled: bool | None = Field(default=None)
    start_sensitivity: StartSensitivity | None = Field(default=None)
    end_sensitivity: EndSensitivity | None = Field(default=None)
    prefix_padding_ms: int | None = Field(default=None)
    silence_duration_ms: int | None = Field(default=None)


class ContextWindowCompressionParams(BaseModel):
    """Parameters for context window compression in Gemini Live.

    Parameters:
        enabled: Whether compression is enabled. Defaults to False.
        trigger_tokens: Token count to trigger compression. None uses 80% of context window.
    """

    enabled: bool = Field(default=False)
    trigger_tokens: int | None = Field(default=None)  # None = use default (80% of context window)


class InputParams(BaseModel):
    """Input parameters for Gemini Live generation.

    .. deprecated:: 0.0.105
        Use ``GeminiLiveLLMService.Settings`` instead.

    Parameters:
        frequency_penalty: Frequency penalty for generation (0.0-2.0). Defaults to None.
        max_tokens: Maximum tokens to generate. Must be >= 1. Defaults to 4096.
        presence_penalty: Presence penalty for generation (0.0-2.0). Defaults to None.
        temperature: Sampling temperature (0.0-2.0). Defaults to None.
        top_k: Top-k sampling parameter. Must be >= 0. Defaults to None.
        top_p: Top-p sampling parameter (0.0-1.0). Defaults to None.
        modalities: Response modalities. Defaults to AUDIO.
        language: Language for generation. Defaults to EN_US.
        media_resolution: Media resolution setting. Defaults to UNSPECIFIED.
        vad: Voice activity detection parameters. Defaults to None.
        context_window_compression: Context compression settings. Defaults to None.
        thinking: Thinking settings. Defaults to None.
            Note that these settings may require specifying a model that
            supports them, e.g. "gemini-2.5-flash-native-audio-preview-12-2025".
        enable_affective_dialog: Enable affective dialog, which allows Gemini
            to adapt to expression and tone. Defaults to None.
            Note that these settings may require specifying a model that
            supports them, e.g. "gemini-2.5-flash-native-audio-preview-12-2025".
            Also note that this setting may require specifying an API version that
            supports it, e.g. HttpOptions(api_version="v1alpha").
        proactivity: Proactivity settings, which allows Gemini to proactively
            decide how to behave, such as whether to avoid responding to
            content that is not relevant. Defaults to None.
            Note that these settings may require specifying a model that
            supports them, e.g. "gemini-2.5-flash-native-audio-preview-12-2025".
            Also note that this setting may require specifying an API version that
            supports it, e.g. HttpOptions(api_version="v1alpha").
        extra: Additional parameters. Defaults to empty dict.
    """

    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=4096, ge=1)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    modalities: GeminiModalities | None = Field(default=GeminiModalities.AUDIO)
    language: Language | None = Field(default=Language.EN_US)
    media_resolution: GeminiMediaResolution | None = Field(
        default=GeminiMediaResolution.UNSPECIFIED
    )
    vad: GeminiVADParams | None = Field(default=None)
    context_window_compression: ContextWindowCompressionParams | None = Field(default=None)
    thinking: ThinkingConfig | None = Field(default=None)
    enable_affective_dialog: bool | None = Field(default=None)
    proactivity: ProactivityConfig | None = Field(default=None)
    extra: dict[str, Any] | None = Field(default_factory=dict)


@dataclass
class GeminiLiveLLMSettings(LLMSettings):
    """Settings for GeminiLiveLLMService.

    Parameters:
        voice: TTS voice identifier (e.g. ``"Charon"``).
        modalities: Response modalities.
        language: Language for generation.
        media_resolution: Media resolution setting.
        vad: Voice activity detection parameters.
        context_window_compression: Context window compression configuration.
        thinking: Thinking configuration.
        enable_affective_dialog: Whether to enable affective dialog.
        proactivity: Proactivity configuration.
    """

    voice: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    modalities: GeminiModalities | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language: Language | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    media_resolution: GeminiMediaResolution | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad: GeminiVADParams | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    context_window_compression: ContextWindowCompressionParams | dict | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    thinking: ThinkingConfig | dict | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_affective_dialog: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    proactivity: ProactivityConfig | dict | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GeminiLiveLLMService(LLMService):
    """Provides access to Google's Gemini Live API.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage.
    """

    Settings = GeminiLiveLLMSettings
    _settings: Settings

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    @property
    def _is_gemini_3(self) -> bool:
        """Check if the current model is a Gemini 3.x model."""
        return "gemini-3" in (self._settings.model or "")

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: str | None = None,
        tools: list[dict] | ToolsSchema | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        inference_on_context_initialization: bool = True,
        file_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/files",
        http_options: HttpOptions | None = None,
        **kwargs,
    ):
        """Initialize the Gemini Live LLM service.

        Args:
            api_key: Google AI API key for authentication.
            model: Model identifier to use.

                .. deprecated:: 0.0.105
                    Use ``settings=GeminiLiveLLMService.Settings(model=...)`` instead.

            voice_id: TTS voice identifier. Defaults to "Charon".

                .. deprecated:: 0.0.105
                    Use ``settings=GeminiLiveLLMService.Settings(voice=...)`` instead.
            start_audio_paused: Whether to start with audio input paused. Defaults to False.
            start_video_paused: Whether to start with video input paused. Defaults to False.
            system_instruction: System prompt for the model. Defaults to None.
            tools: Tools/functions available to the model. Defaults to None.
            params: Configuration parameters for the model.

                .. deprecated:: 0.0.105
                    Use ``settings=GeminiLiveLLMService.Settings(...)`` instead.

            settings: Gemini Live LLM settings. If provided together with deprecated
                top-level parameters, the ``settings`` values take precedence.
            inference_on_context_initialization: Whether to generate a response when context
                is first set. Defaults to True.
            file_api_base_url: Base URL for the Gemini File API. Defaults to the official endpoint.
            http_options: HTTP options for the client.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="models/gemini-2.5-flash-native-audio-preview-12-2025",
            system_instruction=system_instruction,
            voice="Charon",
            frequency_penalty=None,
            max_tokens=4096,
            presence_penalty=None,
            temperature=None,
            top_k=None,
            top_p=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            modalities=GeminiModalities.AUDIO,
            language="en-US",
            media_resolution=GeminiMediaResolution.UNSPECIFIED,
            vad=None,
            context_window_compression={},
            thinking={},
            enable_affective_dialog=False,
            proactivity={},
            extra={},
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id != "Charon":
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.frequency_penalty = params.frequency_penalty
                default_settings.max_tokens = params.max_tokens
                default_settings.presence_penalty = params.presence_penalty
                default_settings.temperature = params.temperature
                default_settings.top_k = params.top_k
                default_settings.top_p = params.top_p
                default_settings.modalities = params.modalities
                default_settings.language = (
                    language_to_gemini_language(params.language) if params.language else "en-US"
                )
                default_settings.media_resolution = params.media_resolution
                default_settings.vad = params.vad
                default_settings.context_window_compression = (
                    params.context_window_compression.model_dump()
                    if params.context_window_compression
                    else {}
                )
                default_settings.thinking = params.thinking or {}
                default_settings.enable_affective_dialog = params.enable_affective_dialog or False
                default_settings.proactivity = params.proactivity or {}
                if isinstance(params.extra, dict):
                    default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Warn if user requested TEXT modality
        if default_settings.modalities == GeminiModalities.TEXT:
            logger.warning(
                f"Modality {default_settings.modalities.value!r} may not be supported by recent "
                "Gemini Live models."
            )

        super().__init__(
            settings=default_settings,
            **kwargs,
        )

        self._last_sent_time = 0

        self._system_instruction_from_init = self._settings.system_instruction
        self._tools_from_init = tools
        self._inference_on_context_initialization = inference_on_context_initialization
        self._needs_initial_turn_complete_message = False

        self._audio_input_paused = start_audio_paused
        self._video_input_paused = start_video_paused
        self._ready_for_realtime_input = False
        self._context = None
        self._api_key = api_key
        self._http_options = update_google_client_http_options(http_options)
        self._session: AsyncSession = None
        self._connection_task = None

        self._disconnecting = False
        self._run_llm_when_session_ready = False

        self._user_is_speaking = False
        self._bot_is_responding = False
        self._user_audio_buffer = bytearray()
        self._user_transcription_buffer = ""
        self._last_transcription_sent = ""
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""
        self._llm_output_buffer = ""
        self._transcription_timeout_task = None

        self._sample_rate = 24000

        self._language = self._settings.language
        self._language_code = (
            language_to_gemini_language(self._settings.language)
            if self._settings.language
            else "en-US"
        )
        self._vad_disabled = bool(self._settings.vad and self._settings.vad.disabled)

        # Reconnection tracking
        self._consecutive_failures = 0
        self._connection_start_time = None

        self._file_api_base_url = file_api_base_url
        self._file_api: GeminiFileAPI | None = None

        # Grounding metadata tracking
        self._search_result_buffer = ""
        self._accumulated_grounding_metadata = None

        # Session resumption
        self._session_resumption_handle: str | None = None

        # Bookkeeping for ending gracefully (i.e. after the bot is finished)
        self._end_frame_pending_bot_turn_finished: EndFrame | None = None
        self._end_frame_deferral_timeout_task: asyncio.Task | None = None

        # Initialize the API client. Subclasses can override this if needed.
        self.create_client()

        # Bookkeeping for tool calls
        self._completed_tool_calls = set()

    def create_client(self):
        """Create the Gemini API client instance. Subclasses can override this."""
        self._client = Client(api_key=self._api_key, http_options=self._http_options)

    @property
    def file_api(self) -> GeminiFileAPI:
        """Get the Gemini File API client instance. Subclasses can override this.

        Returns:
            The Gemini File API client.
        """
        if not self._file_api:
            self._file_api = GeminiFileAPI(api_key=self._api_key, base_url=self._file_api_base_url)
        return self._file_api

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True as Gemini Live supports token usage metrics.
        """
        return True

    async def _update_settings(self, delta: LLMSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored but not applied to the active connection.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # TODO: someday we could reconnect here to apply updated settings.
        # Code might look something like the below:
        # await self._disconnect()
        # await self._connect()

        self._warn_unhandled_updated_settings(changed)

        return changed

    def set_audio_input_paused(self, paused: bool):
        """Set the audio input pause state.

        Args:
            paused: Whether to pause audio input.
        """
        self._audio_input_paused = paused

    def set_video_input_paused(self, paused: bool):
        """Set the video input pause state.

        Args:
            paused: Whether to pause video input.
        """
        self._video_input_paused = paused

    def set_model_modalities(self, modalities: GeminiModalities):
        """Set the model response modalities.

        Args:
            modalities: The modalities to use for responses.
        """
        if modalities == GeminiModalities.TEXT:
            logger.warning(
                f"Modality {modalities.value!r} may not be supported by recent Gemini Live models."
            )
        self._settings.modalities = modalities

    def set_language(self, language: Language):
        """Set the language for generation.

        Args:
            language: The language to use for generation.
        """
        self._language = language
        self._language_code = language_to_gemini_language(language) or "en-US"
        self._settings.language = self._language_code
        logger.info(f"Set Gemini language to: {self._language_code}")

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        """Start the service and establish connection.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connections.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close connections.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    #
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        if self._bot_is_responding:
            await self._set_bot_is_responding(False)
            if self._settings.modalities == GeminiModalities.AUDIO:
                await self.push_frame(TTSStoppedFrame())
            # Do not send LLMFullResponseEndFrame here - an interruption
            # already tells the assistant context aggregator that the response
            # is over.

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        if self._vad_disabled and self._session and self._ready_for_realtime_input:
            try:
                await self._session.send_realtime_input(activity_start=ActivityStart())
            except Exception as e:
                await self._handle_send_error(e)

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        self._user_audio_buffer = bytearray()
        await self.start_ttfb_metrics()
        if self._vad_disabled and self._session and self._ready_for_realtime_input:
            try:
                await self._session.send_realtime_input(activity_end=ActivityEnd())
            except Exception as e:
                await self._handle_send_error(e)
        if self._needs_initial_turn_complete_message:
            self._needs_initial_turn_complete_message = False
            # NOTE: without this, the model ignores the context it's been
            # seeded with before the user started speaking
            await self._session.send_client_content(turn_complete=True)

    #
    # frame processing
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames for the Gemini Live service.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        # Defer EndFrame handling until after the bot turn is finished
        if isinstance(frame, EndFrame):
            if self._bot_is_responding:
                logger.debug("Deferring handling EndFrame until bot turn is finished")
                self._end_frame_pending_bot_turn_finished = frame
                self._create_end_frame_deferral_timeout()
                return

        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputTextRawFrame):
            await self._send_user_text(frame.text)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputImageRawFrame):
            await self._send_user_video(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Ignore this frame. Use the serverContent API message instead
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore this frame. Use the serverContent.turnComplete API message
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMMessagesAppendFrame):
            # NOTE: handling LLMMessagesAppendFrame here in the LLMService is
            # unusual - typically this would be handled in the user context
            # aggregator. Leaving this handling here so that legacy user code
            # that uses this frame *without* a user context aggregator to kick
            # off a conversation still works (we used to have an example that
            # did that).
            await self._create_single_response(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            # TODO: implement runtime tool updates for Gemini Live.
            pass
        else:
            await self.push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        if not self._context:
            # We got our initial context
            self._context = context

            # Reconnect if context changes the effective system instruction
            # or tools compared to the initial connection (which used the
            # init-provided values). Note that the determination of "effective"
            # system instruction is delegated to the adapter, which still
            # chooses the init-provided value if there is one.
            adapter: GeminiLLMAdapter = self.get_llm_adapter()
            params = adapter.get_llm_invocation_params(
                self._context, system_instruction=self._system_instruction_from_init
            )
            system_instruction = params["system_instruction"]
            tools = params["tools"]
            system_instruction_changed = system_instruction != self._system_instruction_from_init
            if tools and self._tools_from_init:
                logger.warning(
                    "Tools provided both at init time and in context; using context-provided value."
                )
            # For tools we simply check presence rather than diffing against
            # init-provided tools, assuming that if context provides tools
            # they warrant a reconnect.
            if system_instruction_changed or tools:
                await self._reconnect()

            # Initialize our bookkeeping of already-completed tool calls in
            # the context
            await self._process_completed_function_calls(send_new_results=False)

            # Create initial response if needed, based on conversation history
            # in context.
            # (If the context has no messages but we do have a system
            # instruction — meaning it was provided at init time — doctor our
            # context now so that we'll have something to send to the service
            # to trigger a response).
            messages = params["messages"]
            if not messages and self._inference_on_context_initialization:
                if self._system_instruction_from_init:
                    logger.debug(
                        "No messages found in initial context; seeding with system instruction to trigger bot response."
                    )
                    self._context.add_message(
                        {"role": "system", "content": self._system_instruction_from_init}
                    )
                else:
                    logger.warning(
                        "No messages found in initial context; cannot trigger initial bot response without messages or system instruction."
                    )
            await self._create_initial_response()
        else:
            # We got an updated context.
            self._context = context

            # Here we assume that the updated context will contain either:
            # - new messages (that the Gemini Live service, with its own
            #   context management, is already aware of), or
            # - tool call results (that we need to tell the remote service
            #   about).
            # (In the future, we could do more sophisticated diffing here,
            # which would enable the user to programmatically manipulate the
            # context).

            # Send results for newly-completed function calls, if any.
            await self._process_completed_function_calls(send_new_results=True)

    async def _process_completed_function_calls(self, send_new_results: bool):
        # Check for set of completed function calls in the context
        adapter: GeminiLLMAdapter = self.get_llm_adapter()
        messages = adapter.get_llm_invocation_params(self._context).get("messages", [])
        for message in messages:
            if message.parts:
                for part in message.parts:
                    if part.function_response:
                        tool_call_id = part.function_response.id
                        tool_name = part.function_response.name
                        response = part.function_response.response
                        if (
                            tool_call_id
                            and tool_call_id not in self._completed_tool_calls
                            and response
                            and response.get("value") != "IN_PROGRESS"
                        ):
                            # Found a newly-completed function call - send the result to the service
                            if send_new_results:
                                await self._tool_result(
                                    tool_call_id, tool_name, part.function_response.response
                                )
                            self._completed_tool_calls.add(tool_call_id)

    async def _set_bot_is_responding(self, responding: bool):
        if self._bot_is_responding == responding:
            return

        self._bot_is_responding = responding

        if not self._bot_is_responding and self._end_frame_pending_bot_turn_finished:
            await self._release_deferred_end_frame()

    async def _release_deferred_end_frame(self):
        """Release a deferred EndFrame and cancel the deferral timeout."""
        if self._end_frame_pending_bot_turn_finished:
            self._cancel_end_frame_deferral_timeout()
            self._bot_is_responding = False
            frame = self._end_frame_pending_bot_turn_finished
            self._end_frame_pending_bot_turn_finished = None
            await self.queue_frame(frame)

    # Timeout (in seconds) for the EndFrame deferral. If turn_complete is not
    # received within this time, the EndFrame is released anyway to prevent the
    # pipeline from hanging indefinitely.
    _END_FRAME_DEFERRAL_TIMEOUT_SECS = 30.0

    def _create_end_frame_deferral_timeout(self):
        """Start a timeout that releases the deferred EndFrame if turn_complete never arrives."""
        self._cancel_end_frame_deferral_timeout()

        async def _timeout():
            await asyncio.sleep(self._END_FRAME_DEFERRAL_TIMEOUT_SECS)
            if self._end_frame_pending_bot_turn_finished:
                logger.warning(
                    f"EndFrame deferral timed out after {self._END_FRAME_DEFERRAL_TIMEOUT_SECS}s "
                    "without receiving turn_complete — releasing EndFrame"
                )
                await self._release_deferred_end_frame()

        self._end_frame_deferral_timeout_task = self.create_task(
            _timeout(), "end_frame_deferral_timeout"
        )

    def _cancel_end_frame_deferral_timeout(self):
        """Cancel the EndFrame deferral timeout if active."""
        if (
            self._end_frame_deferral_timeout_task
            and not self._end_frame_deferral_timeout_task.done()
        ):
            self._end_frame_deferral_timeout_task.cancel()
        self._end_frame_deferral_timeout_task = None

    def _get_history_config(self) -> HistoryConfig | None:
        """Return the history config for the Live API connection.

        Subclasses can override this to disable history config (e.g. Vertex AI
        does not support it).
        """
        return HistoryConfig(initial_history_in_client_content=True)

    async def _connect(self, session_resumption_handle: str | None = None):
        """Establish client connection to Gemini Live API."""
        if self._session:
            # Here we assume that if we have a client, we are connected. We
            # handle disconnections in the send/recv code paths.
            return

        if session_resumption_handle:
            logger.info(
                f"Connecting to Gemini service with session_resumption_handle: {session_resumption_handle}"
            )
        else:
            logger.info("Connecting to Gemini service")
        try:
            # Assemble basic configuration
            config = LiveConnectConfig(
                generation_config=GenerationConfig(
                    frequency_penalty=self._settings.frequency_penalty,
                    max_output_tokens=self._settings.max_tokens,
                    presence_penalty=self._settings.presence_penalty,
                    temperature=self._settings.temperature,
                    top_k=self._settings.top_k,
                    top_p=self._settings.top_p,
                    response_modalities=[Modality(self._settings.modalities.value)],
                    speech_config=SpeechConfig(
                        voice_config=VoiceConfig(
                            prebuilt_voice_config={"voice_name": self._settings.voice}
                        ),
                        language_code=self._settings.language,
                    ),
                    media_resolution=MediaResolution(self._settings.media_resolution.value),
                ),
                input_audio_transcription=AudioTranscriptionConfig(),
                output_audio_transcription=AudioTranscriptionConfig(),
                session_resumption=SessionResumptionConfig(handle=session_resumption_handle),
            )

            # Add history config, if supported (not supported by Vertex)
            history_config = self._get_history_config()
            if history_config:
                config.history_config = history_config

            # Add context window compression to configuration, if enabled
            cwc = self._settings.context_window_compression or {}
            if cwc.get("enabled", False):
                compression_config = ContextWindowCompressionConfig()

                # Add sliding window (always true if compression is enabled)
                compression_config.sliding_window = SlidingWindow()

                # Add trigger_tokens if specified
                trigger_tokens = cwc.get("trigger_tokens")
                if trigger_tokens is not None:
                    compression_config.trigger_tokens = trigger_tokens

                config.context_window_compression = compression_config

            # Add thinking configuration to configuration, if provided
            if self._settings.thinking:
                config.thinking_config = self._settings.thinking

            # Add affective dialog setting, if provided
            if self._settings.enable_affective_dialog:
                config.enable_affective_dialog = self._settings.enable_affective_dialog

            # Add proactivity configuration to configuration, if provided
            if self._settings.proactivity:
                config.proactivity = self._settings.proactivity

            # Add VAD configuration to configuration, if provided
            if self._settings.vad:
                vad_config = AutomaticActivityDetection()
                vad_params = self._settings.vad
                has_vad_settings = False

                # Only add parameters that are explicitly set
                if vad_params.disabled is not None:
                    vad_config.disabled = vad_params.disabled
                    has_vad_settings = True

                if vad_params.start_sensitivity:
                    vad_config.start_of_speech_sensitivity = vad_params.start_sensitivity
                    has_vad_settings = True

                if vad_params.end_sensitivity:
                    vad_config.end_of_speech_sensitivity = vad_params.end_sensitivity
                    has_vad_settings = True

                if vad_params.prefix_padding_ms is not None:
                    vad_config.prefix_padding_ms = vad_params.prefix_padding_ms
                    has_vad_settings = True

                if vad_params.silence_duration_ms is not None:
                    vad_config.silence_duration_ms = vad_params.silence_duration_ms
                    has_vad_settings = True

                # Only add automatic_activity_detection if we have VAD settings
                if has_vad_settings:
                    config.realtime_input_config = RealtimeInputConfig(
                        automatic_activity_detection=vad_config
                    )

            # Add system instruction and tools to configuration, if provided.
            # These settings from the context take precedence over the ones
            # provided at initialization time.
            adapter: GeminiLLMAdapter = self.get_llm_adapter()
            system_instruction = None
            tools = None
            if self._context:
                params = adapter.get_llm_invocation_params(
                    self._context, system_instruction=self._system_instruction_from_init
                )
                system_instruction = params["system_instruction"]
                tools = params["tools"]
            else:
                system_instruction = self._system_instruction_from_init
            if not tools:
                tools = adapter.from_standard_tools(self._tools_from_init)
            if system_instruction:
                logger.debug(f"Setting system instruction: {system_instruction}")
                config.system_instruction = system_instruction
            if tools:
                logger.debug(f"Setting tools: {tools}")
                config.tools = tools

            # Start the connection
            self._connection_task = self.create_task(self._connection_task_handler(config=config))

        except Exception as e:
            await self.push_error(error_msg=f"Initialization error: {e}", exception=e)

    async def _connection_task_handler(self, config: LiveConnectConfig):
        async with self._client.aio.live.connect(
            model=self._settings.model, config=config
        ) as session:
            logger.info("Connected to Gemini service")

            # Mark connection start time
            self._connection_start_time = time.time()

            await self._handle_session_ready(session)

            while True:
                try:
                    turn = self._session.receive()
                    async for message in turn:
                        # Reset failure counter if connection has been stable
                        self._check_and_reset_failure_counter()

                        # server_content fields are NOT mutually exclusive —
                        # Gemini 3.x can bundle multiple content fields and
                        # turn_complete on the same message, so process the
                        # content-bearing fields before closing the turn.
                        sc = message.server_content
                        if sc and sc.interrupted:
                            # NOTE: while the service triggers interruptions in
                            # the specific case of barge-ins, it does *not*
                            # emit UserStarted/StoppedSpeakingFrames, as the
                            # Gemini Live API does not give us broadly reliable
                            # signals to base those off of. Pipelines that
                            # require turn tracking (like those using context
                            # aggregators) still need an independent way to
                            # track turns, such as local Silero VAD in
                            # combination with the context aggregator default
                            # turn strategies.
                            logger.debug("Gemini VAD: interrupted signal received")
                            await self.broadcast_interruption()
                        if sc and sc.model_turn:
                            await self._handle_msg_model_turn(message)
                        if sc and sc.input_transcription:
                            await self._handle_msg_input_transcription(message)
                        if sc and sc.output_transcription:
                            await self._handle_msg_output_transcription(message)
                        if (
                            sc
                            and sc.grounding_metadata
                            and not sc.model_turn
                            and not sc.output_transcription
                        ):
                            # model_turn/output_transcription already defer
                            # bundled grounding metadata to turn_complete.
                            await self._handle_msg_grounding_metadata(message)
                        if sc and sc.turn_complete:
                            if not message.usage_metadata:
                                logger.warning("Received turn_complete without usage_metadata")
                            await self._handle_msg_turn_complete(message)
                            if message.usage_metadata:
                                await self._handle_msg_usage_metadata(message)
                        if message.tool_call:
                            await self._handle_msg_tool_call(message)
                        if message.session_resumption_update:
                            self._handle_msg_resumption_update(message)
                except Exception as e:
                    if not self._disconnecting:
                        should_reconnect = await self._handle_connection_error(e)
                        if should_reconnect:
                            await self._reconnect()
                            return  # Exit this connection handler, _reconnect will start a new one
                    break

    def _check_and_reset_failure_counter(self):
        """Check if connection has been stable long enough to reset the failure counter.

        If the connection has been active for longer than the established threshold
        and there are accumulated failures, reset the counter to 0.
        """
        if (
            self._connection_start_time
            and self._consecutive_failures > 0
            and time.time() - self._connection_start_time >= CONNECTION_ESTABLISHED_THRESHOLD
        ):
            logger.info(
                f"Connection stable for {CONNECTION_ESTABLISHED_THRESHOLD}s, "
                f"resetting failure counter from {self._consecutive_failures} to 0"
            )
            self._consecutive_failures = 0

    async def _handle_connection_error(self, error: Exception) -> bool:
        """Handle a connection error and determine if reconnection should be attempted.

        Args:
            error: The exception that caused the connection error.

        Returns:
            True if reconnection should be attempted, False if a fatal error should be pushed.
        """
        self._consecutive_failures += 1
        logger.warning(
            f"Connection error (failure {self._consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {error}"
        )

        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            error_msg = (
                f"Max consecutive failures ({MAX_CONSECUTIVE_FAILURES}) reached, "
                "treating as fatal error"
            )
            await self.push_error(error_msg=error_msg, exception=error)
            return False
        else:
            logger.info(
                f"Attempting reconnection ({self._consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
            )
            return True

    async def _reconnect(self):
        """Reconnect to Gemini Live API."""
        await self._disconnect()
        await self._connect(session_resumption_handle=self._session_resumption_handle)

    async def _disconnect(self):
        """Disconnect from Gemini Live API and clean up resources."""
        logger.info("Disconnecting from Gemini service")
        try:
            self._disconnecting = True
            await self.stop_all_metrics()
            if self._connection_task:
                await self.cancel_task(self._connection_task, timeout=1.0)
                self._connection_task = None
            if self._transcription_timeout_task:
                await self.cancel_task(self._transcription_timeout_task)
                self._transcription_timeout_task = None
            self._cancel_end_frame_deferral_timeout()
            self._end_frame_pending_bot_turn_finished = None
            if self._session:
                await self._session.close()
                self._session = None
            self._completed_tool_calls = set()
            self._ready_for_realtime_input = False
            self._disconnecting = False
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)

    async def _send_user_audio(self, frame):
        """Send user audio frame to Gemini Live API."""
        if (
            self._audio_input_paused
            or self._disconnecting
            or not self._session
            or not self._ready_for_realtime_input
        ):
            return

        # Send all audio to Gemini
        try:
            await self._session.send_realtime_input(
                audio=Blob(data=frame.audio, mime_type=f"audio/pcm;rate={frame.sample_rate}")
            )
        except Exception as e:
            await self._handle_send_error(e)

        # Manage a buffer of audio to use for transcription
        audio = frame.audio
        if self._user_is_speaking:
            self._user_audio_buffer.extend(audio)
        else:
            # Keep 1/2 second of audio in the buffer even when not speaking.
            self._user_audio_buffer.extend(audio)
            length = int((frame.sample_rate * frame.num_channels * 2) * 0.5)
            self._user_audio_buffer = self._user_audio_buffer[-length:]

    async def _send_user_text(self, text: str):
        """Send user text via Gemini Live API's realtime input stream.

        This method sends text through the realtimeInput stream (via TextInputMessage)
        rather than the clientContent stream. This ensures text input is synchronized
        with audio and video inputs, preventing temporal misalignment that can occur
        when different modalities are processed through separate API pathways.

        For realtimeInput, turn completion is automatically inferred by the API based
        on user activity, so no explicit turnComplete signal is needed.

        Args:
            text: The text to send as user input.
        """
        if self._disconnecting or not self._session or not self._ready_for_realtime_input:
            return

        try:
            await self._session.send_realtime_input(text=text)
        except Exception as e:
            await self._handle_send_error(e)

    async def _send_user_video(self, frame):
        """Send user video frame to Gemini Live API."""
        if (
            self._video_input_paused
            or self._disconnecting
            or not self._session
            or not self._ready_for_realtime_input
        ):
            return

        now = time.time()
        if now - self._last_sent_time < 1:
            return  # Ignore if less than 1 second has passed

        self._last_sent_time = now  # Update last sent time
        logger.trace(f"Sending video frame to Gemini: {frame}")

        buffer = io.BytesIO()
        Image.frombytes(frame.format, frame.size, frame.image).save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            await self._session.send_realtime_input(video=Blob(data=data, mime_type="image/jpeg"))
        except Exception as e:
            await self._handle_send_error(e)

    async def _create_initial_response(self):
        """Create initial response based on context history."""
        if self._disconnecting:
            return

        if not self._session:
            self._run_llm_when_session_ready = True
            return

        adapter: GeminiLLMAdapter = self.get_llm_adapter()
        messages = adapter.get_llm_invocation_params(self._context).get("messages", [])
        if not messages:
            # No messages to seed convo with, so we're ready for realtime input right away
            self._ready_for_realtime_input = True
            return

        logger.debug(f"Creating initial response: {messages}")

        await self.start_ttfb_metrics()

        try:
            await self._session.send_client_content(
                turns=messages, turn_complete=self._inference_on_context_initialization
            )
            # Gemini 3.x wants turn_complete=True, but also won't run inference without a realtime input
            if self._is_gemini_3 and self._inference_on_context_initialization:
                await self._session.send_realtime_input(text=" ")
        except Exception as e:
            await self._handle_send_error(e)

        # If we're generating a response right away upon initializing
        # conversation history, set a flag saying that we'll need a turn
        # complete message when the user stops speaking.
        # This is a quirky workaround, and not one that Gemini 3 needs.
        if not self._inference_on_context_initialization and not self._is_gemini_3:
            self._needs_initial_turn_complete_message = True

        self._ready_for_realtime_input = True

    async def _create_single_response(self, messages_list):
        """Create a single response from a list of messages.

        This is only here to support the very specific 'legacy' scenario of
        kicking off a conversation using LLMMessagesAppendFrame when there's no
        context aggregators in the pipeline (see process_frame for more details).
        """
        if self._disconnecting or not self._session:
            return

        # Create a throwaway context just for the purpose of getting messages
        # in the right format
        context = LLMContext(messages=messages_list)
        adapter: GeminiLLMAdapter = self.get_llm_adapter()
        messages = adapter.get_llm_invocation_params(context).get("messages", [])

        if not messages:
            return

        logger.debug(f"Creating response: {messages}")

        await self.start_ttfb_metrics()

        try:
            await self._session.send_client_content(turns=messages, turn_complete=True)
            # Gemini 3.x wants turn_complete=True, but also won't run inference without a realtime input
            if self._is_gemini_3:
                await self._session.send_realtime_input(text=" ")
        except Exception as e:
            await self._handle_send_error(e)

    @traced_gemini_live(operation="llm_tool_result")
    async def _tool_result(
        self, tool_call_id: str, tool_name: str, tool_result_message: dict[str, Any]
    ):
        """Send tool result back to the API."""
        if self._disconnecting or not self._session:
            return

        # For now we're shoving the name into the tool_call_id field, so this
        # will work until we revisit that.
        response = FunctionResponse(name=tool_name, id=tool_call_id, response=tool_result_message)

        try:
            await self._session.send_tool_response(function_responses=response)
        except Exception as e:
            await self._handle_send_error(e)

    @traced_gemini_live(operation="llm_setup")
    async def _handle_session_ready(self, session: AsyncSession):
        """Handle the session being ready."""
        self._session = session
        if self._run_llm_when_session_ready:
            # Initial connection: context arrived before session was ready.
            self._run_llm_when_session_ready = False
            await self._create_initial_response()
        elif self._session_resumption_handle:
            # Reconnect with session resumption: the server will restore
            # session state, so we can accept realtime input right away.
            self._ready_for_realtime_input = True
        elif self._context:
            # Reconnect without session resumption (e.g. error occurred
            # before server sent a resumption handle): re-seed conversation
            # history so the new session retains full context before
            # accepting input.
            try:
                adapter = self.get_llm_adapter()
                messages = adapter.get_llm_invocation_params(self._context).get("messages", [])
                if messages:
                    logger.info(
                        f"Re-seeding {len(messages)} conversation turns after reconnect"
                    )
                    await self._session.send_client_content(
                        turns=messages, turn_complete=False
                    )
            except Exception as e:
                logger.error(f"Failed to re-seed conversation on reconnect: {e}")
            self._ready_for_realtime_input = True
        else:
            # Initial connection: session is ready before context has
            # arrived. Nothing to do — _handle_context will call
            # _create_initial_response when the context arrives.
            pass

    async def _handle_msg_model_turn(self, msg: LiveServerMessage):
        """Handle the model turn message."""
        part = msg.server_content.model_turn.parts[0]
        if not part:
            return

        await self.stop_ttfb_metrics()

        # part.text is added when `modalities` is set to TEXT; otherwise, it's None
        text = part.text
        if text:
            if not self._bot_is_responding:
                # Update bot responding state and send service start frame
                # (AUDIO modality case)
                await self._set_bot_is_responding(True)
                await self.push_frame(LLMFullResponseStartFrame())

            # Check if this is a thought
            if part.thought:
                # Gemini Live emits fully-formed thoughts rather than chunks,
                # so bracket each thought in start/end frames
                await self.push_frame(LLMThoughtStartFrame())
                await self.push_frame(LLMThoughtTextFrame(text))
                await self.push_frame(LLMThoughtEndFrame())
            else:
                # Regular text response
                self._bot_text_buffer += text
                self._search_result_buffer += text  # Also accumulate for grounding
                frame = LLMTextFrame(text=text)
                await self.push_frame(frame)

        # Check for grounding metadata in server content
        if msg.server_content and msg.server_content.grounding_metadata:
            self._accumulated_grounding_metadata = msg.server_content.grounding_metadata

        # If we have no audio, stop here.
        # All logic below this point pertains to the AUDIO modality.
        inline_data = part.inline_data
        if not inline_data:
            return

        # Check if mime type matches expected format
        expected_mime_type = f"audio/pcm;rate={self._sample_rate}"
        if inline_data.mime_type == expected_mime_type:
            # Perfect match, continue processing
            pass
        elif inline_data.mime_type == "audio/pcm":
            # Sample rate not provided in mime type, assume default
            if not hasattr(self, "_sample_rate_warning_logged"):
                logger.warning(
                    f"Sample rate not provided in mime type '{inline_data.mime_type}', assuming rate of {self._sample_rate}"
                )
                self._sample_rate_warning_logged = True
        else:
            # Unrecognized format
            logger.warning(f"Unrecognized server_content format {inline_data.mime_type}")
            return

        audio = inline_data.data
        if not audio:
            return

        # Update bot responding state and send service start frames
        # (AUDIO modality case)
        if not self._bot_is_responding:
            await self._set_bot_is_responding(True)
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(LLMFullResponseStartFrame())

        self._bot_audio_buffer.extend(audio)
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=self._sample_rate,
            num_channels=1,
        )
        await self.push_frame(frame)

    @traced_gemini_live(operation="llm_tool_call")
    async def _handle_msg_tool_call(self, message: LiveServerMessage):
        """Handle tool call messages."""
        function_calls = message.tool_call.function_calls
        if not function_calls:
            return
        if not self._context:
            logger.error("Function calls are not supported without a context object.")

        function_calls_llm = [
            FunctionCallFromLLM(
                context=self._context,
                tool_call_id=(
                    # NOTE: when using Vertex AI we don't get server-provided
                    # tool call IDs here
                    f.id or str(uuid.uuid4())
                ),
                function_name=f.name,
                arguments=f.args,
            )
            for f in function_calls
        ]

        await self.run_function_calls(function_calls_llm)

    @traced_gemini_live(operation="llm_response")
    async def _handle_msg_turn_complete(self, message: LiveServerMessage):
        """Handle the turn complete message."""
        text = self._bot_text_buffer

        # Trace the complete LLM response (this will be handled by the decorator)
        # The decorator will extract the output text and usage metadata from the message

        self._bot_text_buffer = ""
        self._llm_output_buffer = ""

        # Process grounding metadata if we have accumulated any
        if self._accumulated_grounding_metadata:
            await self._process_grounding_metadata(
                self._accumulated_grounding_metadata, self._search_result_buffer
            )

        # Reset grounding tracking for next response
        self._search_result_buffer = ""
        self._accumulated_grounding_metadata = None

        if self._bot_is_responding:
            await self._set_bot_is_responding(False)
            if not text:
                # AUDIO modality case
                await self.push_frame(TTSStoppedFrame())
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                # TEXT modality case
                await self.push_frame(LLMFullResponseEndFrame())

    @traced_stt
    async def _handle_user_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _push_user_transcription(self, text: str, result: LiveServerMessage | None = None):
        """Push a user transcription frame upstream.

        Helper method to ensure consistent handling of user transcriptions
        from both punctuation-based and timeout-based paths.

        Args:
            text: The transcription text to push
            result: Optional LiveServerMessage that triggered this transcription
        """
        await self._handle_user_transcription(text, True, self._settings.language)
        await self.push_frame(
            TranscriptionFrame(
                text=text,
                user_id="",
                timestamp=time_now_iso8601(),
                result=result,
            ),
            FrameDirection.UPSTREAM,
        )

    async def _transcription_timeout_handler(self):
        """Handle timeout for user transcription buffer.

        If no new transcription messages arrive within the timeout period,
        flush any remaining text in the buffer as a complete sentence.
        """
        try:
            # Wait for timeout period (0.5 seconds)
            await asyncio.sleep(0.5)

            # If we still have buffered text after timeout, flush it
            if self._user_transcription_buffer:
                logger.trace(
                    f"[Transcription:user:timeout] Flushing buffer: [{self._user_transcription_buffer}]"
                )
                complete_sentence = self._user_transcription_buffer
                self._user_transcription_buffer = ""

                await self._push_user_transcription(complete_sentence, result=None)
        except asyncio.CancelledError:
            # Task was cancelled because new transcription arrived. This is expected
            # when back to back transcription messages arrive.
            logger.trace("Transcription timeout task cancelled (new text arrived)")
            raise

    async def _handle_msg_input_transcription(self, message: LiveServerMessage):
        """Handle the input transcription message.

        Gemini Live sends user transcriptions in either single words or multi-word
        phrases. As a result, we have to aggregate the input transcription. This handler
        aggregates into sentences, splitting on the end of sentence markers. If no
        punctuation arrives within a timeout period, the buffer is flushed automatically.
        """
        if not message.server_content.input_transcription:
            return

        text = message.server_content.input_transcription.text

        if not text:
            return

        # Cancel any existing timeout task since we received new text
        if self._transcription_timeout_task:
            await self.cancel_task(self._transcription_timeout_task)
            self._transcription_timeout_task = None

        # Strip leading space from sentence starts if buffer is empty
        if text.startswith(" ") and not self._user_transcription_buffer:
            text = text.lstrip()

        # Accumulate text in the buffer
        self._user_transcription_buffer += text

        # Check for complete sentences
        while True:
            eos_end_marker = match_endofsentence(self._user_transcription_buffer)
            if not eos_end_marker:
                break

            # Extract the complete sentence
            complete_sentence = self._user_transcription_buffer[:eos_end_marker]
            # Keep the remainder for the next chunk
            self._user_transcription_buffer = self._user_transcription_buffer[eos_end_marker:]

            # Send a TranscriptionFrame with the complete sentence
            logger.debug(f"[Transcription:user] [{complete_sentence}]")
            await self._push_user_transcription(complete_sentence, result=message)

        # If there's still text in the buffer (no end-of-sentence marker found),
        # start a timeout task to flush it later
        if self._user_transcription_buffer:
            self._transcription_timeout_task = self.create_task(
                self._transcription_timeout_handler()
            )
            # Let the event loop schedule the taks before it gets cancelled.
            await asyncio.sleep(0)

    async def _handle_msg_output_transcription(self, message: LiveServerMessage):
        """Handle the output transcription message."""
        if not message.server_content.output_transcription:
            return

        # This is the output transcription text when modalities is set to AUDIO.
        # In this case, we push TTSTextFrame to be handled by the downstream
        # assistant context aggregator.
        text = message.server_content.output_transcription.text

        if not text:
            return

        # Accumulate text for grounding as well
        self._search_result_buffer += text

        # Check for grounding metadata in server content
        if message.server_content and message.server_content.grounding_metadata:
            self._accumulated_grounding_metadata = message.server_content.grounding_metadata
        # Collect text for tracing
        self._llm_output_buffer += text

        # NOTE: Shoot. When using Vertex AI, output transcription messages
        # arrive *before* the model_turn messages with audio, so we need to
        # handle sending TTSStartedFrame and LLMFullResponseStartFrame here as
        # well. These messages also contain much *more* text (it looks further
        # ahead). That means that on an interruption our recorded context will
        # contain some text that was actually never spoken.
        if not self._bot_is_responding:
            await self._set_bot_is_responding(True)
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(LLMFullResponseStartFrame())

        await self._push_output_transcription_text_frames(text)

    async def _push_output_transcription_text_frames(self, text: str):
        # In a typical "cascade" LLM + TTS setup, LLMTextFrames would not
        # proceed beyond the TTS service. Therefore, since a speech-to-speech
        # service like Gemini Live combines both LLM and TTS functionality, you
        # might think we wouldn't need to push LLMTextFrames at all. However,
        # RTVI relies on LLMTextFrames being pushed to trigger its
        # "bot-llm-text" event. So here we push an LLMTextFrame, too, but avoid
        # appending it to context to avoid context message duplication.

        # Push LLMTextFrame
        llm_text_frame = LLMTextFrame(text)
        llm_text_frame.append_to_context = False
        await self.push_frame(llm_text_frame)

        # Push TTSTextFrame
        tts_text_frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        tts_text_frame.includes_inter_frame_spaces = True
        await self.push_frame(tts_text_frame)

    async def _handle_msg_grounding_metadata(self, message: LiveServerMessage):
        """Handle dedicated grounding metadata messages."""
        if message.server_content and message.server_content.grounding_metadata:
            grounding_metadata = message.server_content.grounding_metadata
            # Process the grounding metadata immediately
            await self._process_grounding_metadata(grounding_metadata, self._search_result_buffer)

    async def _process_grounding_metadata(
        self, grounding_metadata: GroundingMetadata, search_result: str = ""
    ):
        """Process grounding metadata and emit LLMSearchResponseFrame."""
        if not grounding_metadata:
            return

        # Extract rendered content for search suggestions
        rendered_content = None
        if (
            grounding_metadata.search_entry_point
            and grounding_metadata.search_entry_point.rendered_content
        ):
            rendered_content = grounding_metadata.search_entry_point.rendered_content

        # Convert grounding chunks and supports to LLMSearchOrigin format
        origins = []

        if grounding_metadata.grounding_chunks and grounding_metadata.grounding_supports:
            # Create a mapping of chunk indices to origins
            chunk_to_origin: dict[int, LLMSearchOrigin] = {}

            for index, chunk in enumerate(grounding_metadata.grounding_chunks):
                if chunk.web:
                    origin = LLMSearchOrigin(
                        site_uri=chunk.web.uri, site_title=chunk.web.title, results=[]
                    )
                    chunk_to_origin[index] = origin
                    origins.append(origin)

            # Add grounding support results to the appropriate origins
            for support in grounding_metadata.grounding_supports:
                if support.segment and support.grounding_chunk_indices:
                    text = support.segment.text or ""
                    confidence_scores = support.confidence_scores or []

                    # Add this result to all origins referenced by this support
                    for chunk_index in support.grounding_chunk_indices:
                        if chunk_index in chunk_to_origin:
                            result = LLMSearchResult(text=text, confidence=confidence_scores)
                            chunk_to_origin[chunk_index].results.append(result)

        # Create and push the search response frame
        search_frame = LLMSearchResponseFrame(
            search_result=search_result, origins=origins, rendered_content=rendered_content
        )

        await self.push_frame(search_frame)

    async def _handle_msg_usage_metadata(self, message: LiveServerMessage):
        """Handle the usage metadata message."""
        if not message.usage_metadata:
            return

        usage = message.usage_metadata

        # Ensure we have valid integers for all token counts
        prompt_tokens = usage.prompt_token_count or 0
        completion_tokens = usage.response_token_count or 0
        total_tokens = usage.total_token_count or (prompt_tokens + completion_tokens)

        tokens = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_read_input_tokens=usage.cached_content_token_count,
            reasoning_tokens=usage.thoughts_token_count,
        )

        await self.start_llm_usage_metrics(tokens)

    def _handle_msg_resumption_update(self, message: LiveServerMessage):
        update = message.session_resumption_update
        if update.resumable and update.new_handle:
            self._session_resumption_handle = update.new_handle

    async def _handle_send_error(self, error: Exception):
        # Ignore "expected" errors that may have occurred for messages that
        # were in-flight when a disconnection occurred.
        if self._disconnecting or not self._session:
            return

        # In server-to-server contexts, a WebSocket error should be quite rare.
        # Given how hard it is to recover from a send-side error with proper
        # state management, and that exponential backoff for retries can have
        # cost/stability implications for a service cluster, let's just treat a
        # send-side error as fatal.
        await self.push_error(error_msg=f"Send error: {error}")

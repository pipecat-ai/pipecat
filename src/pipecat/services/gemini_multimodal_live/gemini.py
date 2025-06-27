#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini Multimodal Live API service implementation.

This module provides real-time conversational AI capabilities using Google's
Gemini Multimodal Live API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.
"""

import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.string import match_endofsentence
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_gemini_live, traced_stt

from . import events

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


def language_to_gemini_language(language: Language) -> Optional[str]:
    """Maps a Language enum value to a Gemini Live supported language code.

    Source:
    https://ai.google.dev/api/generate-content#MediaResolution

    Args:
        language: The language enum value to convert.

    Returns:
        The Gemini language code string, or None if the language is not supported.
    """
    language_map = {
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
    return language_map.get(language)


class GeminiMultimodalLiveContext(OpenAILLMContext):
    """Extended OpenAI context for Gemini Multimodal Live API.

    Provides Gemini-specific context management including system instruction
    extraction and message format conversion for the Live API.
    """

    @staticmethod
    def upgrade(obj: OpenAILLMContext) -> "GeminiMultimodalLiveContext":
        """Upgrade an OpenAI context to Gemini context.

        Args:
            obj: The OpenAI context to upgrade.

        Returns:
            The upgraded Gemini context instance.
        """
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GeminiMultimodalLiveContext):
            logger.debug(f"Upgrading to Gemini Multimodal Live Context: {obj}")
            obj.__class__ = GeminiMultimodalLiveContext
            obj._restructure_from_openai_messages()
        return obj

    def _restructure_from_openai_messages(self):
        pass

    def extract_system_instructions(self):
        """Extract system instructions from context messages.

        Returns:
            Combined system instruction text from all system messages.
        """
        system_instruction = ""
        for item in self.messages:
            if item.get("role") == "system":
                content = item.get("content", "")
                if content:
                    if system_instruction and not system_instruction.endswith("\n"):
                        system_instruction += "\n"
                    system_instruction += str(content)
        return system_instruction

    def get_messages_for_initializing_history(self):
        """Get messages formatted for Gemini history initialization.

        Returns:
            List of messages in Gemini format for conversation history.
        """
        messages = []
        for item in self.messages:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        return messages


class GeminiMultimodalLiveUserContextAggregator(OpenAIUserContextAggregator):
    """User context aggregator for Gemini Multimodal Live.

    Extends OpenAI user aggregator to handle Gemini-specific message passing
    while maintaining compatibility with the standard aggregation pipeline.
    """

    async def process_frame(self, frame, direction):
        """Process incoming frames for user context aggregation.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        await super().process_frame(frame, direction)
        # kind of a hack just to pass the LLMMessagesAppendFrame through, but it's fine for now
        if isinstance(frame, LLMMessagesAppendFrame):
            await self.push_frame(frame, direction)


class GeminiMultimodalLiveAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Assistant context aggregator for Gemini Multimodal Live.

    Handles assistant response aggregation while filtering out LLMTextFrames
    to prevent duplicate context entries, as Gemini Live pushes both
    LLMTextFrames and TTSTextFrames.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames for assistant context aggregation.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        # The LLMAssistantContextAggregator uses TextFrames to aggregate the LLM output,
        # but the GeminiMultimodalLiveAssistantContextAggregator pushes LLMTextFrames and TTSTextFrames. We
        # need to override this proces_frame for LLMTextFrame, so that only the TTSTextFrames
        # are process. This ensures that the context gets only one set of messages.
        if not isinstance(frame, LLMTextFrame):
            await super().process_frame(frame, direction)

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle user image frames.

        Args:
            frame: The user image frame to handle.
        """
        # We don't want to store any images in the context. Revisit this later
        # when the API evolves.
        pass


@dataclass
class GeminiMultimodalLiveContextAggregatorPair:
    """Pair of user and assistant context aggregators for Gemini Multimodal Live.

    Parameters:
        _user: The user context aggregator instance.
        _assistant: The assistant context aggregator instance.
    """

    _user: GeminiMultimodalLiveUserContextAggregator
    _assistant: GeminiMultimodalLiveAssistantContextAggregator

    def user(self) -> GeminiMultimodalLiveUserContextAggregator:
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> GeminiMultimodalLiveAssistantContextAggregator:
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class GeminiMultimodalModalities(Enum):
    """Supported modalities for Gemini Multimodal Live."""

    TEXT = "TEXT"
    AUDIO = "AUDIO"


class GeminiMediaResolution(str, Enum):
    """Media resolution options for Gemini Multimodal Live."""

    UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"  # Use default
    LOW = "MEDIA_RESOLUTION_LOW"  # 64 tokens
    MEDIUM = "MEDIA_RESOLUTION_MEDIUM"  # 256 tokens
    HIGH = "MEDIA_RESOLUTION_HIGH"  # Zoomed reframing with 256 tokens


class GeminiVADParams(BaseModel):
    """Voice Activity Detection parameters for Gemini Live.

    Parameters:
        disabled: Whether to disable VAD. Defaults to None.
        start_sensitivity: Sensitivity for speech start detection. Defaults to None.
        end_sensitivity: Sensitivity for speech end detection. Defaults to None.
        prefix_padding_ms: Prefix padding in milliseconds. Defaults to None.
        silence_duration_ms: Silence duration threshold in milliseconds. Defaults to None.
    """

    disabled: Optional[bool] = Field(default=None)
    start_sensitivity: Optional[events.StartSensitivity] = Field(default=None)
    end_sensitivity: Optional[events.EndSensitivity] = Field(default=None)
    prefix_padding_ms: Optional[int] = Field(default=None)
    silence_duration_ms: Optional[int] = Field(default=None)


class ContextWindowCompressionParams(BaseModel):
    """Parameters for context window compression in Gemini Live.

    Parameters:
        enabled: Whether compression is enabled. Defaults to False.
        trigger_tokens: Token count to trigger compression. None uses 80% of context window.
    """

    enabled: bool = Field(default=False)
    trigger_tokens: Optional[int] = Field(
        default=None
    )  # None = use default (80% of context window)


class InputParams(BaseModel):
    """Input parameters for Gemini Multimodal Live generation.

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
        extra: Additional parameters. Defaults to empty dict.
    """

    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    modalities: Optional[GeminiMultimodalModalities] = Field(
        default=GeminiMultimodalModalities.AUDIO
    )
    language: Optional[Language] = Field(default=Language.EN_US)
    media_resolution: Optional[GeminiMediaResolution] = Field(
        default=GeminiMediaResolution.UNSPECIFIED
    )
    vad: Optional[GeminiVADParams] = Field(default=None)
    context_window_compression: Optional[ContextWindowCompressionParams] = Field(default=None)
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GeminiMultimodalLiveLLMService(LLMService):
    """Provides access to Google's Gemini Multimodal Live API.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage.

    Args:
        api_key: Google AI API key for authentication.
        base_url: API endpoint base URL. Defaults to the official Gemini Live endpoint.
        model: Model identifier to use. Defaults to "models/gemini-2.0-flash-live-001".
        voice_id: TTS voice identifier. Defaults to "Charon".
        start_audio_paused: Whether to start with audio input paused. Defaults to False.
        start_video_paused: Whether to start with video input paused. Defaults to False.
        system_instruction: System prompt for the model. Defaults to None.
        tools: Tools/functions available to the model. Defaults to None.
        params: Configuration parameters for the model. Defaults to InputParams().
        inference_on_context_initialization: Whether to generate a response when context
            is first set. Defaults to True.
        **kwargs: Additional arguments passed to parent LLMService.
    """

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        model="models/gemini-2.0-flash-live-001",
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: Optional[str] = None,
        tools: Optional[Union[List[dict], ToolsSchema]] = None,
        params: Optional[InputParams] = None,
        inference_on_context_initialization: bool = True,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)

        params = params or InputParams()

        self._last_sent_time = 0
        self._api_key = api_key
        self._base_url = base_url
        self.set_model_name(model)
        self._voice_id = voice_id
        self._language_code = params.language

        self._system_instruction = system_instruction
        self._tools = tools
        self._inference_on_context_initialization = inference_on_context_initialization
        self._needs_turn_complete_message = False

        self._audio_input_paused = start_audio_paused
        self._video_input_paused = start_video_paused
        self._context = None
        self._websocket = None
        self._receive_task = None

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._user_is_speaking = False
        self._bot_is_speaking = False
        self._user_audio_buffer = bytearray()
        self._user_transcription_buffer = ""
        self._last_transcription_sent = ""
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""
        self._llm_output_buffer = ""

        self._sample_rate = 24000

        self._language = params.language
        self._language_code = (
            language_to_gemini_language(params.language) if params.language else "en-US"
        )
        self._vad_params = params.vad

        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "max_tokens": params.max_tokens,
            "presence_penalty": params.presence_penalty,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "modalities": params.modalities,
            "language": self._language_code,
            "media_resolution": params.media_resolution,
            "vad": params.vad,
            "context_window_compression": params.context_window_compression.model_dump()
            if params.context_window_compression
            else {},
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True as Gemini Live supports token usage metrics.
        """
        return True

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

    def set_model_modalities(self, modalities: GeminiMultimodalModalities):
        """Set the model response modalities.

        Args:
            modalities: The modalities to use for responses.
        """
        self._settings["modalities"] = modalities

    def set_language(self, language: Language):
        """Set the language for generation.

        Args:
            language: The language to use for generation.
        """
        self._language = language
        self._language_code = language_to_gemini_language(language) or "en-US"
        self._settings["language"] = self._language_code
        logger.info(f"Set Gemini language to: {self._language_code}")

    async def set_context(self, context: OpenAILLMContext):
        """Set the context explicitly from outside the pipeline.

        This is useful when initializing a conversation because in server-side VAD mode we might not have a
        way to trigger the pipeline. This sends the history to the server. The `inference_on_context_initialization`
        flag controls whether to set the turnComplete flag when we do this. Without that flag, the model will
        not respond. This is often what we want when setting the context at the beginning of a conversation.

        Args:
            context: The OpenAI LLM context to set.
        """
        if self._context:
            logger.error(
                "Context already set. Can only set up Gemini Multimodal Live context once."
            )
            return
        self._context = GeminiMultimodalLiveContext.upgrade(context)
        await self._create_initial_response()

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        """Start the service and establish websocket connection.

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
        self._bot_is_speaking = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        pass

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        self._user_audio_buffer = bytearray()
        await self.start_ttfb_metrics()
        if self._needs_turn_complete_message:
            self._needs_turn_complete_message = False
            evt = events.ClientContentMessage.model_validate(
                {"clientContent": {"turnComplete": True}}
            )
            await self.send_client_event(evt)

    #
    # frame processing
    #
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames for the Gemini Live service.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, OpenAILLMContextFrame):
            context: GeminiMultimodalLiveContext = GeminiMultimodalLiveContext.upgrade(
                frame.context
            )
            # For now, we'll only trigger inference here when either:
            #   1. We have not seen a context frame before
            #   2. The last message is a tool call result
            if not self._context:
                self._context = context
                if frame.context.tools:
                    self._tools = frame.context.tools
                await self._create_initial_response()
            elif context.messages and context.messages[-1].get("role") == "tool":
                # Support just one tool call per context frame for now
                tool_result_message = context.messages[-1]
                await self._tool_result(tool_result_message)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputImageRawFrame):
            await self._send_user_video(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
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
            await self._create_single_response(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()
        else:
            await self.push_frame(frame, direction)

    #
    # websocket communication
    #

    async def send_client_event(self, event):
        """Send a client event to the Gemini Live API.

        Args:
            event: The event to send.
        """
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        if self._websocket:
            # Here we assume that if we have a websocket, we are connected. We
            # handle disconnections in the send/recv code paths.
            return

        logger.info("Connecting to Gemini service")
        try:
            logger.info(f"Connecting to wss://{self._base_url}")
            uri = f"wss://{self._base_url}?key={self._api_key}"
            self._websocket = await websockets.connect(uri=uri)
            self._receive_task = self.create_task(self._receive_task_handler())

            # Create the basic configuration
            config_data = {
                "setup": {
                    "model": self._model_name,
                    "generation_config": {
                        "frequency_penalty": self._settings["frequency_penalty"],
                        "max_output_tokens": self._settings["max_tokens"],
                        "presence_penalty": self._settings["presence_penalty"],
                        "temperature": self._settings["temperature"],
                        "top_k": self._settings["top_k"],
                        "top_p": self._settings["top_p"],
                        "response_modalities": self._settings["modalities"].value,
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {"voice_name": self._voice_id}
                            },
                            "language_code": self._settings["language"],
                        },
                        "media_resolution": self._settings["media_resolution"].value,
                    },
                    "input_audio_transcription": {},
                    "output_audio_transcription": {},
                }
            }

            # Add context window compression if enabled
            if self._settings.get("context_window_compression", {}).get("enabled", False):
                compression_config = {}
                # Add sliding window (always true if compression is enabled)
                compression_config["sliding_window"] = {}

                # Add trigger_tokens if specified
                trigger_tokens = self._settings.get("context_window_compression", {}).get(
                    "trigger_tokens"
                )
                if trigger_tokens is not None:
                    compression_config["trigger_tokens"] = trigger_tokens

                config_data["setup"]["context_window_compression"] = compression_config

            # Add VAD configuration if provided
            if self._settings.get("vad"):
                vad_config = {}
                vad_params = self._settings["vad"]

                # Only add parameters that are explicitly set
                if vad_params.disabled is not None:
                    vad_config["disabled"] = vad_params.disabled

                if vad_params.start_sensitivity:
                    vad_config["start_of_speech_sensitivity"] = vad_params.start_sensitivity.value

                if vad_params.end_sensitivity:
                    vad_config["end_of_speech_sensitivity"] = vad_params.end_sensitivity.value

                if vad_params.prefix_padding_ms is not None:
                    vad_config["prefix_padding_ms"] = vad_params.prefix_padding_ms

                if vad_params.silence_duration_ms is not None:
                    vad_config["silence_duration_ms"] = vad_params.silence_duration_ms

                # Only add automatic_activity_detection if we have VAD settings
                if vad_config:
                    realtime_config = {"automatic_activity_detection": vad_config}

                    config_data["setup"]["realtime_input_config"] = realtime_config

            config = events.Config.model_validate(config_data)

            # Add system instruction if available
            system_instruction = self._system_instruction or ""
            if self._context and hasattr(self._context, "extract_system_instructions"):
                system_instruction += "\n" + self._context.extract_system_instructions()
            if system_instruction:
                logger.debug(f"Setting system instruction: {system_instruction}")
                config.setup.system_instruction = events.SystemInstruction(
                    parts=[events.ContentPart(text=system_instruction)]
                )

            # Add tools if available
            if self._tools:
                logger.debug(f"Gemini is configuring to use tools{self._tools}")
                config.setup.tools = self.get_llm_adapter().from_standard_tools(self._tools)

            # Send the configuration
            await self.send_client_event(config)

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        logger.info("Disconnecting from Gemini service")
        try:
            self._disconnecting = True
            self._api_session_ready = False
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            self._disconnecting = False
        except Exception as e:
            logger.error(f"{self} error disconnecting: {e}")

    async def _ws_send(self, message):
        # logger.debug(f"Sending message to websocket: {message}")
        try:
            if self._websocket:
                await self._websocket.send(json.dumps(message))
        except Exception as e:
            if self._disconnecting:
                return
            logger.error(f"Error sending message to websocket: {e}")
            # In server-to-server contexts, a WebSocket error should be quite rare. Given how hard
            # it is to recover from a send-side error with proper state management, and that exponential
            # backoff for retries can have cost/stability implications for a service cluster, let's just
            # treat a send-side error as fatal.
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

    #
    # inbound server event handling
    # todo: docs link here
    #

    async def _receive_task_handler(self):
        async for message in WatchdogAsyncIterator(self._websocket, manager=self.task_manager):
            evt = events.parse_server_event(message)
            # logger.debug(f"Received event: {message[:500]}")
            # logger.debug(f"Received event: {evt}")

            if evt.setupComplete:
                await self._handle_evt_setup_complete(evt)
            elif evt.serverContent and evt.serverContent.modelTurn:
                await self._handle_evt_model_turn(evt)
            elif evt.serverContent and evt.serverContent.turnComplete and evt.usageMetadata:
                await self._handle_evt_turn_complete(evt)
                await self._handle_evt_usage_metadata(evt)
            elif evt.serverContent and evt.serverContent.inputTranscription:
                await self._handle_evt_input_transcription(evt)
            elif evt.serverContent and evt.serverContent.outputTranscription:
                await self._handle_evt_output_transcription(evt)
            elif evt.toolCall:
                await self._handle_evt_tool_call(evt)
            elif False:  # !!! todo: error events?
                await self._handle_evt_error(evt)
                # errors are fatal, so exit the receive loop
                return

    #
    #
    #

    async def _send_user_audio(self, frame):
        if self._audio_input_paused:
            return
        # Send all audio to Gemini
        evt = events.AudioInputMessage.from_raw_audio(frame.audio, frame.sample_rate)
        await self.send_client_event(evt)
        # Manage a buffer of audio to use for transcription
        audio = frame.audio
        if self._user_is_speaking:
            self._user_audio_buffer.extend(audio)
        else:
            # Keep 1/2 second of audio in the buffer even when not speaking.
            self._user_audio_buffer.extend(audio)
            length = int((frame.sample_rate * frame.num_channels * 2) * 0.5)
            self._user_audio_buffer = self._user_audio_buffer[-length:]

    async def _send_user_video(self, frame):
        if self._video_input_paused:
            return

        now = time.time()
        if now - self._last_sent_time < 1:
            return  # Ignore if less than 1 second has passed

        self._last_sent_time = now  # Update last sent time
        logger.debug(f"Sending video frame to Gemini: {frame}")
        evt = events.VideoInputMessage.from_image_frame(frame)
        await self.send_client_event(evt)

    async def _create_initial_response(self):
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        messages = self._context.get_messages_for_initializing_history()
        if not messages:
            return

        logger.debug(f"Creating initial response: {messages}")

        await self.start_ttfb_metrics()

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": self._inference_on_context_initialization,
                }
            }
        )
        await self.send_client_event(evt)
        if not self._inference_on_context_initialization:
            self._needs_turn_complete_message = True

    async def _create_single_response(self, messages_list):
        # refactor to combine this logic with same logic in GeminiMultimodalLiveContext
        messages = []
        for item in messages_list:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        if not messages:
            return
        logger.debug(f"Creating response: {messages}")

        await self.start_ttfb_metrics()

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": True,
                }
            }
        )
        await self.send_client_event(evt)

    @traced_gemini_live(operation="llm_tool_result")
    async def _tool_result(self, tool_result_message):
        # For now we're shoving the name into the tool_call_id field, so this
        # will work until we revisit that.
        id = tool_result_message.get("tool_call_id")
        name = tool_result_message.get("tool_call_name")
        result = json.loads(tool_result_message.get("content") or "")
        response_message = json.dumps(
            {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": id,
                            "name": name,
                            "response": {
                                "result": result,
                            },
                        }
                    ],
                }
            }
        )
        await self._websocket.send(response_message)
        # await self._websocket.send(json.dumps({"clientContent": {"turnComplete": True}}))

    @traced_gemini_live(operation="llm_setup")
    async def _handle_evt_setup_complete(self, evt):
        # If this is our first context frame, run the LLM
        self._api_session_ready = True
        # Now that we've configured the session, we can run the LLM if we need to.
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_initial_response()

    async def _handle_evt_model_turn(self, evt):
        part = evt.serverContent.modelTurn.parts[0]
        if not part:
            return

        await self.stop_ttfb_metrics()

        # part.text is added when `modalities` is set to TEXT; otherwise, it's None
        text = part.text
        if text:
            if not self._bot_text_buffer:
                await self.push_frame(LLMFullResponseStartFrame())

            self._bot_text_buffer += text
            await self.push_frame(LLMTextFrame(text=text))

        inline_data = part.inlineData
        if not inline_data:
            return
        if inline_data.mimeType != f"audio/pcm;rate={self._sample_rate}":
            logger.warning(f"Unrecognized server_content format {inline_data.mimeType}")
            return

        audio = base64.b64decode(inline_data.data)
        if not audio:
            return

        if not self._bot_is_speaking:
            self._bot_is_speaking = True
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
    async def _handle_evt_tool_call(self, evt):
        function_calls = evt.toolCall.functionCalls
        if not function_calls:
            return
        if not self._context:
            logger.error("Function calls are not supported without a context object.")

        function_calls_llm = [
            FunctionCallFromLLM(
                context=self._context,
                tool_call_id=f.id,
                function_name=f.name,
                arguments=f.args,
            )
            for f in function_calls
        ]

        await self.run_function_calls(function_calls_llm)

    @traced_gemini_live(operation="llm_response")
    async def _handle_evt_turn_complete(self, evt):
        self._bot_is_speaking = False
        text = self._bot_text_buffer

        # Determine output and modality for tracing
        if text:
            # TEXT modality
            output_text = text
            output_modality = "TEXT"
        else:
            # AUDIO modality
            output_text = self._llm_output_buffer
            output_modality = "AUDIO"

        # Trace the complete LLM response (this will be handled by the decorator)
        # The decorator will extract the output text and usage metadata from the event

        self._bot_text_buffer = ""
        self._llm_output_buffer = ""

        # Only push the TTSStoppedFrame if the bot is outputting audio
        # when text is found, modalities is set to TEXT and no audio
        # is produced.
        if not text:
            await self.push_frame(TTSStoppedFrame())

        await self.push_frame(LLMFullResponseEndFrame())

    @traced_stt
    async def _handle_user_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _handle_evt_input_transcription(self, evt):
        """Handle the input transcription event.

        Gemini Live sends user transcriptions in either single words or multi-word
        phrases. As a result, we have to aggregate the input transcription. This handler
        aggregates into sentences, splitting on the end of sentence markers.
        """
        if not evt.serverContent.inputTranscription:
            return

        text = evt.serverContent.inputTranscription.text

        if not text:
            return

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
            await self._handle_user_transcription(
                complete_sentence, True, self._settings["language"]
            )
            await self.push_frame(
                TranscriptionFrame(
                    text=complete_sentence,
                    user_id="",
                    timestamp=time_now_iso8601(),
                    result=evt,
                ),
                FrameDirection.UPSTREAM,
            )

    async def _handle_evt_output_transcription(self, evt):
        if not evt.serverContent.outputTranscription:
            return

        # This is the output transcription text when modalities is set to AUDIO.
        # In this case, we push LLMTextFrame and TTSTextFrame to be handled by the
        # downstream assistant context aggregator.
        text = evt.serverContent.outputTranscription.text

        if not text:
            return

        # Collect text for tracing
        self._llm_output_buffer += text

        await self.push_frame(LLMTextFrame(text=text))
        await self.push_frame(TTSTextFrame(text=text))

    async def _handle_evt_usage_metadata(self, evt):
        if not evt.usageMetadata:
            return

        usage = evt.usageMetadata

        # Ensure we have valid integers for all token counts
        prompt_tokens = usage.promptTokenCount or 0
        completion_tokens = usage.responseTokenCount or 0
        total_tokens = usage.totalTokenCount or (prompt_tokens + completion_tokens)

        tokens = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        await self.start_llm_usage_metrics(tokens)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GeminiMultimodalLiveContextAggregatorPair:
        """Create an instance of GeminiMultimodalLiveContextAggregatorPair from an OpenAILLMContext.

        Constructor keyword arguments for both the user and assistant aggregators can be provided.

        Args:
            context: The LLM context to use.
            user_params: User aggregator parameters. Defaults to LLMUserAggregatorParams().
            assistant_params: Assistant aggregator parameters. Defaults to LLMAssistantAggregatorParams().

        Returns:
            GeminiMultimodalLiveContextAggregatorPair: A pair of context
            aggregators, one for the user and one for the assistant,
            encapsulated in an GeminiMultimodalLiveContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        GeminiMultimodalLiveContext.upgrade(context)
        user = GeminiMultimodalLiveUserContextAggregator(context, params=user_params)

        assistant_params.expect_stripped_words = False
        assistant = GeminiMultimodalLiveAssistantContextAggregator(context, params=assistant_params)
        return GeminiMultimodalLiveContextAggregatorPair(_user=user, _assistant=assistant)

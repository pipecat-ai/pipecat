#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure Voice Live LLM service implementation with WebSocket support.

Based on Azure's Voice Live API documentation:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live
"""

import base64
import json
import re
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any, Self, cast

from loguru import logger
from typing_extensions import override
from websockets.asyncio.client import connect as websocket_connect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.azure_voicelive_adapter import AzureVoiceLiveLLMAdapter
from pipecat.frames.frames import (
    AggregationType,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given

from . import events

AzureTokenProvider = Callable[[], Awaitable[str]]
"""Async callable supplying a Microsoft Entra ID bearer token.

Matches :func:`azure.identity.aio.get_bearer_token_provider` used with the
``https://ai.azure.com/.default`` scope.
"""

DEFAULT_API_VERSION = "2026-07-15"
"""Voice Live API version this service speaks."""

# Output sample rate implied by each PCM output format Voice Live offers.
_OUTPUT_FORMAT_SAMPLE_RATES: dict[int, events.OutputAudioFormat] = {
    8000: "pcm16_8000hz",
    16000: "pcm16_16000hz",
    24000: "pcm16",
}


@dataclass
class CurrentAudioResponse:
    """Tracks the current audio response from the assistant.

    Parameters:
        item_id: Unique identifier for the audio response item.
        content_index: Index of the audio content within the item.
        start_time_ms: Timestamp when the audio response started in milliseconds.
        total_size: Total size of audio data received in bytes. Defaults to 0.
    """

    item_id: str
    content_index: int
    start_time_ms: int
    total_size: int = 0


@dataclass
class AzureVoiceLiveLLMSettings(LLMSettings):
    """Settings for AzureVoiceLiveLLMService.

    Parameters:
        session_properties: Voice Live session properties (voice, turn
            detection, transcription, tools, etc.). ``model``,
            ``instructions`` and ``temperature`` are synced bidirectionally
            with the top-level ``model``, ``system_instruction`` and
            ``temperature`` fields.
    """

    session_properties: events.SessionProperties | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )

    # -- Bidirectional sync helpers ------------------------------------------

    @staticmethod
    def _sync_top_level_to_sp(settings: "AzureVoiceLiveLLMService.Settings"):
        """Push top-level ``model``/``system_instruction``/``temperature`` into SP."""
        if not is_given(settings.session_properties):
            return
        sp = settings.session_properties
        if is_given(settings.model) and settings.model is not None:
            sp.model = settings.model
        if is_given(settings.system_instruction):
            sp.instructions = settings.system_instruction
        if is_given(settings.temperature) and settings.temperature is not None:
            sp.temperature = settings.temperature

    # -- apply_update override -----------------------------------------------

    def apply_update(self, delta: Self) -> dict[str, Any]:
        """Merge a delta, keeping ``model``/``system_instruction`` in sync with SP.

        When the delta contains ``session_properties``, it **replaces** the
        stored SP wholesale. Top-level field values always take precedence over
        conflicting SP values.
        """
        changed = super().apply_update(delta)

        if "session_properties" in changed and is_given(self.session_properties):
            sp = self.session_properties
            if "model" not in changed and sp.model is not None:
                old_model = self.model
                self.model = sp.model
                if old_model != self.model:
                    changed["model"] = old_model
            if "system_instruction" not in changed and sp.instructions is not None:
                old_si = self.system_instruction
                self.system_instruction = sp.instructions
                if old_si != self.system_instruction:
                    changed["system_instruction"] = old_si

        self._sync_top_level_to_sp(self)

        return changed

    # -- from_mapping override -----------------------------------------------

    @classmethod
    def from_mapping(
        cls: type["AzureVoiceLiveLLMService.Settings"], settings: Mapping[str, Any]
    ) -> "AzureVoiceLiveLLMService.Settings":
        """Build a delta from a plain dict, routing SP keys into ``session_properties``.

        Keys that correspond to ``SessionProperties`` fields are collected into
        a nested ``session_properties`` value. ``model`` is always routed to the
        top-level field. Unknown keys go to ``extra``.
        """
        own_field_names = {f.name for f in dataclass_fields(cls)} - {"extra"}

        top: dict[str, Any] = {}
        sp_dict: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        sp_keys = set(events.SessionProperties.model_fields.keys()) - {"model"}

        for key, value in settings.items():
            canonical = cls._aliases.get(key, key)
            if canonical in own_field_names:
                top[canonical] = value
            elif canonical in sp_keys:
                sp_dict[canonical] = value
            else:
                extra[key] = value

        if sp_dict:
            top["session_properties"] = events.SessionProperties(**sp_dict)

        instance = cls(**top)
        instance.extra = extra
        return instance


# Error codes that are non-fatal and should not exit the receive loop.
_NON_FATAL_ERROR_CODES = {
    "response_cancel_not_active",
    "conversation_already_has_active_response",
}


class AzureVoiceLiveLLMService(LLMService[AzureVoiceLiveLLMAdapter]):
    """Azure Voice Live LLM service for real-time audio and text communication.

    Implements Azure's Voice Live API over WebSocket for low-latency
    bidirectional audio. Voice Live combines speech recognition, a generative
    model and Azure text to speech behind a single realtime session, and adds
    noise suppression, echo cancellation and semantic end-of-turn detection.

    Supports function calling, conversation management and real-time
    transcription.

    Proposes turn boundaries from Voice Live's server-side VAD events, which the
    recommended external user turn strategies resolve into
    ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame``.
    ``LLMContextAggregatorPair`` auto-detects this realtime service and
    decouples context writes from those frames. If you wire local VAD
    (``LLMUserAggregatorParams.vad_analyzer``) on top of this service, disable
    Voice Live's server-side turn detection first by passing
    ``turn_detection=None`` in ``session_properties`` (manual mode); otherwise
    both sources broadcast duplicate user-turn frames.

    Azure's own Realtime deployments are a different product served by
    :class:`~pipecat.services.azure.realtime.llm.AzureRealtimeLLMService`; the
    two speak different event names and are not interchangeable.

    Example::

        llm = AzureVoiceLiveLLMService(
            api_key=os.getenv("AZURE_VOICE_LIVE_API_KEY"),
            endpoint=os.getenv("AZURE_VOICE_LIVE_ENDPOINT"),
            model="gpt-4o-mini",
            voice="en-US-Ava:DragonHDLatestNeural",
        )

    For full control over session properties (note: ``session_properties``
    **replaces** all defaults, so provide a complete config)::

        from pipecat.services.azure.voicelive.events import (
            AzureStandardVoice,
            InputAudioNoiseReduction,
            InputAudioTranscription,
            SessionProperties,
            TurnDetection,
        )

        llm = AzureVoiceLiveLLMService(
            api_key=os.getenv("AZURE_VOICE_LIVE_API_KEY"),
            endpoint=os.getenv("AZURE_VOICE_LIVE_ENDPOINT"),
            settings=AzureVoiceLiveLLMService.Settings(
                session_properties=SessionProperties(
                    modalities=["text", "audio"],
                    voice=AzureStandardVoice(name="en-US-Ava:DragonHDLatestNeural"),
                    turn_detection=TurnDetection(
                        type="azure_semantic_vad",
                        silence_duration_ms=500,
                        remove_filler_words=True,
                    ),
                    input_audio_transcription=InputAudioTranscription(model="azure-speech"),
                    input_audio_noise_reduction=InputAudioNoiseReduction(),
                ),
            ),
        )
    """

    Settings = AzureVoiceLiveLLMSettings
    _settings: Settings

    adapter_class = AzureVoiceLiveLLMAdapter

    # Target ~60ms audio chunks when sending to Voice Live (16-bit mono).
    _AUDIO_CHUNK_TARGET_MS = 60

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str | None = None,
        token_provider: AzureTokenProvider | None = None,
        model: str = "gpt-4o-mini",
        voice: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        settings: Settings | None = None,
        start_audio_paused: bool = False,
        **kwargs,
    ):
        """Initialize the Azure Voice Live LLM service.

        Args:
            endpoint: Voice Live endpoint for the Foundry resource. Accepts the
                resource endpoint as shown in the Azure portal
                (``https://<resource>.services.ai.azure.com``) or a full
                WebSocket URL; the scheme and ``/voice-live/realtime`` path are
                filled in when absent.
            api_key: API key for the resource. Required unless
                ``token_provider`` is given.
            token_provider: Async callable supplying a Microsoft Entra ID bearer
                token, used instead of ``api_key`` when given. Build one with
                :func:`azure.identity.aio.get_bearer_token_provider` and the
                ``https://ai.azure.com/.default`` scope.
            model: Model backing the session, e.g. "gpt-4o-mini" or
                "gpt-realtime". Sent as a query parameter on the connection.
            voice: Azure text to speech voice for audio responses, e.g.
                "en-US-Ava:DragonHDLatestNeural". Shorthand for
                ``session_properties.voice``.
            api_version: Voice Live API version to request.
            settings: Full settings for fine-grained control. When
                ``session_properties`` is provided in settings, it **replaces**
                all defaults wholesale — provide a complete ``SessionProperties``
                in that case.
            start_audio_paused: Whether to start with audio input paused.
            **kwargs: Additional arguments passed to parent LLMService.

        Raises:
            ValueError: If neither ``api_key`` nor ``token_provider`` is given.
        """
        if api_key is None and token_provider is None:
            raise ValueError("Either `api_key` or `token_provider` is required.")

        default_voice = voice or "en-US-Ava:DragonHDLatestNeural"

        default_settings = self.Settings(
            model=model,
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
            session_properties=events.SessionProperties(
                model=model,
                modalities=["text", "audio"],
                voice=events.AzureStandardVoice(name=default_voice),
                input_audio_format="pcm16",
                output_audio_format="pcm16",
                turn_detection=events.TurnDetection(
                    type="azure_semantic_vad",
                    create_response=True,
                    interrupt_response=True,
                ),
                input_audio_transcription=events.InputAudioTranscription(model="azure-speech"),
                input_audio_noise_reduction=events.InputAudioNoiseReduction(),
            ),
        )

        self.Settings._sync_top_level_to_sp(default_settings)

        if settings is not None:
            default_settings.apply_update(settings)

        base_url = self._build_base_url(endpoint)

        super().__init__(
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

        self.api_key = api_key
        self.base_url = base_url
        self._token_provider = token_provider
        self._api_version = api_version
        self._model = model

        self._audio_input_paused = start_audio_paused
        self._audio_buffer = b""
        self._audio_send_logged = False
        self._interim_transcription_text = ""
        self._websocket = None
        self._receive_task = None
        self._context: LLMContext | None = None

        self._input_sample_rate: int | None = None
        self._output_sample_rate: int | None = None

        self._llm_needs_conversation_setup = True

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._current_assistant_response = None
        self._current_audio_response: CurrentAudioResponse | None = None

        self._messages_added_manually = {}
        self._pending_function_calls = {}
        self._completed_tool_calls = set()
        self._async_tool_warning_logged: bool = False

        self._register_event_handler("on_conversation_item_created")
        self._register_event_handler("on_conversation_item_updated")

    @staticmethod
    def _build_base_url(endpoint: str) -> str:
        """Normalize a portal endpoint or full URL into a Voice Live WebSocket URL."""
        url = re.sub(r"^https?://", "wss://", endpoint.strip()).rstrip("/")
        if not url.startswith("wss://"):
            url = f"wss://{url}"
        if "/voice-live/realtime" not in url:
            url = f"{url}/voice-live/realtime"
        return url

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics."""
        return True

    def set_audio_input_paused(self, paused: bool):
        """Set whether audio input is paused.

        Args:
            paused: Whether to pause audio input.
        """
        self._audio_input_paused = paused

    def _get_output_sample_rate(self) -> int:
        """Sample rate of the audio Voice Live returns."""
        return self._output_sample_rate or 24000

    def _is_manual_turn_detection(self) -> bool:
        """Whether the caller drives turn boundaries instead of Voice Live."""
        props = self._settings.session_properties
        if not is_given(props):
            return False
        return not props.turn_detection

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Describe this service to the rest of the pipeline."""
        emits_turn_frames = not self._is_manual_turn_detection()
        self._warn_if_realtime_service_emits_no_turn_frames(emits_turn_frames)
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
            user_turn_strategies=ExternalUserTurnStrategies() if emits_turn_frames else None,
        )

    async def _handle_interruption(self):
        """Handle user interruption of assistant speech.

        Server-side VAD cancels the response and clears the buffer itself; in
        manual mode the client must send the cancel and clear events.
        """
        if self._is_manual_turn_detection():
            await self.send_client_event(events.InputAudioBufferClearEvent())
            await self.send_client_event(events.ResponseCancelEvent())
        await self._truncate_current_audio_response()
        await self.stop_all_metrics()

        if self._current_assistant_response:
            # The cancelled response still reports response.done, so clear the
            # tracked item to keep that from closing the turn a second time.
            self._current_assistant_response = None
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(TTSStoppedFrame())

    async def _handle_user_started_speaking(self, frame):
        """Handle user started speaking event."""
        pass

    async def _handle_user_stopped_speaking(self, frame):
        """Handle user stopped speaking event.

        Server-side VAD commits the buffer and creates the response itself; in
        manual mode the client must send them. Metrics are started in
        ``_handle_evt_speech_stopped`` in the server-VAD path.
        """
        if self._is_manual_turn_detection():
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.send_client_event(events.InputAudioBufferCommitEvent())
            await self.send_client_event(events.ResponseCreateEvent())

    async def _handle_bot_stopped_speaking(self):
        """Handle bot stopped speaking event."""
        self._current_audio_response = None

    async def _truncate_current_audio_response(self):
        """Truncate the assistant audio the caller spoke over.

        Tells Voice Live how much of the response was actually heard, so the
        conversation history matches what the caller received.
        """
        if not self._current_audio_response:
            return

        current = self._current_audio_response
        self._current_audio_response = None

        elapsed_ms = int(time.time() * 1000) - current.start_time_ms
        await self.send_client_event(
            events.ConversationItemTruncateEvent(
                item_id=current.item_id,
                content_index=current.content_index,
                audio_end_ms=max(elapsed_ms, 0),
            )
        )

    #
    # Standard AIService frame handling
    #

    def _ensure_audio_config(self, input_sample_rate: int, output_sample_rate: int):
        """Sync the session's audio formats with the transport's sample rates.

        Voice Live takes the input rate directly but selects the output rate
        through the output format, so an unsupported output rate falls back to
        the 24 kHz default.

        Args:
            input_sample_rate: Sample rate for audio input (Hz).
            output_sample_rate: Sample rate for audio output (Hz).
        """
        self._input_sample_rate = input_sample_rate
        props = assert_given(self._settings.session_properties)
        props.input_audio_sampling_rate = input_sample_rate

        output_format = _OUTPUT_FORMAT_SAMPLE_RATES.get(output_sample_rate)
        if output_format:
            props.output_audio_format = output_format
            self._output_sample_rate = output_sample_rate
        else:
            logger.warning(
                f"{self}: Voice Live has no output format for {output_sample_rate}Hz; "
                f"using 24000Hz instead."
            )
            props.output_audio_format = "pcm16"
            self._output_sample_rate = 24000

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._ensure_audio_config(setup.audio_in_sample_rate, setup.audio_out_sample_rate)
        await self._connect()

    async def cleanup(self):
        """Release resources on teardown."""
        await super().cleanup()
        await self._disconnect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close WebSocket connection."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close WebSocket connection."""
        await super().cancel(frame)
        await self._disconnect()

    #
    # Frame processing
    #

    @override
    def _service_tools(self) -> "ToolsSchema | list[Any] | None":
        """Return the tools configured on ``session_properties``, if any."""
        return assert_given(self._settings.session_properties).tools

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            pass
        elif isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            if not self._audio_input_paused:
                await self._send_user_audio(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking()
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_messages_append(frame)
        elif isinstance(frame, LLMSetToolsFrame):
            # Continuous session: no fresh context frame per turn, so sync the
            # registered tool handlers to the new tool set here (the base service
            # only does this on LLMContextFrame).
            self._sync_registered_tool_handlers(frame.tools)
            await self._send_session_update()

        await self.push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        """Handle LLM context updates."""
        if not self._context:
            self._context = context
            await self._process_completed_function_calls(send_new_results=False)
            await self._create_response()
        else:
            self._context = context
            await self._process_completed_function_calls(send_new_results=True)

    async def _handle_messages_append(self, frame):
        """Handle a request to append messages to the conversation."""
        logger.warning(f"{self}: LLMMessagesAppendFrame is not yet supported by Voice Live")

    #
    # WebSocket communication
    #

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event to the Voice Live API.

        Args:
            event: The client event to send.
        """
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        """Establish WebSocket connection to Voice Live."""
        try:
            if self._websocket:
                return

            if self._token_provider:
                headers = {"Authorization": f"Bearer {await self._token_provider()}"}
            else:
                headers = {"api-key": self.api_key or ""}

            params = urllib.parse.urlencode(
                {"api-version": self._api_version, "model": self._model}
            )
            separator = "&" if "?" in self.base_url else "?"
            uri = f"{self.base_url}{separator}{params}"

            logger.info(f"Connecting to {self.base_url}")
            self._websocket = await websocket_connect(uri=uri, additional_headers=headers)
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Voice Live: {e}", exception=e)
            self._websocket = None

    async def _disconnect(self):
        """Close WebSocket connection."""
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

            self._completed_tool_calls = set()
            self._async_tool_warning_logged = False
            self._audio_buffer = b""
            self._interim_transcription_text = ""
            self._disconnecting = False
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)

    async def _ws_send(self, realtime_message):
        """Send a message over the WebSocket connection."""
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(realtime_message))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(error_msg=f"Error sending client event: {e}", exception=e)

    async def _update_settings(self, delta):
        """Apply a settings delta, sending a session update when needed."""
        input_rate = self._input_sample_rate
        output_rate = self._output_sample_rate

        changed = await super()._update_settings(delta)

        if "session_properties" in changed and input_rate and output_rate:
            self._ensure_audio_config(input_rate, output_rate)

        handled = {"session_properties", "system_instruction", "model", "temperature"}
        if changed.keys() & handled:
            await self._send_session_update()
        self._warn_unhandled_updated_settings(changed.keys() - handled)
        return changed

    async def _send_session_update(self):
        """Update session settings on the server."""
        # Mutate a copy: the stored session_properties is read elsewhere (e.g.
        # _service_tools) and must stay intact.
        settings = assert_given(self._settings.session_properties).model_copy()
        adapter = self.get_llm_adapter()

        # The model is selected by the connection URL; repeating it in the
        # session payload is rejected.
        settings.model = None

        if self._context:
            llm_invocation_params = adapter.get_llm_invocation_params(
                self._context,
                system_instruction=assert_given(self._settings.system_instruction),
            )

            # tools given in the context override the tools in the session properties
            if llm_invocation_params["tools"]:
                settings.tools = cast(list[events.VoiceLiveTool], llm_invocation_params["tools"])

            # The adapter resolves conflicts between init-provided and
            # context-provided system instructions (preferring init-provided).
            if llm_invocation_params["system_instruction"]:
                settings.instructions = llm_invocation_params["system_instruction"]

        # Convert ToolsSchema to list of dicts if needed
        if settings.tools and isinstance(settings.tools, ToolsSchema):
            settings.tools = cast(
                list[events.VoiceLiveTool], adapter.from_standard_tools(settings.tools)
            )

        await self.send_client_event(events.SessionUpdateEvent(session=settings))

    #
    # Inbound server event handling
    #

    async def _receive_task_handler(self):
        """Handle incoming WebSocket messages."""
        assert self._websocket is not None

        async for message in self._websocket:
            try:
                raw = json.loads(message)
                event_type = raw.get("type", "")
            except Exception:
                logger.warning(f"Failed to decode server message: {message[:200]}")
                continue

            try:
                evt = events.parse_server_event(message)
            except Exception as e:
                logger.warning(f"Failed to parse server event: {e}")
                continue

            # Unrecognized event type (e.g. an avatar or animation event this
            # service doesn't model). Benign — log quietly and skip.
            if evt is None:
                logger.debug(f"{self} ignoring unhandled server event: {event_type}")
                continue

            if evt.type == "session.created":
                await self._handle_evt_session_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.created":
                pass
            elif evt.type == "response.audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type in (
                "response.content_part.added",
                "response.content_part.done",
                "response.audio_transcript.done",
                "response.text.done",
                "response.audio_timestamp.delta",
                "response.audio_timestamp.done",
                "rate_limits.updated",
            ):
                pass
            elif evt.type == "response.output_item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "response.output_item.done":
                pass
            elif evt.type == "conversation.item.created":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "conversation.item.input_audio_transcription.delta":
                await self._handle_evt_input_audio_transcription_delta(evt)
            elif evt.type == "conversation.item.input_audio_transcription.completed":
                await self._handle_evt_input_audio_transcription_completed(evt)
            elif evt.type == "response.done":
                await self._handle_evt_response_done(evt)
            elif evt.type == "input_audio_buffer.speech_started":
                await self._handle_evt_speech_started(evt)
            elif evt.type == "input_audio_buffer.speech_stopped":
                await self._handle_evt_speech_stopped(evt)
            elif evt.type == "response.audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "response.function_call_arguments.delta":
                pass
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                if evt.error.code in _NON_FATAL_ERROR_CODES:
                    logger.debug(f"{self} {evt.error.message}")
                else:
                    await self._handle_evt_error(evt)
                    return
            else:
                logger.debug(f"{self} received known but undispatched server event: {evt.type}")

    async def _handle_evt_session_created(self, evt):
        """Handle session.created event — first event after connecting."""
        await self._send_session_update()

    async def _handle_evt_session_updated(self, evt):
        """Handle session.updated event."""
        self._api_session_ready = True
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_response()

    async def _handle_evt_audio_delta(self, evt):
        """Handle audio delta event — streaming audio from assistant."""
        await self.stop_ttfb_metrics()

        if self._current_audio_response and self._current_audio_response.item_id != evt.item_id:
            logger.warning(
                "Received a new audio delta for an already completed audio response before receiving the BotStoppedSpeakingFrame."
            )
            logger.debug("Forcing previous audio response to None")
            self._current_audio_response = None

        if not self._current_audio_response:
            self._current_audio_response = CurrentAudioResponse(
                item_id=evt.item_id,
                content_index=evt.content_index,
                start_time_ms=int(time.time() * 1000),
            )
            await self.push_frame(TTSStartedFrame())

        audio = base64.b64decode(evt.delta)
        self._current_audio_response.total_size += len(audio)

        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=self._get_output_sample_rate(),
            num_channels=1,
        )
        await self.push_frame(frame)

    async def _handle_evt_audio_done(self, evt):
        """Handle audio done event."""
        if self._current_audio_response:
            await self.push_frame(TTSStoppedFrame())

    async def _handle_evt_conversation_item_added(self, evt):
        """Handle conversation.item.created and response.output_item.added events."""
        if evt.item.type == "function_call":
            if evt.item.call_id not in self._pending_function_calls:
                self._pending_function_calls[evt.item.call_id] = evt.item
            else:
                logger.debug(f"Function call {evt.item.call_id} already tracked, skipping")

        await self._call_event_handler("on_conversation_item_created", evt.item.id, evt.item)

        if self._messages_added_manually.get(evt.item.id):
            del self._messages_added_manually[evt.item.id]
            return

        if evt.item.role == "assistant":
            # An assistant item is announced twice, by conversation.item.created
            # and by response.output_item.added, so open the response only the
            # first time this item is seen.
            already_open = (
                self._current_assistant_response is not None
                and self._current_assistant_response.id == evt.item.id
            )
            self._current_assistant_response = evt.item
            if not already_open:
                await self.push_frame(LLMFullResponseStartFrame())

    async def _handle_evt_input_audio_transcription_delta(self, evt):
        """Handle streaming input audio transcription delta.

        Accumulates deltas per item and pushes the running text as an
        InterimTranscriptionFrame so the UI shows the full partial transcript.
        """
        if evt.delta:
            self._interim_transcription_text += evt.delta
            await self.push_frame(
                InterimTranscriptionFrame(self._interim_transcription_text, "", time_now_iso8601()),
                FrameDirection.UPSTREAM,
            )

    async def _handle_evt_input_audio_transcription_completed(self, evt):
        """Handle input audio transcription completed event."""
        self._interim_transcription_text = ""
        await self._call_event_handler("on_conversation_item_updated", evt.item_id, None)

        transcript = evt.transcript.strip() if evt.transcript else ""
        if transcript:
            await self.push_frame(
                TranscriptionFrame(transcript, "", time_now_iso8601(), result=evt),
                FrameDirection.UPSTREAM,
            )

    async def _handle_evt_response_done(self, evt):
        """Handle response.done event."""
        usage = evt.usage
        if usage and usage.total_tokens:
            tokens = LLMTokenUsage(
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )
            await self.start_llm_usage_metrics(tokens)

        await self.stop_processing_metrics()

        # An interruption closes the turn before the cancelled response reports
        # done, so only close a turn that is still open.
        if self._current_assistant_response:
            self._current_assistant_response = None
            await self.push_frame(LLMFullResponseEndFrame())

        if evt.status == "failed":
            details = evt.response.get("status_details")
            await self.push_error(error_msg=str(details) if details else "Response failed")
            return

        for item in evt.response.get("output", []):
            await self._call_event_handler("on_conversation_item_updated", item.get("id"), item)

    async def _handle_evt_audio_transcript_delta(self, evt):
        """Handle audio transcript delta event."""
        if evt.delta:
            await self._push_output_transcript_text_frames(evt.delta)

    async def _push_output_transcript_text_frames(self, text: str):
        # Push LLMTextFrame for RTVI "bot-llm-text" events (not appended to context
        # to avoid duplication since the realtime API manages its own context).
        llm_text_frame = LLMTextFrame(text)
        llm_text_frame.append_to_context = False
        await self.push_frame(llm_text_frame)

        # Push TTSTextFrame for output aggregation
        tts_text_frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        tts_text_frame.includes_inter_frame_spaces = True
        await self.push_frame(tts_text_frame)

    async def _handle_evt_function_call_arguments_done(self, evt):
        """Handle function call arguments done event."""
        try:
            args = json.loads(evt.arguments)

            function_call_item = self._pending_function_calls.get(evt.call_id)
            if function_call_item:
                del self._pending_function_calls[evt.call_id]

                function_name = evt.name or function_call_item.name
                if not function_name:
                    logger.warning(f"No function name for call_id: {evt.call_id}")
                    return

                function_calls = [
                    FunctionCallFromLLM(
                        context=self._context,
                        tool_call_id=evt.call_id,
                        function_name=function_name,
                        arguments=args,
                    )
                ]

                await self.run_function_calls(function_calls)
                logger.debug(f"Processed function call: {function_name}")
            else:
                logger.warning(f"No tracked function call found for call_id: {evt.call_id}")

        except Exception as e:
            logger.error(f"Failed to process function call arguments: {e}")

    async def _handle_evt_speech_started(self, evt):
        """Handle speech started event from server-side VAD."""
        if self._is_manual_turn_detection():
            # In manual mode, the client is responsible for broadcasting user turn frames
            return

        await self._truncate_current_audio_response()
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _handle_evt_speech_stopped(self, evt):
        """Handle speech stopped event from server-side VAD."""
        if self._is_manual_turn_detection():
            # In manual mode, the client is responsible for broadcasting user turn frames
            return

        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _handle_evt_error(self, evt):
        """Handle fatal error event."""
        await self.push_error(error_msg=f"Azure Voice Live Error: {evt.error.message}")

    #
    # Response creation
    #

    async def reset_conversation(self):
        """Reset the conversation by disconnecting and reconnecting.

        This fully resets the server-side conversation state. Audio buffers,
        pending function calls, and conversation history are cleared.
        """
        logger.debug("Resetting Voice Live conversation")
        await self._disconnect()

        self._llm_needs_conversation_setup = True
        await self._process_completed_function_calls(send_new_results=False)

        await self._connect()

    async def _create_response(self):
        """Create an assistant response."""
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        assert self._context is not None

        adapter = self.get_llm_adapter()

        if self._llm_needs_conversation_setup:
            logger.debug(
                f"Setting up Voice Live conversation with initial messages: "
                f"{adapter.get_messages_for_logging(self._context)}"
            )

            llm_invocation_params = adapter.get_llm_invocation_params(
                self._context,
                system_instruction=assert_given(self._settings.system_instruction),
            )

            for item in llm_invocation_params["messages"]:
                evt = events.ConversationItemCreateEvent(item=item)
                if evt.item.id:
                    self._messages_added_manually[evt.item.id] = True
                await self.send_client_event(evt)

            await self._send_session_update()
            self._llm_needs_conversation_setup = False

        logger.debug("Creating Voice Live response")

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        modalities = assert_given(self._settings.session_properties).modalities or [
            "text",
            "audio",
        ]
        await self.send_client_event(
            events.ResponseCreateEvent(response=events.ResponseProperties(modalities=modalities))
        )

    async def _process_completed_function_calls(self, send_new_results: bool):
        """Process completed function calls and send results to the service."""
        assert self._context is not None

        # If the user registered a function with cancel_on_interruption=False,
        # the aggregator emits async-tool-style messages into the context.
        # Voice Live has no channel for streamed intermediate results, so
        # surface a one-time warning where the expectation is set.
        if not self._async_tool_warning_logged:
            for message in self._context.get_messages():
                if isinstance(message, LLMSpecificMessage):
                    continue
                if async_tool_messages.parse_message(message) is not None:
                    logger.error(
                        f"{self}: cancel_on_interruption=False is not reliably "
                        f"supported by Voice Live as of this writing. "
                        f"Use cancel_on_interruption=True (the default), or "
                        f"consider another LLM service if your tool needs the "
                        f"async semantics."
                    )
                    await self.push_error(
                        error_msg=(
                            "cancel_on_interruption=False is not reliably supported "
                            "by Voice Live as of this writing."
                        ),
                    )
                    self._async_tool_warning_logged = True
                    break

        sent_new_result = False

        for message in self._context.get_messages():
            # LLMSpecificMessages are opaque provider-specific payloads, not
            # standard tool-result messages — skip them.
            if isinstance(message, LLMSpecificMessage):
                continue

            # Async-tool messages live alongside regular tool messages in the
            # context; detect and route them before the regular logic so we
            # don't try to send the async-tool envelope JSON as a tool result.
            async_payload = async_tool_messages.parse_message(message)
            if async_payload is not None:
                if async_payload.tool_call_id in self._completed_tool_calls:
                    continue
                if async_payload.kind == "started":
                    # The provider already issued the tool call and natively
                    # awaits a result; nothing to send for the started marker.
                    continue
                if async_payload.kind == "intermediate":
                    logger.error(
                        f"{self}: Voice Live does not support streamed async "
                        f"tool results; dropping intermediate result for "
                        f"tool_call_id={async_payload.tool_call_id}. Consider "
                        f"another LLM service if your tool needs to stream "
                        f"intermediate results."
                    )
                    await self.push_error(
                        error_msg="Voice Live does not support streamed async tool results.",
                    )
                    continue
                if async_payload.kind == "final":
                    # Deliver via the formal tool-result channel — same path
                    # as a synchronous tool result, just delayed.
                    if send_new_results:
                        sent_new_result = True
                        await self._send_tool_result(
                            async_payload.tool_call_id, async_payload.result
                        )
                    self._completed_tool_calls.add(async_payload.tool_call_id)
                    continue
                # Defensive: any async-tool message must not fall through
                # to the regular tool-result block below, even if it
                # carries a kind we don't recognize.
                continue

            # Look for newly-completed "regular" (as opposed to async-tool) results
            if message.get("role") == "tool" and message.get("content") != "IN_PROGRESS":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in self._completed_tool_calls:
                    if send_new_results:
                        sent_new_result = True
                        await self._send_tool_result(
                            tool_call_id, cast(str | None, message.get("content"))
                        )
                    self._completed_tool_calls.add(tool_call_id)

        # If we reported any new tool call results to the service, trigger
        # another response
        if sent_new_result:
            await self._create_response()

    async def _send_user_audio(self, frame):
        """Send user audio to Voice Live, buffered to ~60ms chunks."""
        if self._llm_needs_conversation_setup:
            return

        if not self._audio_send_logged:
            logger.debug(
                f"Streaming audio to Voice Live: {frame.sample_rate}Hz, "
                f"{frame.num_channels}ch, {len(frame.audio)}B/frame"
            )
            self._audio_send_logged = True

        # Compute chunk size from actual sample rate (16-bit mono = 2 bytes/sample)
        chunk_bytes = int(frame.sample_rate * 2 * self._AUDIO_CHUNK_TARGET_MS / 1000)

        # Accumulate and send in chunks
        self._audio_buffer += frame.audio
        while len(self._audio_buffer) >= chunk_bytes:
            chunk = self._audio_buffer[:chunk_bytes]
            self._audio_buffer = self._audio_buffer[chunk_bytes:]
            payload = base64.b64encode(chunk).decode("utf-8")
            await self.send_client_event(events.InputAudioBufferAppendEvent(audio=payload))

    async def _send_tool_result(self, tool_call_id: str, result: str | None):
        """Send a tool call result to Voice Live."""
        item = events.ConversationItem(
            type="function_call_output",
            call_id=tool_call_id,
            output=result,
        )
        await self.send_client_event(events.ConversationItemCreateEvent(item=item))

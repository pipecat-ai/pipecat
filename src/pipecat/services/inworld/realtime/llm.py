#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld Realtime LLM service implementation with WebSocket support.

Based on Inworld's Realtime API documentation:
https://docs.inworld.ai/api-reference/realtimeAPI/realtime/realtime-websocket
"""

import base64
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Literal, Mapping, Optional, Type

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.inworld_realtime_adapter import InworldRealtimeLLMAdapter
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
    LLMSetToolsFrame,
    LLMTextFrame,
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
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import (
    NOT_GIVEN,
    LLMSettings,
    _NotGiven,
    is_given,
)
from pipecat.utils.time import time_now_iso8601

from . import events

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Inworld Realtime, you need to `pip install pipecat-ai[inworld]`.")
    raise Exception(f"Missing module: {e}")


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
class InworldRealtimeLLMSettings(LLMSettings):
    """Settings for InworldRealtimeLLMService.

    Parameters:
        session_properties: Inworld Realtime session properties (audio config,
            tools, etc.).  ``model`` and ``instructions`` are synced
            bidirectionally with the top-level ``model`` and
            ``system_instruction`` fields.
    """

    session_properties: events.SessionProperties | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )

    # -- Bidirectional sync helpers ------------------------------------------

    @staticmethod
    def _sync_top_level_to_sp(settings: "InworldRealtimeLLMService.Settings"):
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

    def apply_update(self, delta: "InworldRealtimeLLMService.Settings") -> Dict[str, Any]:
        """Merge a delta, keeping ``model``/``system_instruction`` in sync with SP.

        When the delta contains ``session_properties``, it **replaces** the
        stored SP wholesale (matching legacy behaviour).  Top-level field
        values always take precedence over conflicting SP values.
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
        cls: Type["InworldRealtimeLLMService.Settings"], settings: Mapping[str, Any]
    ) -> "InworldRealtimeLLMService.Settings":
        """Build a delta from a plain dict, routing SP keys into ``session_properties``.

        Keys that correspond to ``SessionProperties`` fields are collected into
        a nested ``session_properties`` value.  ``model`` is always routed to
        the top-level field.  Unknown keys go to ``extra``.
        """
        own_field_names = {f.name for f in dataclass_fields(cls)} - {"extra"}

        top: Dict[str, Any] = {}
        sp_dict: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

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


class InworldRealtimeLLMService(LLMService):
    """Inworld Realtime LLM service for real-time audio and text communication.

    Implements the Inworld Realtime API with WebSocket communication for
    low-latency bidirectional audio and text interactions. The API operates
    as a cascade STT/LLM/TTS pipeline under the hood, with built-in semantic
    voice activity detection (VAD) for turn management.

    Supports function calling, conversation management, and real-time
    transcription.

    Example::

        llm = InworldRealtimeLLMService(
            api_key=os.getenv("INWORLD_API_KEY"),
            llm_model="openai/gpt-4.1-nano",
            voice="Sarah",
            tts_model="inworld-tts-1.5-max",
        )

    For full control over session properties (note: ``session_properties``
    **replaces** all defaults, so provide a complete config)::

        from pipecat.services.inworld.realtime.events import *

        llm = InworldRealtimeLLMService(
            api_key=os.getenv("INWORLD_API_KEY"),
            settings=InworldRealtimeLLMService.Settings(
                session_properties=SessionProperties(
                    model="openai/gpt-4.1-nano",
                    temperature=0.7,
                    audio=AudioConfiguration(
                        input=AudioInput(
                            format=PCMAudioFormat(rate=24000),
                            turn_detection=TurnDetection(
                                type="semantic_vad",
                                eagerness="low",
                            ),
                        ),
                        output=AudioOutput(
                            format=PCMAudioFormat(rate=24000),
                            voice="Sarah",
                            model="inworld-tts-1.5-max",
                        ),
                    ),
                ),
            ),
        )
    """

    Settings = InworldRealtimeLLMSettings
    _settings: Settings

    adapter_class = InworldRealtimeLLMAdapter

    # Target ~60ms audio chunks when sending to Inworld (16-bit mono).
    _AUDIO_CHUNK_TARGET_MS = 60

    def __init__(
        self,
        *,
        api_key: str,
        llm_model: Optional[str] = None,
        voice: Optional[str] = None,
        tts_model: Optional[str] = None,
        stt_model: Optional[str] = None,
        base_url: str = "wss://api.inworld.ai/api/v1/realtime/session",
        auth_type: Literal["basic", "bearer"] = "basic",
        settings: Optional[Settings] = None,
        start_audio_paused: bool = False,
        **kwargs,
    ):
        """Initialize the Inworld Realtime LLM service.

        Args:
            api_key: Inworld API key for authentication.
            llm_model: LLM model to use (e.g. "openai/gpt-4.1-nano").
                Shorthand for ``session_properties.model``.
            voice: Voice ID for TTS output (e.g. "Sarah", "Clive").
                Shorthand for ``session_properties.audio.output.voice``.
            tts_model: TTS model to use (e.g. "inworld-tts-1.5-max").
                Shorthand for ``session_properties.audio.output.model``.
            stt_model: STT model for input transcription
                (e.g. "assemblyai/universal-streaming-multilingual").
                Shorthand for ``session_properties.audio.input.transcription.model``.
            base_url: WebSocket base URL for the realtime API.
            auth_type: Authentication type. ``"basic"`` for server-side API key
                auth, ``"bearer"`` for client-side JWT auth.
            settings: Full settings for fine-grained control. When
                ``session_properties`` is provided in settings, it **replaces**
                all defaults wholesale — provide a complete ``SessionProperties``
                in that case.
            start_audio_paused: Whether to start with audio input paused.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        default_model = llm_model or "openai/gpt-4.1-nano"
        default_voice = voice or "Clive"

        default_settings = self.Settings(
            model=default_model,
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
                model=default_model,
                output_modalities=["audio", "text"],
                audio=events.AudioConfiguration(
                    input=events.AudioInput(
                        format=events.PCMAudioFormat(rate=24000),
                        transcription=(
                            events.InputTranscription(model=stt_model) if stt_model else None
                        ),
                        turn_detection=events.TurnDetection(
                            type="semantic_vad",
                            eagerness="high",
                            create_response=True,
                            interrupt_response=True,
                        ),
                    ),
                    output=events.AudioOutput(
                        format=events.PCMAudioFormat(rate=24000),
                        model=tts_model,
                        voice=default_voice,
                    ),
                ),
            ),
        )

        self.Settings._sync_top_level_to_sp(default_settings)

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

        self.api_key = api_key
        self.base_url = base_url
        self._auth_type = auth_type

        self._audio_input_paused = start_audio_paused
        self._audio_buffer = b""
        self._audio_send_logged = False
        self._interim_transcription_text = ""
        self._websocket = None
        self._receive_task = None
        self._context: LLMContext = None
        self._last_context_message_count = 0

        self._llm_needs_conversation_setup = True

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._current_assistant_response = None
        self._current_audio_response = None
        self._server_vad_handled_turn = False

        self._messages_added_manually = {}
        self._pending_function_calls = {}
        self._completed_tool_calls = set()

        self._register_event_handler("on_conversation_item_created")
        self._register_event_handler("on_conversation_item_updated")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics."""
        return True

    def set_audio_input_paused(self, paused: bool):
        """Set whether audio input is paused.

        Args:
            paused: True to pause audio input, False to resume.
        """
        self._audio_input_paused = paused

    def _get_configured_sample_rate(self, direction: str) -> Optional[int]:
        """Get manually configured sample rate for input or output.

        Args:
            direction: Either "input" or "output".

        Returns:
            Configured sample rate or None if not manually configured.
        """
        if not self._settings.session_properties.audio:
            return None

        audio_config = (
            self._settings.session_properties.audio.input
            if direction == "input"
            else self._settings.session_properties.audio.output
        )

        if audio_config and audio_config.format:
            if hasattr(audio_config.format, "rate"):
                return audio_config.format.rate
            elif audio_config.format.type in ("audio/pcmu", "audio/pcma"):
                return 8000

        return None

    def _get_output_sample_rate(self) -> int:
        """Get the output sample rate.

        Returns:
            Output sample rate in Hz, defaulting to 24000.
        """
        rate = self._get_configured_sample_rate("output")
        if rate is not None:
            return rate
        return getattr(self, "_output_sample_rate", 24000)

    def _is_turn_detection_enabled(self) -> bool:
        """Check if server-side turn detection is enabled."""
        sp = self._settings.session_properties
        if sp.audio and sp.audio.input and sp.audio.input.turn_detection:
            return sp.audio.input.turn_detection.type in ("server_vad", "semantic_vad")
        return False

    async def _handle_interruption(self):
        """Handle user interruption of assistant speech."""
        if not self._is_turn_detection_enabled():
            await self.send_client_event(events.InputAudioBufferClearEvent())
            await self.send_client_event(events.ResponseCancelEvent())

        await self._truncate_current_audio_response()
        await self.stop_all_metrics()

        if self._current_assistant_response:
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(TTSStoppedFrame())

    async def _handle_user_started_speaking(self, frame):
        """Handle user started speaking event."""
        pass

    async def _handle_user_stopped_speaking(self, frame):
        """Handle user stopped speaking event.

        When server-side turn detection is disabled, pipecat's local VAD
        drives commit + response. When enabled, the server handles it.
        """
        if not self._is_turn_detection_enabled():
            await self.send_client_event(events.InputAudioBufferCommitEvent())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            await self.send_client_event(events.ResponseCreateEvent())

    async def _handle_bot_stopped_speaking(self):
        """Handle bot stopped speaking event."""
        self._current_audio_response = None

    async def _truncate_current_audio_response(self):
        """Truncates the current audio response (best-effort cleanup)."""
        self._current_audio_response = None

    #
    # Standard AIService frame handling
    #

    def _ensure_audio_config(self, input_sample_rate: int, output_sample_rate: int):
        """Ensure session_properties.audio has input and output configs.

        Preserves Inworld-specific fields (turn_detection, voice, model).

        Args:
            input_sample_rate: Sample rate for audio input (Hz).
            output_sample_rate: Sample rate for audio output (Hz).
        """
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        props = self._settings.session_properties
        if not props.audio:
            props.audio = events.AudioConfiguration()
        if not props.audio.input:
            props.audio.input = events.AudioInput()
        if not props.audio.output:
            props.audio.output = events.AudioOutput()

    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection."""
        await super().start(frame)
        self._ensure_audio_config(frame.audio_in_sample_rate, frame.audio_out_sample_rate)
        await self._connect()

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
            await self._send_session_update()

        await self.push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        """Handle LLM context updates."""
        if not self._context:
            self._context = context
            self._last_context_message_count = len(context.get_messages())
            await self._process_completed_function_calls(send_new_results=False)
            await self._create_response()
        else:
            self._context = context
            await self._process_completed_function_calls(send_new_results=True)

            # Check for new user messages (e.g. from text input).
            # The context is a shared mutable object, so we track the last
            # known message count to detect new additions.
            messages = self._context.get_messages()
            current_count = len(messages)
            if current_count > self._last_context_message_count:
                last_msg = messages[-1]
                self._last_context_message_count = current_count

                # When server-side VAD handled this turn, the server already
                # has the user's audio and auto-created a response.  Skip
                # sending a duplicate text item + response.create.
                if self._server_vad_handled_turn:
                    self._server_vad_handled_turn = False
                    return

                if last_msg.get("role") == "user":
                    content = last_msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    if content:
                        item = events.ConversationItem(
                            role="user",
                            type="message",
                            content=[events.ItemContent(type="input_text", text=content)],
                        )
                        await self.send_client_event(events.ConversationItemCreateEvent(item=item))
                        await self.start_processing_metrics()
                        await self.start_ttfb_metrics()
                        await self.send_client_event(events.ResponseCreateEvent())

    async def _handle_messages_append(self, frame):
        """Handle appending messages to the context (not yet supported)."""
        logger.warning(f"{self}: LLMMessagesAppendFrame is not yet supported by Inworld Realtime")

    #
    # WebSocket communication
    #

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event to the Inworld Realtime API.

        Args:
            event: The client event to send.
        """
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        """Establish WebSocket connection to Inworld."""
        try:
            if self._websocket:
                return

            if self._auth_type == "bearer":
                auth_header = f"Bearer {self.api_key}"
            else:
                auth_header = f"Basic {self.api_key}"

            # Inworld requires key and protocol query parameters
            session_key = f"voice-{int(time.time() * 1000)}"
            params = urllib.parse.urlencode({"key": session_key, "protocol": "realtime"})
            separator = "&" if "?" in self.base_url else "?"
            uri = f"{self.base_url}{separator}{params}"

            self._websocket = await websocket_connect(
                uri=uri,
                additional_headers={
                    "Authorization": auth_header,
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Inworld: {e}", exception=e)
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
        input_rate = self._get_configured_sample_rate("input")
        output_rate = self._get_configured_sample_rate("output")

        changed = await super()._update_settings(delta)

        if "session_properties" in changed and input_rate and output_rate:
            self._ensure_audio_config(input_rate, output_rate)

        handled = {"session_properties", "system_instruction", "model"}
        if changed.keys() & handled:
            await self._send_session_update()
        self._warn_unhandled_updated_settings(changed.keys() - handled)
        return changed

    async def _send_session_update(self):
        """Update session settings on the server."""
        settings = self._settings.session_properties
        adapter: InworldRealtimeLLMAdapter = self.get_llm_adapter()

        if self._context:
            llm_invocation_params = adapter.get_llm_invocation_params(self._context)

            if llm_invocation_params["tools"]:
                settings.tools = llm_invocation_params["tools"]

            if llm_invocation_params["system_instruction"]:
                settings.instructions = llm_invocation_params["system_instruction"]

        # Convert ToolsSchema to list of dicts if needed
        if settings.tools and isinstance(settings.tools, ToolsSchema):
            settings.tools = adapter.from_standard_tools(settings.tools)

        await self.send_client_event(events.SessionUpdateEvent(session=settings))

    #
    # Inbound server event handling
    #

    async def _receive_task_handler(self):
        """Handle incoming WebSocket messages."""
        async for message in self._websocket:
            try:
                raw = json.loads(message)
                event_type = raw.get("type", "")
            except Exception:
                logger.warning(f"Failed to decode server message: {message[:200]}")
                continue

            # Skip events that don't have a matching Pydantic model
            if event_type in ("conversation.item.done",):
                continue

            try:
                evt = events.parse_server_event(message)
            except Exception as e:
                logger.warning(f"Failed to parse server event: {e}")
                continue

            if evt.type == "ping":
                pass
            elif evt.type == "session.created":
                await self._handle_evt_session_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.created":
                pass
            elif evt.type == "response.output_audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.output_audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type in ("response.content_part.added", "response.content_part.done"):
                pass
            elif evt.type == "response.output_item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "response.output_item.done":
                pass
            elif evt.type == "conversation.item.added":
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
            elif evt.type == "response.output_audio_transcript.delta":
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
        """Handle conversation.item.added event."""
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
            self._current_assistant_response = evt.item
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
        usage = evt.usage or evt.response.usage
        if usage and usage.total_tokens:
            tokens = LLMTokenUsage(
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )
            await self.start_llm_usage_metrics(tokens)

        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
        self._current_assistant_response = None

        if evt.response.status == "failed":
            error_msg = "Response failed"
            if evt.response.status_details:
                error_msg = str(evt.response.status_details)
            await self.push_error(error_msg=error_msg)
            return

        for item in evt.response.output:
            await self._call_event_handler("on_conversation_item_updated", item.id, item)

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

                # Inworld may omit `name` from the done event — resolve from
                # the tracked function call item.
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
        await self._truncate_current_audio_response()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        await self.broadcast_interruption()

    async def _handle_evt_speech_stopped(self, evt):
        """Handle speech stopped event from server-side VAD."""
        # Mark that the server is handling this turn (and will auto-create a
        # response when create_response=True).  This prevents _handle_context
        # from sending a duplicate ResponseCreateEvent when the user aggregator
        # appends the transcribed text to the context.
        self._server_vad_handled_turn = True
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    async def _handle_evt_error(self, evt):
        """Handle fatal error event."""
        await self.push_error(error_msg=f"Inworld Realtime Error: {evt.error.message}")

    #
    # Response creation
    #

    async def reset_conversation(self):
        """Reset the conversation by disconnecting and reconnecting.

        This fully resets the server-side conversation state. Audio buffers,
        pending function calls, and conversation history are cleared.
        """
        logger.debug("Resetting Inworld conversation")
        await self._disconnect()

        self._llm_needs_conversation_setup = True
        await self._process_completed_function_calls(send_new_results=False)

        await self._connect()

    async def _create_response(self):
        """Create an assistant response."""
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        adapter: InworldRealtimeLLMAdapter = self.get_llm_adapter()

        if self._llm_needs_conversation_setup:
            logger.debug(
                f"Setting up Inworld conversation with initial messages: "
                f"{adapter.get_messages_for_logging(self._context)}"
            )

            llm_invocation_params = adapter.get_llm_invocation_params(self._context)
            messages = llm_invocation_params["messages"]

            for item in messages:
                evt = events.ConversationItemCreateEvent(item=item)
                self._messages_added_manually[evt.item.id] = True
                await self.send_client_event(evt)

            await self._send_session_update()
            self._llm_needs_conversation_setup = False

        logger.debug("Creating Inworld response")

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        modalities = self._settings.session_properties.output_modalities or ["text", "audio"]
        await self.send_client_event(
            events.ResponseCreateEvent(response=events.ResponseProperties(modalities=modalities))
        )

    async def _process_completed_function_calls(self, send_new_results: bool):
        """Process completed function calls and send results to the service."""
        sent_new_result = False

        for message in self._context.get_messages():
            if message.get("role") == "tool" and message.get("content") != "IN_PROGRESS":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in self._completed_tool_calls:
                    if send_new_results:
                        sent_new_result = True
                        await self._send_tool_result(tool_call_id, message.get("content"))
                    self._completed_tool_calls.add(tool_call_id)

        if sent_new_result:
            await self._create_response()

    async def _send_user_audio(self, frame):
        """Send user audio to Inworld, buffered to ~60ms chunks."""
        if self._llm_needs_conversation_setup:
            return

        if not self._audio_send_logged:
            logger.debug(
                f"Streaming audio to Inworld: {frame.sample_rate}Hz, "
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
            await self._ws_send({"type": "input_audio_buffer.append", "audio": payload})

    async def _send_tool_result(self, tool_call_id: str, result: str):
        """Send a tool call result to Inworld."""
        item = events.ConversationItem(
            type="function_call_output",
            call_id=tool_call_id,
            output=json.dumps(result, ensure_ascii=False),
        )
        await self.send_client_event(events.ConversationItemCreateEvent(item=item))

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> LLMContextAggregatorPair:
        """Create context aggregators for the Inworld Realtime service.

        Args:
            context: The LLM context.
            user_params: User aggregator parameters.
            assistant_params: Assistant aggregator parameters.

        Returns:
            LLMContextAggregatorPair for user and assistant context aggregation.
        """
        context = LLMContext.from_openai_context(context)
        assistant_params.expect_stripped_words = False
        return LLMContextAggregatorPair(
            context, user_params=user_params, assistant_params=assistant_params
        )

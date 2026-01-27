#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime Voice Agent LLM service implementation with WebSocket support.

Based on xAI's Grok Voice Agent API documentation:
https://docs.x.ai/docs/guides/voice/agent
"""

import base64
import json
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.grok_realtime_adapter import GrokRealtimeLLMAdapter
from pipecat.frames.frames import (
    AggregationType,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
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
from pipecat.utils.time import time_now_iso8601

from . import events

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Grok Realtime, you need to `pip install pipecat-ai[grok]`.")
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


class GrokRealtimeLLMService(LLMService):
    """Grok Realtime Voice Agent LLM service providing real-time audio and text communication.

    Implements the Grok Voice Agent API with WebSocket communication for low-latency
    bidirectional audio and text interactions. Supports function calling, conversation
    management, and real-time transcription.

    Features:
        - Real-time audio streaming (PCM, PCMU, PCMA formats)
        - Configurable sample rates (8kHz to 48kHz for PCM)
        - Multiple voice options (Ara, Rex, Sal, Eve, Leo)
        - Built-in tools (web_search, x_search, file_search)
        - Custom function calling
        - Server-side VAD (Voice Activity Detection)
    """

    # Use the Grok-specific adapter
    adapter_class = GrokRealtimeLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.x.ai/v1/realtime",
        session_properties: Optional[events.SessionProperties] = None,
        start_audio_paused: bool = False,
        **kwargs,
    ):
        """Initialize the Grok Realtime Voice Agent LLM service.

        Args:
            api_key: xAI API key for authentication.
            base_url: WebSocket base URL for the realtime API.
                Defaults to "wss://api.x.ai/v1/realtime".
            session_properties: Configuration properties for the realtime session.
                If None, uses default SessionProperties with voice "Ara".
                To set a different voice, configure it in session_properties:

                    session_properties = events.SessionProperties(voice="Rex")

                Available voices: Ara, Rex, Sal, Eve, Leo.
            start_audio_paused: Whether to start with audio input paused. Defaults to False.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        super().__init__(base_url=base_url, **kwargs)

        self.api_key = api_key
        self.base_url = base_url

        # Initialize session_properties
        self._session_properties: events.SessionProperties = (
            session_properties or events.SessionProperties()
        )

        self._audio_input_paused = start_audio_paused
        self._websocket = None
        self._receive_task = None
        self._context: LLMContext = None

        self._llm_needs_conversation_setup = True

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._current_assistant_response = None
        self._current_audio_response = None

        self._messages_added_manually = {}
        self._pending_function_calls = {}
        self._completed_tool_calls = set()

        self._register_event_handler("on_conversation_item_created")
        self._register_event_handler("on_conversation_item_updated")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True if metrics generation is supported.
        """
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
            For PCMU/PCMA formats, returns 8000 Hz (G.711 standard).
        """
        if not self._session_properties.audio:
            return None

        audio_config = (
            self._session_properties.audio.input
            if direction == "input"
            else self._session_properties.audio.output
        )

        if audio_config and audio_config.format:
            # PCM format has configurable rate
            if hasattr(audio_config.format, "rate"):
                return audio_config.format.rate
            # PCMU/PCMA formats are fixed at 8000 Hz (G.711 standard)
            elif audio_config.format.type in ("audio/pcmu", "audio/pcma"):
                return 8000

        return None

    def _get_output_sample_rate(self) -> int:
        """Get the output sample rate from session properties.

        Returns:
            Output sample rate in Hz.

        Note:
            This assumes start() has been called, which guarantees
            session_properties.audio.output exists.
        """
        rate = self._get_configured_sample_rate("output")
        if rate is None:
            raise RuntimeError("Output sample rate not configured.")
        return rate

    def _is_turn_detection_enabled(self) -> bool:
        """Check if server-side VAD is enabled."""
        if self._session_properties.turn_detection:
            return self._session_properties.turn_detection.type == "server_vad"
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
        """Handle user stopped speaking event."""
        if not self._is_turn_detection_enabled():
            await self.send_client_event(events.InputAudioBufferCommitEvent())
            await self.send_client_event(events.ResponseCreateEvent())

    async def _handle_bot_stopped_speaking(self):
        """Handle bot stopped speaking event."""
        self._current_audio_response = None

    def _calculate_audio_duration_ms(
        self, total_bytes: int, sample_rate: int = None, bytes_per_sample: int = 2
    ) -> int:
        """Calculate audio duration in milliseconds based on PCM audio parameters."""
        if sample_rate is None:
            sample_rate = self._get_output_sample_rate()
        samples = total_bytes / bytes_per_sample
        duration_seconds = samples / sample_rate
        return int(duration_seconds * 1000)

    async def _truncate_current_audio_response(self):
        """Truncates the current audio response.

        Note: Grok may not support truncation events like OpenAI.
        This is a best-effort cleanup.
        """
        if not self._current_audio_response:
            return

        try:
            self._current_audio_response = None
        except Exception as e:
            logger.warning(f"Audio truncation cleanup failed (non-fatal): {e}")

    #
    # Standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)

        # Ensure audio configuration exists with both input and output
        if not self._session_properties.audio:
            self._session_properties.audio = events.AudioConfiguration()

        # Fill in missing input configuration
        if not self._session_properties.audio.input:
            self._session_properties.audio.input = events.AudioInput(
                format=events.PCMAudioFormat(rate=frame.audio_in_sample_rate)
            )

        # Fill in missing output configuration
        if not self._session_properties.audio.output:
            self._session_properties.audio.output = events.AudioOutput(
                format=events.PCMAudioFormat(rate=frame.audio_out_sample_rate)
            )

        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close WebSocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close WebSocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    #
    # Frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
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
        elif isinstance(frame, LLMUpdateSettingsFrame):
            self._session_properties = events.SessionProperties(**frame.settings)
            await self._update_settings()
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()

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
        """Handle appending messages to the context."""
        logger.warning("LLMMessagesAppendFrame not yet implemented for Grok Realtime")

    #
    # WebSocket communication
    #

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event to the Grok Voice Agent API.

        Args:
            event: The client event to send.
        """
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        """Establish WebSocket connection to Grok."""
        try:
            if self._websocket:
                return

            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Grok: {e}", exception=e)
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

    async def _update_settings(self):
        """Update session settings on the server."""
        settings = self._session_properties
        adapter: GrokRealtimeLLMAdapter = self.get_llm_adapter()

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
                evt = events.parse_server_event(message)
            except Exception as e:
                logger.warning(f"Failed to parse server event: {e}")
                continue

            if evt.type == "ping":
                # Ignore ping events (keep-alive)
                pass
            elif evt.type == "conversation.created":
                await self._handle_evt_conversation_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.created":
                await self._handle_evt_response_created(evt)
            elif evt.type == "response.output_audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.output_audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type == "response.content_part.added":
                # Content part added - we can ignore this for now
                pass
            elif evt.type == "response.content_part.done":
                # Content part done - we can ignore this for now
                pass
            elif evt.type == "response.output_item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "response.output_item.done":
                # Output item done - we can ignore this for now
                pass
            elif evt.type == "conversation.item.added":
                await self._handle_evt_conversation_item_added(evt)
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
                # Function call arguments streaming - we wait for the .done event
                pass
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                await self._handle_evt_error(evt)
                return

    async def _handle_evt_conversation_created(self, evt):
        """Handle conversation.created event - first event after connecting."""
        await self._update_settings()

    async def _handle_evt_response_created(self, evt):
        """Handle response.created event - response generation started."""
        pass

    async def _handle_evt_session_updated(self, evt):
        """Handle session.updated event."""
        self._api_session_ready = True
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_response()

    async def _handle_evt_audio_delta(self, evt):
        """Handle audio delta event - streaming audio from assistant."""
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
            # Track this function call for when arguments are completed
            # Only add if not already tracked (prevent duplicates)
            if evt.item.call_id not in self._pending_function_calls:
                self._pending_function_calls[evt.item.call_id] = evt.item
            else:
                # Grok may send multiple conversation.item.added events for the same function call
                logger.debug(f"Function call {evt.item.call_id} already tracked, skipping")

        await self._call_event_handler("on_conversation_item_created", evt.item.id, evt.item)

        if self._messages_added_manually.get(evt.item.id):
            del self._messages_added_manually[evt.item.id]
            return

        if evt.item.role == "assistant":
            self._current_assistant_response = evt.item
            await self.push_frame(LLMFullResponseStartFrame())

    async def _handle_evt_input_audio_transcription_completed(self, evt):
        """Handle input audio transcription completed event."""
        await self._call_event_handler("on_conversation_item_updated", evt.item_id, None)

        # Only push transcription if we have actual text (not empty or just whitespace)
        transcript = evt.transcript.strip() if evt.transcript else ""
        if transcript:
            await self.push_frame(
                TranscriptionFrame(transcript, "", time_now_iso8601(), result=evt),
                FrameDirection.UPSTREAM,
            )

    async def _handle_evt_response_done(self, evt):
        """Handle response.done event."""
        # Usage metrics - check both response.usage and top-level usage
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

        # Error handling
        if evt.response.status == "failed":
            error_msg = "Response failed"
            if evt.response.status_details:
                error_msg = str(evt.response.status_details)
            await self.push_error(error_msg=error_msg)
            return

        # Update conversation items
        for item in evt.response.output:
            await self._call_event_handler("on_conversation_item_updated", item.id, item)

    async def _handle_evt_audio_transcript_delta(self, evt):
        """Handle audio transcript delta event."""
        if evt.delta:
            await self._push_output_transcript_text_frames(evt.delta)

    async def _push_output_transcript_text_frames(self, text: str):
        # In a typical "cascade" LLM + TTS setup, LLMTextFrames would not
        # proceed beyond the TTS service. Therefore, since a speech-to-speech
        # service like Grok Realtime combines both LLM and TTS functionality,
        # you might think we wouldn't need to push LLMTextFrames at all.
        # However, RTVI relies on LLMTextFrames being pushed to trigger its
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

    async def _handle_evt_function_call_arguments_done(self, evt):
        """Handle function call arguments done event."""
        try:
            args = json.loads(evt.arguments)

            function_call_item = self._pending_function_calls.get(evt.call_id)
            if function_call_item:
                del self._pending_function_calls[evt.call_id]

                function_calls = [
                    FunctionCallFromLLM(
                        context=self._context,
                        tool_call_id=evt.call_id,
                        function_name=evt.name,
                        arguments=args,
                    )
                ]

                await self.run_function_calls(function_calls)
                logger.debug(f"Processed function call: {evt.name}")
            else:
                logger.warning(f"No tracked function call found for call_id: {evt.call_id}")

        except Exception as e:
            logger.error(f"Failed to process function call arguments: {e}")

    async def _handle_evt_speech_started(self, evt):
        """Handle speech started event from VAD."""
        await self._truncate_current_audio_response()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        await self.push_interruption_task_frame_and_wait()

    async def _handle_evt_speech_stopped(self, evt):
        """Handle speech stopped event from VAD."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    async def _handle_evt_error(self, evt):
        """Handle error event."""
        await self.push_error(error_msg=f"Grok Realtime Error: {evt.error.message}")

    #
    # Response creation
    #

    async def reset_conversation(self):
        """Reset the conversation by disconnecting and reconnecting."""
        logger.debug("Resetting Grok conversation")
        await self._disconnect()

        self._llm_needs_conversation_setup = True
        await self._process_completed_function_calls(send_new_results=False)

        await self._connect()

    async def _create_response(self):
        """Create an assistant response."""
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        adapter: GrokRealtimeLLMAdapter = self.get_llm_adapter()

        if self._llm_needs_conversation_setup:
            logger.debug(
                f"Setting up Grok conversation with initial messages: "
                f"{adapter.get_messages_for_logging(self._context)}"
            )

            llm_invocation_params = adapter.get_llm_invocation_params(self._context)
            messages = llm_invocation_params["messages"]

            for item in messages:
                evt = events.ConversationItemCreateEvent(item=item)
                self._messages_added_manually[evt.item.id] = True
                await self.send_client_event(evt)

            await self._update_settings()
            self._llm_needs_conversation_setup = False

        logger.debug("Creating Grok response")

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        await self.send_client_event(
            events.ResponseCreateEvent(
                response=events.ResponseProperties(modalities=["text", "audio"])
            )
        )

    async def _process_completed_function_calls(self, send_new_results: bool):
        """Process completed function calls and send results to the service."""
        sent_new_result = False

        for message in self._context.get_messages():
            if message.get("role") and message.get("content") != "IN_PROGRESS":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in self._completed_tool_calls:
                    if send_new_results:
                        sent_new_result = True
                        await self._send_tool_result(tool_call_id, message.get("content"))
                    self._completed_tool_calls.add(tool_call_id)

        if sent_new_result:
            await self._create_response()

    async def _send_user_audio(self, frame):
        """Send user audio to Grok."""
        # Don't send audio if conversation setup is still pending, as it can
        # lead to errors. For example: audio sent before conversation setup
        # will be interpreted as having Grok's default sample rate (24000),
        # and if that differs from the sample rate we eventually set through
        # the conversation setup, Grok will error out.
        if self._llm_needs_conversation_setup:
            return

        payload = base64.b64encode(frame.audio).decode("utf-8")
        await self.send_client_event(events.InputAudioBufferAppendEvent(audio=payload))

    async def _send_tool_result(self, tool_call_id: str, result: str):
        """Send a tool call result to Grok."""
        item = events.ConversationItem(
            type="function_call_output",
            call_id=tool_call_id,
            output=json.dumps(result),
        )
        await self.send_client_event(events.ConversationItemCreateEvent(item=item))

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> LLMContextAggregatorPair:
        """Create context aggregators for the Grok Realtime service.

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

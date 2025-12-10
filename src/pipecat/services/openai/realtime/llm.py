#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime LLM service implementation with WebSocket support."""

import base64
import json
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_realtime_adapter import (
    OpenAIRealtimeLLMAdapter,
)
from pipecat.frames.frames import (
    AggregationType,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
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
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_openai_realtime, traced_stt

from . import events

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use OpenAI, you need to `pip install pipecat-ai[openai]`.")
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


class OpenAIRealtimeLLMService(LLMService):
    """OpenAI Realtime LLM service providing real-time audio and text communication.

    Implements the OpenAI Realtime API with WebSocket communication for low-latency
    bidirectional audio and text interactions. Supports function calling, conversation
    management, and real-time transcription.
    """

    # Overriding the default adapter to use the OpenAIRealtimeLLMAdapter one.
    adapter_class = OpenAIRealtimeLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-realtime",
        base_url: str = "wss://api.openai.com/v1/realtime",
        session_properties: Optional[events.SessionProperties] = None,
        start_audio_paused: bool = False,
        send_transcription_frames: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize the OpenAI Realtime LLM service.

        Args:
            api_key: OpenAI API key for authentication.
            model: OpenAI model name. Defaults to "gpt-realtime".
                This is a connection-level parameter set via the WebSocket URL query
                parameter and cannot be changed during the session.
            base_url: WebSocket base URL for the realtime API.
                Defaults to "wss://api.openai.com/v1/realtime".
            session_properties: Configuration properties for the realtime session.
                These are session-level settings that can be updated during the session
                (except for voice and model). If None, uses default SessionProperties.
            start_audio_paused: Whether to start with audio input paused. Defaults to False.
            send_transcription_frames: Whether to emit transcription frames.

                .. deprecated:: 0.0.92
                    This parameter is deprecated and will be removed in a future version.
                    Transcription frames are always sent.

            **kwargs: Additional arguments passed to parent LLMService.
        """
        if send_transcription_frames is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`send_transcription_frames` is deprecated and will be removed in a future version. "
                    "Transcription frames are always sent.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Build WebSocket URL with model query parameter
        # Source: https://platform.openai.com/docs/guides/realtime-websocket
        full_url = f"{base_url}?model={model}"
        super().__init__(base_url=full_url, **kwargs)

        self.api_key = api_key
        self.base_url = full_url
        self.set_model_name(model)

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
        self._pending_function_calls = {}  # Track function calls by call_id
        self._completed_tool_calls = set()

        self._register_event_handler("on_conversation_item_created")
        self._register_event_handler("on_conversation_item_updated")
        self._retrieve_conversation_item_futures = {}

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

    def _is_modality_enabled(self, modality: str) -> bool:
        """Check if a specific modality is enabled, "text" or "audio"."""
        modalities = self._session_properties.output_modalities or ["audio", "text"]
        return modality in modalities

    def _get_enabled_modalities(self) -> list[str]:
        """Get the list of enabled modalities."""
        modalities = self._session_properties.output_modalities or ["audio", "text"]
        # API only supports single modality responses: either ["text"] or ["audio"]
        if "audio" in modalities:
            return ["audio"]
        elif "text" in modalities:
            return ["text"]

    async def retrieve_conversation_item(self, item_id: str):
        """Retrieve a conversation item by ID from the server.

        Args:
            item_id: The ID of the conversation item to retrieve.

        Returns:
            The retrieved conversation item.
        """
        future = self.get_event_loop().create_future()
        retrieval_in_flight = False
        if not self._retrieve_conversation_item_futures.get(item_id):
            self._retrieve_conversation_item_futures[item_id] = []
        else:
            retrieval_in_flight = True
        self._retrieve_conversation_item_futures[item_id].append(future)
        if not retrieval_in_flight:
            await self.send_client_event(
                # Set event_id to "rci_{item_id}" so that we can identify an
                # error later if the retrieval fails. We don't need a UUID
                # suffix to the event_id because we're ensuring only one
                # in-flight retrieval per item_id. (Note: "rci" = "retrieve
                # conversation item")
                events.ConversationItemRetrieveEvent(item_id=item_id, event_id=f"rci_{item_id}")
            )
        return await future

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)
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
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        # None and False are different. Check for False. None means we're using OpenAI's
        # built-in turn detection defaults.
        turn_detection_disabled = (
            self._session_properties.audio
            and self._session_properties.audio.input
            and self._session_properties.audio.input.turn_detection is False
        )
        if turn_detection_disabled:
            await self.send_client_event(events.InputAudioBufferClearEvent())
            await self.send_client_event(events.ResponseCancelEvent())
        await self._truncate_current_audio_response()
        await self.stop_all_metrics()
        if self._current_assistant_response:
            await self.push_frame(LLMFullResponseEndFrame())
            # Only push TTSStoppedFrame if audio modality is enabled
            if self._is_modality_enabled("audio"):
                await self.push_frame(TTSStoppedFrame())

    async def _handle_user_started_speaking(self, frame):
        pass

    async def _handle_user_stopped_speaking(self, frame):
        # None and False are different. Check for False. None means we're using OpenAI's
        # built-in turn detection defaults.
        turn_detection_disabled = (
            self._session_properties.audio
            and self._session_properties.audio.input
            and self._session_properties.audio.input.turn_detection is False
        )
        if turn_detection_disabled:
            await self.send_client_event(events.InputAudioBufferCommitEvent())
            await self.send_client_event(events.ResponseCreateEvent())

    async def _handle_bot_stopped_speaking(self):
        self._current_audio_response = None

    def _calculate_audio_duration_ms(
        self, total_bytes: int, sample_rate: int = 24000, bytes_per_sample: int = 2
    ) -> int:
        """Calculate audio duration in milliseconds based on PCM audio parameters."""
        samples = total_bytes / bytes_per_sample
        duration_seconds = samples / sample_rate
        return int(duration_seconds * 1000)

    async def _truncate_current_audio_response(self):
        """Truncates the current audio response at the appropriate duration.

        Calculates the actual duration of the audio content and truncates at the shorter of
        either the wall clock time or the actual audio duration to prevent invalid truncation
        requests.
        """
        if not self._current_audio_response:
            return

        # if the bot is still speaking, truncate the last message
        try:
            current = self._current_audio_response
            self._current_audio_response = None

            # Calculate actual audio duration instead of using wall clock time
            audio_duration_ms = self._calculate_audio_duration_ms(current.total_size)

            # Use the shorter of wall clock time or actual audio duration
            elapsed_ms = int(time.time() * 1000 - current.start_time_ms)
            truncate_ms = min(elapsed_ms, audio_duration_ms)

            logger.trace(
                f"Truncating audio: duration={audio_duration_ms}ms, "
                f"elapsed={elapsed_ms}ms, truncate={truncate_ms}ms"
            )

            await self.send_client_event(
                events.ConversationItemTruncateEvent(
                    item_id=current.item_id,
                    content_index=current.content_index,
                    audio_end_ms=truncate_ms,
                )
            )
        except Exception as e:
            # Log warning and don't re-raise - allow session to continue
            logger.warning(f"Audio truncation failed (non-fatal): {e}")

    #
    # frame processing
    #
    # StartFrame, StopFrame, CancelFrame implemented in base class
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
        elif isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            context = (
                frame.context
                if isinstance(frame, LLMContextFrame)
                else LLMContext.from_openai_context(frame.context)
            )
            await self._handle_context(context)
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
        if not self._context:
            # We got our initial context
            self._context = context
            # Initialize our bookkeeping of already-completed tool calls in
            # the context
            await self._process_completed_function_calls(send_new_results=False)
            # Run the LLM at next opportunity
            await self._create_response()
        else:
            # We got an updated context.
            # This may contain a new user message or tool call result.
            self._context = context
            # Send results for newly-completed function calls, if any.
            await self._process_completed_function_calls(send_new_results=True)

    async def _handle_messages_append(self, frame):
        logger.error("!!! NEED TO IMPLEMENT MESSAGES APPEND")

    #
    # websocket communication
    #

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event to the OpenAI Realtime API.

        Args:
            event: The client event to send.
        """
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None

    async def _disconnect(self):
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
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(realtime_message))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                # We're in the process of disconnecting.
                # (If not self._websocket, that could indicate that we
                # somehow *started* the websocket send attempt while we still
                # had a connection)
                return
            # In server-to-server contexts, a WebSocket error should be quite rare. Given how hard
            # it is to recover from a send-side error with proper state management, and that exponential
            # backoff for retries can have cost/stability implications for a service cluster, let's just
            # treat a send-side error as fatal.
            await self.push_error(error_msg=f"Error sending client event: {e}", exception=e)

    async def _update_settings(self):
        settings = self._session_properties
        adapter: OpenAIRealtimeLLMAdapter = self.get_llm_adapter()

        if self._context:
            llm_invocation_params = adapter.get_llm_invocation_params(self._context)

            # tools given in the context override the tools in the session properties
            if llm_invocation_params["tools"]:
                settings.tools = llm_invocation_params["tools"]

            # instructions in the context come from an initial "system" message in the
            # messages list, and override instructions in the session properties
            if llm_invocation_params["system_instruction"]:
                settings.instructions = llm_invocation_params["system_instruction"]

        # If needed, map settings.tools from ToolsSchema to list of dicts,
        # which remote server expects. It would only be a ToolsSchema if that's
        # how it was provided in the constructor or via LLMUpdateSettingsFrame.
        if settings.tools and isinstance(settings.tools, ToolsSchema):
            settings.tools = adapter.from_standard_tools(settings.tools)

        await self.send_client_event(events.SessionUpdateEvent(session=settings))

    #
    # inbound server event handling
    # https://platform.openai.com/docs/api-reference/realtime-server-events
    #

    async def _receive_task_handler(self):
        async for message in self._websocket:
            evt = events.parse_server_event(message)
            if evt.type == "session.created":
                await self._handle_evt_session_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.output_audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.output_audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type == "conversation.item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "conversation.item.done":
                await self._handle_evt_conversation_item_done(evt)
            elif evt.type == "conversation.item.input_audio_transcription.delta":
                await self._handle_evt_input_audio_transcription_delta(evt)
            elif evt.type == "conversation.item.input_audio_transcription.completed":
                await self.handle_evt_input_audio_transcription_completed(evt)
            elif evt.type == "conversation.item.retrieved":
                await self._handle_conversation_item_retrieved(evt)
            elif evt.type == "response.done":
                await self._handle_evt_response_done(evt)
            elif evt.type == "input_audio_buffer.speech_started":
                await self._handle_evt_speech_started(evt)
            elif evt.type == "input_audio_buffer.speech_stopped":
                await self._handle_evt_speech_stopped(evt)
            elif evt.type == "response.output_text.delta":
                await self._handle_evt_text_delta(evt)
            elif evt.type == "response.output_audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                    await self._handle_evt_error(evt)
                    # errors are fatal, so exit the receive loop
                    return

    @traced_openai_realtime(operation="llm_setup")
    async def _handle_evt_session_created(self, evt):
        # session.created is received right after connecting. Send a message
        # to configure the session properties.
        await self._update_settings()

    async def _handle_evt_session_updated(self, evt):
        # If this is our first context frame, run the LLM
        self._api_session_ready = True
        # Now that we've configured the session, we can run the LLM if we need to.
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_response()

    async def _handle_evt_audio_delta(self, evt):
        # note: ttfb is faster by 1/2 RTT than ttfb as measured for other services, since we're getting
        # this event from the server
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
            sample_rate=24000,
            num_channels=1,
        )
        await self.push_frame(frame)

    async def _handle_evt_audio_done(self, evt):
        if self._current_audio_response:
            await self.push_frame(TTSStoppedFrame())
            # Don't clear the self._current_audio_response here. We need to wait until we
            # receive a BotStoppedSpeakingFrame from the output transport.

    async def _handle_evt_conversation_item_added(self, evt):
        """Handle conversation.item.added event - item is added but may still be processing."""
        if evt.item.type == "function_call":
            # Track this function call for when arguments are completed
            # Only add if not already tracked (prevent duplicates)
            if evt.item.call_id not in self._pending_function_calls:
                self._pending_function_calls[evt.item.call_id] = evt.item
            else:
                logger.warning(f"Function call {evt.item.call_id} already tracked, skipping")

        await self._call_event_handler("on_conversation_item_created", evt.item.id, evt.item)

        # This will get sent from the server every time a new "message" is added
        # to the server's conversation state, whether we create it via the API
        # or the server creates it from LLM output.
        if self._messages_added_manually.get(evt.item.id):
            del self._messages_added_manually[evt.item.id]
            return

        if evt.item.role == "assistant":
            self._current_assistant_response = evt.item
            await self.push_frame(LLMFullResponseStartFrame())

    async def _handle_evt_conversation_item_done(self, evt):
        """Handle conversation.item.done event - item is fully completed."""
        await self._call_event_handler("on_conversation_item_updated", evt.item.id, evt.item)
        # The item is now fully processed and ready
        # For now, no additional logic needed beyond the event handler call

    async def _handle_evt_input_audio_transcription_delta(self, evt):
        await self.push_frame(
            # no way to get a language code?
            InterimTranscriptionFrame(evt.delta, "", time_now_iso8601(), result=evt),
            direction=FrameDirection.UPSTREAM,
        )

    @traced_stt
    async def _handle_user_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def handle_evt_input_audio_transcription_completed(self, evt):
        """Handle completion of input audio transcription.

        Args:
            evt: The transcription completed event.
        """
        await self._call_event_handler("on_conversation_item_updated", evt.item_id, None)

        await self.push_frame(
            # no way to get a language code?
            TranscriptionFrame(evt.transcript, "", time_now_iso8601(), result=evt),
            FrameDirection.UPSTREAM,
        )
        await self._handle_user_transcription(evt.transcript, True, Language.EN)

    async def _handle_conversation_item_retrieved(self, evt: events.ConversationItemRetrieved):
        futures = self._retrieve_conversation_item_futures.pop(evt.item.id, None)
        if futures:
            for future in futures:
                future.set_result(evt.item)

    @traced_openai_realtime(operation="llm_response")
    async def _handle_evt_response_done(self, evt):
        # todo: figure out whether there's anything we need to do for "cancelled" events
        # usage metrics
        cached_tokens = (
            evt.response.usage.input_token_details.cached_tokens
            if hasattr(evt.response.usage, "input_token_details")
            and evt.response.usage.input_token_details
            else None
        )
        tokens = LLMTokenUsage(
            prompt_tokens=evt.response.usage.input_tokens,
            completion_tokens=evt.response.usage.output_tokens,
            total_tokens=evt.response.usage.total_tokens,
            cache_read_input_tokens=cached_tokens,
        )
        await self.start_llm_usage_metrics(tokens)
        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
        self._current_assistant_response = None
        # error handling
        if evt.response.status == "failed":
            await self.push_error(error_msg=evt.response.status_details["error"]["message"])
            return
        # response content
        for item in evt.response.output:
            await self._call_event_handler("on_conversation_item_updated", item.id, item)

    async def _handle_evt_text_delta(self, evt):
        # We receive text deltas (as opposed to audio transcript deltas) when
        # the output modality is "text"
        if evt.delta:
            frame = LLMTextFrame(evt.delta)
            await self.push_frame(frame)

    async def _handle_evt_audio_transcript_delta(self, evt):
        # We receive audio transcript deltas (as opposed to text deltas) when
        # the output modality is "audio" (the default)
        if evt.delta:
            frame = TTSTextFrame(evt.delta, aggregated_by=AggregationType.SENTENCE)
            # OpenAI Realtime text already includes any necessary inter-chunk spaces
            frame.includes_inter_frame_spaces = True
            await self.push_frame(frame)

    async def _handle_evt_function_call_arguments_done(self, evt):
        """Handle completion of function call arguments.

        Args:
            evt: The response.function_call_arguments.done event.
        """
        # Process the function call immediately when arguments are complete
        # This is needed because function calls might not trigger response.done
        try:
            # Parse the arguments
            args = json.loads(evt.arguments)

            # Get the function call item we tracked earlier
            function_call_item = self._pending_function_calls.get(evt.call_id)
            if function_call_item:
                # Remove from pending calls FIRST to prevent duplicate processing
                del self._pending_function_calls[evt.call_id]

                # Create the function call and process it
                function_calls = [
                    FunctionCallFromLLM(
                        context=self._context,
                        tool_call_id=evt.call_id,
                        function_name=function_call_item.name,
                        arguments=args,
                    )
                ]

                await self.run_function_calls(function_calls)
                logger.debug(f"Processed function call: {function_call_item.name}")
            else:
                logger.warning(f"No tracked function call found for call_id: {evt.call_id}")
                logger.warning(
                    f"Available pending calls: {list(self._pending_function_calls.keys())}"
                )

        except Exception as e:
            logger.error(f"Failed to process function call arguments: {e}")

    async def _handle_evt_speech_started(self, evt):
        await self._truncate_current_audio_response()
        await self.push_interruption_task_frame_and_wait()
        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_evt_speech_stopped(self, evt):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _maybe_handle_evt_retrieve_conversation_item_error(self, evt: events.ErrorEvent):
        """Maybe handle an error event related to retrieving a conversation item.

        If the given error event is an error retrieving a conversation item:

        - set an exception on the future that retrieve_conversation_item() is waiting on
        - return true
        Otherwise:
        - return false
        """
        if evt.error.code == "item_retrieve_invalid_item_id":
            item_id = evt.error.event_id.split("_", 1)[1]  # event_id is of the form "rci_{item_id}"
            futures = self._retrieve_conversation_item_futures.pop(item_id, None)
            if futures:
                for future in futures:
                    future.set_exception(Exception(evt.error.message))
            return True
        return False

    async def _handle_evt_error(self, evt):
        # Errors are fatal to this connection. Send an ErrorFrame.
        await self.push_error(error_msg=f"Error: {evt}")

    #
    # state and client events for the current conversation
    # https://platform.openai.com/docs/api-reference/realtime-client-events
    #

    async def reset_conversation(self):
        """Reset the conversation by disconnecting and reconnecting.

        This is the safest way to start a new conversation. Note that this will
        fail if called from the receive task.
        """
        logger.debug("Resetting conversation")
        await self._disconnect()

        # Prepare to setup server-side conversation from local context again
        self._llm_needs_conversation_setup = True
        await self._process_completed_function_calls(send_new_results=False)

        await self._connect()

    @traced_openai_realtime(operation="llm_request")
    async def _create_response(self):
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        adapter: OpenAIRealtimeLLMAdapter = self.get_llm_adapter()

        # Configure the LLM for this session if needed
        if self._llm_needs_conversation_setup:
            logger.debug(
                f"Setting up conversation on OpenAI Realtime LLM service with initial messages: {adapter.get_messages_for_logging(self._context)}"
            )

            # Send initial messages
            llm_invocation_params = adapter.get_llm_invocation_params(self._context)
            messages = llm_invocation_params["messages"]
            for item in messages:
                evt = events.ConversationItemCreateEvent(item=item)
                self._messages_added_manually[evt.item.id] = True
                await self.send_client_event(evt)

            # Send new settings if needed
            await self._update_settings()

            # We're done configuring the LLM for this session
            self._llm_needs_conversation_setup = False

        logger.debug("Creating response")

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        await self.send_client_event(
            events.ResponseCreateEvent(
                response=events.ResponseProperties(output_modalities=self._get_enabled_modalities())
            )
        )

    async def _process_completed_function_calls(self, send_new_results: bool):
        # Check for set of completed function calls in the context
        sent_new_result = False
        for message in self._context.get_messages():
            if message.get("role") and message.get("content") != "IN_PROGRESS":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in self._completed_tool_calls:
                    # Found a newly-completed function call - send the result to the service
                    if send_new_results:
                        sent_new_result = True
                        await self._send_tool_result(tool_call_id, message.get("content"))
                    self._completed_tool_calls.add(tool_call_id)

        # If we reported any new tool call results to the service, trigger
        # another response
        if sent_new_result:
            await self._create_response()

    async def _send_user_audio(self, frame):
        payload = base64.b64encode(frame.audio).decode("utf-8")
        await self.send_client_event(events.InputAudioBufferAppendEvent(audio=payload))

    async def _send_tool_result(self, tool_call_id: str, result: str):
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
        """Create an instance of OpenAIContextAggregatorPair from an OpenAILLMContext.

        NOTE: this method exists only for backward compatibility. New code
        should instead do::

            context = LLMContext(...)
            context_aggregator = LLMContextAggregatorPair(context)

        Constructor keyword arguments for both the user and assistant aggregators can be provided.

        Args:
            context: The LLM context.
            user_params: User aggregator parameters.
            assistant_params: Assistant aggregator parameters.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.
        """
        # Log warning about transcription frame direction change in 0.0.92.
        # We're putting this warning here rather than in the constructor so
        # that it shows up for folks who haven't updated their code at all
        # since 0.0.92, gives them a way to acknowledge and dismiss the
        # warning, and encourages adoption of a new preferred pattern.
        logger.warning(
            "As of version 0.0.92, TranscriptionFrames and InterimTranscriptionFrames "
            "now go upstream from OpenAIRealtimeLLMService, so if you're using "
            "TranscriptProcessor, say, you'll want to adjust accordingly:\n\n"
            "pipeline = Pipeline(\n"
            "  [\n"
            "    transport.input(),\n"
            "    context_aggregator.user(),\n\n"
            "    # BEFORE\n"
            "    llm,\n"
            "    transcript.user(),\n\n"
            "    # AFTER\n"
            "    transcript.user(),\n"
            "    llm,\n\n"
            "    transport.output(),\n"
            "    transcript.assistant(),\n"
            "    context_aggregator.assistant(),\n"
            "  ]\n"
            ")\n\n"
            "Also, LLMTextFrames are no longer pushed from "
            "OpenAIRealtimeLLMService when it's configured with "
            "output_modalities=['audio']. Listen for TTSTextFrames instead.\n\n"
            "Once you've made the appropriate changes (if needed), you can "
            "dismiss this warning by updating to the new context-setup pattern:\n\n"
            "  context = LLMContext(messages, tools)\n"
            "  context_aggregator = LLMContextAggregatorPair(context)\n"
        )

        context = LLMContext.from_openai_context(context)
        assistant_params.expect_stripped_words = False
        return LLMContextAggregatorPair(
            context, user_params=user_params, assistant_params=assistant_params
        )

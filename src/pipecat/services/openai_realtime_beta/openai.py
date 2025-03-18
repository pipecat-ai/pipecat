#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from loguru import logger

from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.services.openai import OpenAIContextAggregatorPair
from pipecat.utils.time import time_now_iso8601

from . import events
from .context import (
    OpenAIRealtimeAssistantContextAggregator,
    OpenAIRealtimeLLMContext,
    OpenAIRealtimeUserContextAggregator,
)
from .frames import RealtimeFunctionCallResultFrame, RealtimeMessagesUpdateFrame


@dataclass
class CurrentAudioResponse:
    item_id: str
    content_index: int
    start_time_ms: int
    total_size: int = 0


class OpenAIUnhandledFunctionException(Exception):
    pass


class OpenAIRealtimeBetaLLMService(LLMService):
    # Overriding the default adapter to use the OpenAIRealtimeLLMAdapter one.
    adapter_class = OpenAIRealtimeLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        base_url: str = "wss://api.openai.com/v1/realtime",
        session_properties: events.SessionProperties = events.SessionProperties(),
        start_audio_paused: bool = False,
        send_transcription_frames: bool = True,
        **kwargs,
    ):
        full_url = f"{base_url}?model={model}"
        super().__init__(base_url=full_url, **kwargs)
        self.api_key = api_key
        self.base_url = full_url

        self._session_properties: events.SessionProperties = session_properties
        self._audio_input_paused = start_audio_paused
        self._send_transcription_frames = send_transcription_frames
        self._websocket = None
        self._receive_task = None
        self._context = None

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._current_assistant_response = None
        self._current_audio_response = None

        self._messages_added_manually = {}
        self._user_and_response_message_tuple = None

    def can_generate_metrics(self) -> bool:
        return True

    def set_audio_input_paused(self, paused: bool):
        self._audio_input_paused = paused

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    #
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        # None and False are different. Check for False. None means we're using OpenAI's
        # built-in turn detection defaults.
        if self._session_properties.turn_detection is False:
            await self.send_client_event(events.InputAudioBufferClearEvent())
            await self.send_client_event(events.ResponseCancelEvent())
        await self._truncate_current_audio_response()
        await self.stop_all_metrics()
        if self._current_assistant_response:
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(TTSStoppedFrame())

    async def _handle_user_started_speaking(self, frame):
        pass

    async def _handle_user_stopped_speaking(self, frame):
        # None and False are different. Check for False. None means we're using OpenAI's
        # built-in turn detection defaults.
        if self._session_properties.turn_detection is False:
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
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            pass
        elif isinstance(frame, OpenAILLMContextFrame):
            context: OpenAIRealtimeLLMContext = OpenAIRealtimeLLMContext.upgrade_to_realtime(
                frame.context
            )
            if not self._context:
                self._context = context
            elif frame.context is not self._context:
                # If the context has changed, reset the conversation
                self._context = context
                await self.reset_conversation()
            # Run the LLM at next opportunity
            await self._create_response()
        elif isinstance(frame, InputAudioRawFrame):
            if not self._audio_input_paused:
                await self._send_user_audio(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking()
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_messages_append(frame)
        elif isinstance(frame, RealtimeMessagesUpdateFrame):
            self._context = frame.context
        elif isinstance(frame, LLMUpdateSettingsFrame):
            self._session_properties = events.SessionProperties(**frame.settings)
            await self._update_settings()
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()
        elif isinstance(frame, RealtimeFunctionCallResultFrame):
            await self._handle_function_call_result(frame.result_frame)

        await self.push_frame(frame, direction)

    async def _handle_messages_append(self, frame):
        logger.error("!!! NEED TO IMPLEMENT MESSAGES APPEND")

    async def _handle_function_call_result(self, frame):
        item = events.ConversationItem(
            type="function_call_output",
            call_id=frame.tool_call_id,
            output=json.dumps(frame.result),
        )
        await self.send_client_event(events.ConversationItemCreateEvent(item=item))

    #
    # websocket communication
    #

    async def send_client_event(self, event: events.ClientEvent):
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return
            self._websocket = await websockets.connect(
                uri=self.base_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
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
            self._disconnecting = False
        except Exception as e:
            logger.error(f"{self} error disconnecting: {e}")

    async def _ws_send(self, realtime_message):
        try:
            if self._websocket:
                await self._websocket.send(json.dumps(realtime_message))
        except Exception as e:
            if self._disconnecting:
                return
            logger.error(f"Error sending message to websocket: {e}")
            # In server-to-server contexts, a WebSocket error should be quite rare. Given how hard
            # it is to recover from a send-side error with proper state management, and that exponential
            # backoff for retries can have cost/stability implications for a service cluster, let's just
            # treat a send-side error as fatal.
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

    async def _update_settings(self):
        settings = self._session_properties
        # tools given in the context override the tools in the session properties
        if self._context and self._context.tools:
            settings.tools = self._context.tools
        # instructions in the context come from an initial "system" message in the
        # messages list, and override instructions in the session properties
        if self._context and self._context._session_instructions:
            settings.instructions = self._context._session_instructions
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
            elif evt.type == "response.audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type == "conversation.item.created":
                await self._handle_evt_conversation_item_created(evt)
            elif evt.type == "conversation.item.input_audio_transcription.completed":
                await self.handle_evt_input_audio_transcription_completed(evt)
            elif evt.type == "response.done":
                await self._handle_evt_response_done(evt)
            elif evt.type == "input_audio_buffer.speech_started":
                await self._handle_evt_speech_started(evt)
            elif evt.type == "input_audio_buffer.speech_stopped":
                await self._handle_evt_speech_stopped(evt)
            elif evt.type == "response.audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "error":
                await self._handle_evt_error(evt)
                # errors are fatal, so exit the receive loop
                return

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

    async def _handle_evt_conversation_item_created(self, evt):
        # This will get sent from the server every time a new "message" is added
        # to the server's conversation state, whether we create it via the API
        # or the server creates it from LLM output.
        if self._messages_added_manually.get(evt.item.id):
            del self._messages_added_manually[evt.item.id]
            return

        if evt.item.role == "user":
            # We need to wait for completion of both user message and response message. Then we'll
            # add both to the context. User message is complete when we have a "transcript" field
            # that is not None. Response message is complete when we get a "response.done" event.
            self._user_and_response_message_tuple = (evt.item, {"done": False, "output": []})
        elif evt.item.role == "assistant":
            self._current_assistant_response = evt.item
            await self.push_frame(LLMFullResponseStartFrame())

    async def handle_evt_input_audio_transcription_completed(self, evt):
        if self._send_transcription_frames:
            await self.push_frame(
                # no way to get a language code?
                TranscriptionFrame(evt.transcript, "", time_now_iso8601())
            )
        pair = self._user_and_response_message_tuple
        if pair:
            user, assistant = pair
            user.content[0].transcript = evt.transcript
            if assistant["done"]:
                self._user_and_response_message_tuple = None
                self._context.add_user_content_item_as_message(user)
                await self._handle_assistant_output(assistant["output"])
        else:
            # User message without preceding conversation.item.created. Bug?
            logger.warning(f"Transcript for unknown user message: {evt}")

    async def _handle_evt_response_done(self, evt):
        # todo: figure out whether there's anything we need to do for "cancelled" events
        # usage metrics
        tokens = LLMTokenUsage(
            prompt_tokens=evt.response.usage.input_tokens,
            completion_tokens=evt.response.usage.output_tokens,
            total_tokens=evt.response.usage.total_tokens,
        )
        await self.start_llm_usage_metrics(tokens)
        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
        self._current_assistant_response = None
        # response content
        pair = self._user_and_response_message_tuple
        if pair:
            user, assistant = pair
            assistant["done"] = True
            assistant["output"] = evt.response.output
            if user.content[0].transcript is not None:
                self._user_and_response_message_tuple = None
                self._context.add_user_content_item_as_message(user)
                await self._handle_assistant_output(assistant["output"])
        else:
            # Response message without preceding user message. Add it to the context.
            await self._handle_assistant_output(evt.response.output)

    async def _handle_evt_audio_transcript_delta(self, evt):
        if evt.delta:
            await self.push_frame(LLMTextFrame(evt.delta))
            await self.push_frame(TTSTextFrame(evt.delta))

    async def _handle_evt_speech_started(self, evt):
        await self._truncate_current_audio_response()
        await self._start_interruption()  # cancels this processor task
        await self.push_frame(StartInterruptionFrame())  # cancels downstream tasks
        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_evt_speech_stopped(self, evt):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self._stop_interruption()
        await self.push_frame(StopInterruptionFrame())
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_evt_error(self, evt):
        # Errors are fatal to this connection. Send an ErrorFrame.
        await self.push_error(ErrorFrame(error=f"Error: {evt}", fatal=True))

    async def _handle_assistant_output(self, output):
        # logger.debug(f"!!! HANDLE Assistant output: {output}")
        # We haven't seen intermixed audio and function_call items in the same response. But let's
        # try to write logic that handles that, if it does happen.
        messages = [item for item in output if item.type == "message"]
        function_calls = [item for item in output if item.type == "function_call"]
        for item in messages:
            self._context.add_assistant_content_item_as_message(item)
        await self._handle_function_call_items(function_calls)

    async def _handle_function_call_items(self, items):
        total_items = len(items)
        for index, item in enumerate(items):
            function_name = item.name
            tool_id = item.call_id
            arguments = json.loads(item.arguments)
            if self.has_function(function_name):
                run_llm = index == total_items - 1
                if function_name in self._callbacks.keys():
                    await self.call_function(
                        context=self._context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                        run_llm=run_llm,
                    )
                elif None in self._callbacks.keys():
                    await self.call_function(
                        context=self._context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                        run_llm=run_llm,
                    )
            else:
                raise OpenAIUnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                )

    #
    # state and client events for the current conversation
    # https://platform.openai.com/docs/api-reference/realtime-client-events
    #

    async def reset_conversation(self):
        # Disconnect/reconnect is the safest way to start a new conversation.
        # Note that this will fail if called from the receive task.
        logger.debug("Resetting conversation")
        await self._disconnect()
        if self._context:
            self._context.llm_needs_settings_update = True
            self._context.llm_needs_initial_messages = True
        await self._connect()

    async def _create_response(self):
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        if self._context.llm_needs_initial_messages:
            messages = self._context.get_messages_for_initializing_history()
            for item in messages:
                evt = events.ConversationItemCreateEvent(item=item)
                self._messages_added_manually[evt.item.id] = True
                await self.send_client_event(evt)
            self._context.llm_needs_initial_messages = False

        if self._context.llm_needs_settings_update:
            await self._update_settings()
            self._context.llm_needs_settings_update = False

        logger.debug(f"Creating response: {self._context.get_messages_for_logging()}")

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        await self.send_client_event(
            events.ResponseCreateEvent(
                response=events.ResponseProperties(modalities=["audio", "text"])
            )
        )

    async def _send_user_audio(self, frame):
        payload = base64.b64encode(frame.audio).decode("utf-8")
        await self.send_client_event(events.InputAudioBufferAppendEvent(audio=payload))

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_kwargs: Mapping[str, Any] = {},
        assistant_kwargs: Mapping[str, Any] = {},
    ) -> OpenAIContextAggregatorPair:
        """Create an instance of OpenAIContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_kwargs (Mapping[str, Any], optional): Additional keyword
                arguments for the user context aggregator constructor. Defaults
                to an empty mapping.
            assistant_kwargs (Mapping[str, Any], optional): Additional keyword
                arguments for the assistant context aggregator
                constructor. Defaults to an empty mapping.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        OpenAIRealtimeLLMContext.upgrade_to_realtime(context)
        user = OpenAIRealtimeUserContextAggregator(context, **user_kwargs)

        default_assistant_kwargs = {"expect_stripped_words": False}
        default_assistant_kwargs.update(assistant_kwargs)
        assistant = OpenAIRealtimeAssistantContextAggregator(context, **default_assistant_kwargs)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

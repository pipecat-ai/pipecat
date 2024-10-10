import asyncio
import base64
import json

# temp: websocket logger
import logging
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import websockets
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    DataFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
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
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAIContextAggregatorPair,
    OpenAIUserContextAggregator,
)
from pipecat.utils.time import time_now_iso8601

from . import events

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
)


@dataclass
class _InternalMessagesUpdateFrame(DataFrame):
    context: "OpenAIRealtimeLLMContext"


class OpenAIUnhandledFunctionException(Exception):
    pass


class OpenAIRealtimeLLMContext(OpenAILLMContext):
    def __init__(self, messages=None, tools=None, **kwargs):
        super().__init__(messages=messages, tools=tools, **kwargs)
        self.__setup_local()

    def __setup_local(self):
        # messages that have been added to the context but not yet sent to the openai server
        self._unsent_messages = deepcopy(self._messages)
        # messages that we added to the context because they were part of our external
        # context store. we do not want to add these again when we see conversation.item.created
        # events about them. map from item_id to True
        self._manually_created_messages = {}
        # "conversation items" that have been created by opeanai realtime api events but are
        # not completely filled in, yet. map from item_id to message
        self._messages_in_progress = {}
        # count of messages prior to recent reset
        self._messages_reset_count = 0
        self._tools_list_updated = True

    @staticmethod
    def upgrade_to_realtime(obj: OpenAILLMContext) -> "OpenAIRealtimeLLMContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, OpenAIRealtimeLLMContext):
            obj.__class__ = OpenAIRealtimeLLMContext
            obj.__setup_local()
        return obj

    # still working on
    #   - clearing the context by deleting all messages
    #   - reloading from a standard messages list
    #   - truncating the last spoken message to maintain context when interrupted

    def set_tools(self, tools: List):
        super().set_tools(tools)
        self._tools_list_updated = True

    def add_message(self, message):
        super().add_message(message)
        self._unsent_messages.append(message)
        return message

    def add_messages(self, messages):
        super().add_messages(messages)
        self._unsent_messages.extend(messages)

    def add_message_already_present_in_api_context(self, message):
        super().add_message(message)
        return message

    def set_messages(self, messages):
        self._messages_reset_count = len(self.messages) - len(self._unsent_messages)
        super().set_messages(messages)
        self._unsent_messages = deepcopy(self._messages)

    def get_unsent_messages(self):
        return self._unsent_messages

    def get_messages_reset_count(self):
        return self._messages_reset_count

    def get_tools_list_updated(self):
        return self._tools_list_updated

    def update_all_messages_sent(self):
        self._unsent_messages = []
        self._messages_reset_count = 0

    def update_tools_list_sent(self):
        self._tools_list_updated = False

    def note_manually_added_message(self, item_id):
        self._manually_created_messages[item_id] = True

    def add_message_from_realtime_event(self, evt):
        if evt.item.id in self._manually_created_messages:
            del self._manually_created_messages[evt.item.id]
            return

        # add messages. don't add function_call or function_call_output items.
        if evt.item.type == "message":
            message = self.add_message_already_present_in_api_context(
                {"role": evt.item.role, "content": []}
            )
            if not evt.item.content:
                self._messages_in_progress[evt.item.id] = message
                return
            for content in evt.item.content:
                message["content"].append({"type": content.type})
                if content.text:
                    message["content"] = content.text
                elif content.transcript:
                    message["content"] = content.transcript
                else:
                    # we will get the transcript in a later event
                    self._messages_in_progress[evt.item.id] = message
            return

    def add_transcript_to_message(self, evt):
        message = self._messages_in_progress.get(evt.item_id)
        if message:
            cs = message["content"]
            cs.extend([{"type": ""}] * (evt.content_index - len(cs) + 1))
            cs[evt.content_index] = {"type": "text", "text": evt.transcript}
            del self._messages_in_progress[evt.item_id]
        else:
            logger.error(
                f"Could not find content {evt.item_id}/{evt.content_index} to add transcript to"
            )


class OpenAIRealtimeUserContextAggregator(OpenAIUserContextAggregator):
    async def process_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        await super().process_frame(frame, direction)
        # Parent does not push LLMMessagesUpdateFrame. This ensures that in a typical pipeline,
        # messages are only processed by the user context aggregator, which is generally what we want. But
        # we also need to send new messages over the websocket, in case audio mode triggers a response before
        # we get any other context frames through the pipeline.
        if isinstance(frame, LLMMessagesUpdateFrame):
            await self.push_frame(_InternalMessagesUpdateFrame(context=self._context))

        # Parent also doesn't push the LLMSetToolsFrame.
        if isinstance(frame, LLMSetToolsFrame):
            await self.push_frame(frame, direction)

    async def _push_aggregation(self):
        # for the moment, ignore all user input coming into the pipeline.
        # todo: think about whether/how to fix this to allow for text input from
        #       upstream (transport/transcription, or other sources)
        pass


class OpenAIRealtimeAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def _push_aggregation(self):
        # the only thing we implement here is function calling. in all other cases, messages
        # are added to the context when we receive openai realtime api events
        if not self._function_call_result:
            return

        self._reset()
        try:
            frame = self._function_call_result
            self._function_call_result = None
            if frame.result:
                self._context.add_message_already_present_in_api_context(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": frame.tool_call_id,
                                "function": {
                                    "name": frame.function_name,
                                    "arguments": json.dumps(frame.arguments),
                                },
                                "type": "function",
                            }
                        ],
                    }
                )
                self._context.add_message(
                    {
                        "role": "tool",
                        "content": json.dumps(frame.result),
                        "tool_call_id": frame.tool_call_id,
                    }
                )
                run_llm = frame.run_llm

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")


class OpenAILLMServiceRealtimeBeta(LLMService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
        session_properties: events.SessionProperties = events.SessionProperties(),
        start_audio_paused: bool = False,
        send_transcription_frames: bool = True,
        send_user_started_speaking_frames: bool = False,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)
        self.api_key = api_key
        self.base_url = base_url

        self._session_properties: events.SessionProperties = session_properties
        self._audio_input_paused = start_audio_paused
        self._send_transcription_frames = send_transcription_frames
        # todo: wire _send_user_started_speaking_frames up correctly
        self._send_user_started_speaking_frames = send_user_started_speaking_frames
        self._websocket = None
        self._receive_task = None
        self._context = None
        self._bot_speaking = False

    def can_generate_metrics(self) -> bool:
        return True

    def set_audio_input_paused(self, paused: bool):
        self._audio_input_paused = paused

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def send_client_event(self, event: events.ClientEvent):
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _ws_send(self, realtime_message):
        try:
            await self._websocket.send(json.dumps(realtime_message))
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

    async def _connect(self):
        try:
            self._websocket = await websockets.connect(
                uri=self.base_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            )
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                await self._websocket.close()
                self._websocket = None

            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _update_settings(self):
        # !!! LEAVE ALL DEFAULT SETTINGS FOR NOW
        return
        settings = self._session_properties
        # tools given in the context override the tools in the session properties
        if self._context and self._context.tools:
            settings.tools = self._context.tools
            self._context.update_tools_list_sent()
        await self.send_client_event(events.SessionUpdateEvent(session=settings))

    async def _receive_task_handler(self):
        try:
            async for message in self._get_websocket():
                evt = events.parse_server_event(message)
                # logger.debug(f"Received event: {evt}")
                if evt.type == "session.created":
                    # session.created is received right after connecting. send a message
                    # to configure the session properties.
                    logger.debug(f"!!! GOT SESSION CREATED {evt}")
                    await self._update_settings()
                elif evt.type == "session.updated":
                    logger.debug(f"!!! GOT SESSION UPDATED {evt}")
                    self._session_properties = evt.session
                elif evt.type == "conversation.created":
                    logger.debug(f"!!! GOT CONVERSATION CREATED: {evt}")
                elif evt.type == "input_audio_buffer.speech_started":
                    # user started speaking
                    if self._send_user_started_speaking_frames:
                        await self.push_frame(UserStartedSpeakingFrame())
                        await self.push_frame(StartInterruptionFrame())
                        logger.debug("User started speaking")
                    pass
                elif evt.type == "input_audio_buffer.speech_stopped":
                    # user stopped speaking
                    if self._send_user_started_speaking_frames:
                        await self.push_frame(UserStoppedSpeakingFrame())
                        await self.push_frame(StopInterruptionFrame())

                        logger.debug("User stopped speaking")
                    await self.start_processing_metrics()
                    await self.start_ttfb_metrics()
                elif evt.type == "conversation.item.created":
                    # this will get sent from the server every time a new "message" is added
                    # to the server's conversation state
                    if self._context:
                        self._context.add_message_from_realtime_event(evt)
                elif evt.type == "response.created":
                    # todo: 1. figure out TTS started/stopped frame semantics better
                    #       2. do not push these frames in text-only mode
                    logger.debug(f"!!! GOT RESPONSE CREATED {evt}")
                    if not self._bot_speaking:
                        self._bot_speaking = True
                        await self.push_frame(TTSStartedFrame())
                    pass
                elif evt.type == "conversation.item.input_audio_transcription.completed":
                    if evt.transcript:
                        if self._context:
                            self._context.add_transcript_to_message(evt)
                        if self._send_transcription_frames:
                            await self.push_frame(
                                # no way to get a language code?
                                TranscriptionFrame(evt.transcript, "", time_now_iso8601())
                            )
                elif evt.type == "response.output_item.added":
                    # todo: think about adding a frame for this (generally, in Pipecat/RTVI), as
                    # it could be useful for managing UI state
                    pass
                elif evt.type == "response.content_part.added":
                    # todo: same thing — possibly a useful event for client-side UI
                    pass
                elif evt.type == "response.audio_transcript.delta":
                    # note: the openai playground app uses this, not "response.text.delta"
                    if evt.delta:
                        await self.push_frame(TextFrame(evt.delta))
                elif evt.type == "response.audio.delta":
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(
                        audio=base64.b64decode(evt.delta),
                        sample_rate=24000,
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif evt.type == "response.audio.done":
                    if self._bot_speaking:
                        self._bot_speaking = False
                        await self.push_frame(TTSStoppedFrame())
                elif evt.type == "response.audio_transcript.done":
                    if self._context:
                        self._context.add_transcript_to_message(evt)
                    pass
                elif evt.type == "response.content_part.done":
                    # this doesn't map to any Pipecat frame types
                    pass
                elif evt.type == "response.output_item.done":
                    # this doesn't map to any Pipecat frame types
                    pass
                elif evt.type == "response.done":
                    # usage metrics
                    tokens = LLMTokenUsage(
                        prompt_tokens=evt.response.usage.input_tokens,
                        completion_tokens=evt.response.usage.output_tokens,
                        total_tokens=evt.response.usage.total_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)
                    await self.stop_processing_metrics()
                    # function calls
                    items = evt.response.output
                    function_calls = [item for item in items if item.type == "function_call"]
                    if function_calls:
                        await self._handle_function_call_items(function_calls)
                    await self.push_frame(LLMFullResponseEndFrame())
                elif evt.type == "rate_limits.updated":
                    # todo: add a Pipecat frame for this. (maybe?)
                    pass
                elif evt.type == "error":
                    # These errors seem to be fatal to this connection. So, close and send an ErrorFrame.
                    raise Exception(f"Error: {evt}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}\n\nStack trace:\n{traceback.format_exc()}")
            await self.push_error(ErrorFrame(error=f"Error receiving: {e}", fatal=True))

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

    async def _reset_conversation(self, count):
        # need to think about how to implement this, and how to think about interop with messages lists
        # used with the HTTP API
        logger.debug(f"!!! RESET CONVERSATION: {count} [WIP]")
        await self._disconnect()
        await self._connect()
        pass

    async def _send_messages_context_update(self):
        if not self._context:
            return
        context = self._context
        messages = context.get_unsent_messages()

        needs_reset = context.get_messages_reset_count()
        context.update_all_messages_sent()

        if needs_reset:
            await self._reset_conversation(needs_reset)
            # debugging
            logger.debug("MESSAGE HISTORY RELOAD NOT IMPLEMENTED YET")
            return

        items = []
        for m in messages:
            if m and (
                m.get("role") == "user" or m.get("role") == "system" or m.get("role") == "assistant"
            ):
                content = m.get("content")
                if isinstance(content, str):
                    # skip any messages that aren't "text" and change "user" message type to "input_text"

                    if m.get("type", "text") == "text":
                        items.append(
                            events.ConversationItem(
                                type="message",
                                status="completed",
                                role=m.get("role", "user"),
                                content=[
                                    events.ItemContent(
                                        type="input_text" if m.get("role") == "user" else "text",
                                        text=content,
                                    )
                                ],
                            )
                        )
                elif isinstance(content, list):
                    # skip any messages that aren't "text" and change "user" message type to "input_text"
                    cs = []
                    for item in content:
                        if item.get("type", "text") == "text":
                            # cs.append(events.ItemContent(type="input_text", text=item.get("text")))
                            (
                                cs.append(
                                    events.ItemContent(
                                        type="input_text" if m.get("role") == "user" else "text",
                                        text=item.get("text"),
                                    )
                                ),
                            )
                    if cs:
                        items.append(
                            events.ConversationItem(
                                type="message",
                                status="completed",
                                role=m.get("role", "user"),
                                content=cs,
                            )
                        )
                elif m.get("role") == "assistant" and m.get("tool_calls"):
                    tc = m.get("tool_calls")[0]
                    items.append(
                        events.ConversationItem(
                            type="function_call",
                            call_id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        )
                    )
                else:
                    raise Exception(f"Invalid message content {m}")
            elif m and m.get("role") == "tool":
                items.append(
                    events.ConversationItem(
                        type="function_call_output",
                        call_id=m.get("tool_call_id"),
                        output=m["content"],
                    )
                )

        for item in items:
            context.note_manually_added_message(item.id)
            evt = events.ConversationItemCreateEvent(item=item)
            logger.debug(
                f"!!! > Sending message: {evt.model_dump_json(indent=2, exclude_none=True)}"
            )
            await self.send_client_event(evt)
            await asyncio.sleep(2)
            # await self.send_client_event(events.ConversationItemCreateEvent(item=item))

    async def _create_response(self):
        if self._context.get_tools_list_updated():
            await self._update_settings()

        # !!! DEBUGGING - testing await on conversation.create
        logger.debug("!!! A waiting on conversation.created")
        await asyncio.sleep(3)
        logger.debug("!!! A ok, done waiting")

        await self._send_messages_context_update()
        logger.debug(f"Creating response: {self._context.get_messages_for_logging()}")
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.send_client_event(
            events.ResponseCreateEvent(
                response=events.ResponseProperties(modalities=["audio", "text"])
            )
        )
        # !!! DEBUGGING
        await asyncio.sleep(2)
        # logger.debug("Unpausing microphone")
        # self.set_audio_input_paused(False)

    async def _send_user_audio(self, frame):
        payload = base64.b64encode(frame.audio).decode("utf-8")
        await self.send_client_event(events.InputAudioBufferAppendEvent(audio=payload))

    async def _handle_interruption(self, frame):
        await self.send_client_event(events.InputAudioBufferClearEvent())
        await self.send_client_event(events.ResponseCancelEvent())
        await self.stop_all_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
        await self.push_frame(TTSStoppedFrame())

    async def _handle_user_started_speaking(self, frame):
        pass

    async def _handle_user_stopped_speaking(self, frame):
        if self._session_properties.turn_detection is None:
            await self.send_client_event(events.InputAudioBufferCommitEvent())
            await self.send_client_event(events.ResponseCreateEvent())
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            pass
        elif isinstance(frame, OpenAILLMContextFrame):
            context: OpenAIRealtimeLLMContext = OpenAIRealtimeLLMContext.upgrade_to_realtime(
                frame.context
            )
            self._context = context
            await self._create_response()
        elif isinstance(frame, InputAudioRawFrame):
            if not self._audio_input_paused:
                await self._send_user_audio(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption(frame)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, _InternalMessagesUpdateFrame):
            self._context = frame.context
            await self._send_messages_context_update()
        elif isinstance(frame, LLMUpdateSettingsFrame):
            self._session_properties = frame.settings
            await self._update_settings()
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()

        await self.push_frame(frame, direction)

    def create_context_aggregator(
        self, context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = False
    ) -> OpenAIContextAggregatorPair:
        OpenAIRealtimeLLMContext.upgrade_to_realtime(context)
        user = OpenAIRealtimeUserContextAggregator(context)
        assistant = OpenAIRealtimeAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

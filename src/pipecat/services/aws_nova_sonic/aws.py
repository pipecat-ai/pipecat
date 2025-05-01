#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInput,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationOutput,
    InvokeModelWithBidirectionalStreamOutput,
)
from loguru import logger
from smithy_aws_core.credentials_resolvers.static import StaticCredentialsResolver
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.aio.eventstream import DuplexEventStream

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.aws_nova_sonic_adapter import AWSNovaSonicLLMAdapter
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
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
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws_nova_sonic.context import (
    AWSNovaSonicAssistantContextAggregator,
    AWSNovaSonicContextAggregatorPair,
    AWSNovaSonicLLMContext,
    AWSNovaSonicUserContextAggregator,
    Role,
)
from pipecat.services.aws_nova_sonic.frames import AWSNovaSonicFunctionCallResultFrame
from pipecat.services.llm_service import LLMService
from pipecat.utils.time import time_now_iso8601


class AWSNovaSonicUnhandledFunctionException(Exception):
    pass


class ContentType(Enum):
    AUDIO = "AUDIO"
    TEXT = "TEXT"
    TOOL = "TOOL"


class TextStage(Enum):
    FINAL = "FINAL"  # what has been said
    SPECULATIVE = "SPECULATIVE"  # what's planned to be said


@dataclass
class CurrentContent:
    type: ContentType
    role: Role
    text_stage: TextStage  # None if not text
    text_content: str  # starts as None, then fills in if text

    def __str__(self):
        return (
            f"CurrentContent(\n"
            f"  type={self.type.name},\n"
            f"  role={self.role.name},\n"
            f"  text_stage={self.text_stage.name if self.text_stage else 'None'}\n"
            f")"
        )


class AWSNovaSonicLLMService(LLMService):
    # Override the default adapter to use the AWSNovaSonicLLMAdapter one
    adapter_class = AWSNovaSonicLLMAdapter

    def __init__(
        self,
        *,
        # TODO: if we have instruction here as an alternative to using context, we should do the same for tools...right?
        secret_access_key: str,
        access_key_id: str,
        region: str,
        model: str = "amazon.nova-sonic-v1:0",
        voice_id: str = "matthew",  # matthew, tiffany, amy
        instruction: Optional[str] = None,
        tools: Optional[ToolsSchema] = None,
        send_transcription_frames: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._secret_access_key = secret_access_key
        self._access_key_id = access_key_id
        self._region = region
        self._model = model
        self._client: BedrockRuntimeClient = None
        self._voice_id = voice_id
        self._instruction = instruction
        self._tools = tools
        self._send_transcription_frames = send_transcription_frames
        self._context: AWSNovaSonicLLMContext = None
        self._stream: DuplexEventStream[
            InvokeModelWithBidirectionalStreamInput,
            InvokeModelWithBidirectionalStreamOutput,
            InvokeModelWithBidirectionalStreamOperationOutput,
        ] = None
        self._receive_task = None
        self._prompt_name = None
        self._input_audio_content_name = None
        self._content_being_received = None
        self._assistant_is_responding = False
        self._context_available = False
        self._ready_to_send_context = False

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # TODO: maybe connect but don't send history until we get all of our settings?
        # how do we know how long to wait?
        # ah, i think we'll *always* get at least one OpenAILLMContextFrame which kicks things off
        # so we need to send the initial history when:
        # - we're connected
        # - we've gotten the first context
        # i *think* this is what's controlled by _api_session_ready/_run_llm_when_api_session_ready
        await self._start_connecting()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    #
    # frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            # TODO: check if _audio_input_paused? what causes that?
            await self._send_user_audio_event(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking()
        elif isinstance(frame, AWSNovaSonicFunctionCallResultFrame):
            await self._handle_function_call_result(frame)
        # TODO: do we need to do anything for the below four frame types?
        elif isinstance(frame, StartInterruptionFrame):
            # print("[pk] StartInterruptionFrame")
            pass
        elif isinstance(frame, UserStartedSpeakingFrame):
            # print("[pk] UserStartedSpeakingFrame")
            pass
        elif isinstance(frame, StopInterruptionFrame):
            # print("[pk] StopInterruptionFrame")
            pass
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # print("[pk] UserStoppedSpeakingFrame")
            pass

        await self.push_frame(frame, direction)

    async def _handle_context(self, context: OpenAILLMContext):
        # TODO: reset connection if needed (if entirely new context object provided, for instance)
        print(f"[pk] receive updated context: {context.get_messages_for_initializing_history()}")
        if not self._context:
            # We got our initial context - try to finish connecting
            self._context = AWSNovaSonicLLMContext.upgrade_to_nova_sonic(context)
            self._context_available = True
            await self._finish_connecting_if_context_available()

    async def _handle_bot_stopped_speaking(self):
        if self._assistant_is_responding:
            # Consider the assistant finished with their response (after a short delay, to allow for
            # any FINAL text block to come in).
            #
            # TODO: ideally we could base this solely on the LLM output events, but I couldn't
            # figure out a reliable way to determine when we've gotten our last FINAL text block
            # after the LLM is done talking.
            #
            # First I looked at stopReason, but it doesn't seem like the last FINAL text block is
            # reliably marked END_TURN (sometimes the *first* one is, but not the last...bug?)
            #
            # Then I considered schemes where we tally or match up SPECULATIVE text blocks with
            # FINAL text blocks to know how many or which FINAL blocks to expect, but user
            # interruptions throw a wrench in these schemes: depending on the exact timing of the
            # interruption, we should or shouldn't expect some FINAL blocks.
            await asyncio.sleep(0.25)
            self._assistant_is_responding = False
            await self._report_assistant_response_ended()

    async def _handle_function_call_result(self, frame: AWSNovaSonicFunctionCallResultFrame):
        result = frame.result_frame
        await self._send_tool_result(tool_call_id=result.tool_call_id, result=result.result)

    #
    # LLM communication: lifecycle
    #

    async def _start_connecting(self):
        try:
            if self._client:
                # Here we assume that if we have a client we are connected or connecting
                return

            # Set IDs for the connection
            self._prompt_name = str(uuid.uuid4())
            self._input_audio_content_name = str(uuid.uuid4())

            # Create the client
            self._client = self._create_client()

            # Start the bidirectional stream
            self._stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model)
            )

            # Send session start event
            await self._send_session_start_event()

            # Finish connecting
            self._ready_to_send_context = True
            await self._finish_connecting_if_context_available()
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._disconnect()

    async def _finish_connecting_if_context_available(self):
        # We can only finish connecting once we've gotten our initial context and we're ready to
        # send it
        if not (self._context_available and self._ready_to_send_context):
            return

        # Read context
        history = self._context.get_messages_for_initializing_history()

        # Send prompt start event, specifying tools
        # Tools from context take priority over tools from __init__()
        tools = (
            self._context.tools
            if self._context.tools
            else self.get_llm_adapter().from_standard_tools(self._tools)
        )
        await self._send_prompt_start_event(tools)

        # Send system instruction
        # Instruction from context takes priority over instruction from __init__()
        instruction = history.instruction if history.instruction else self._instruction
        if instruction:
            await self._send_text_event(text=instruction, role=Role.SYSTEM)

        # Send conversation history
        for message in history.messages:
            await self._send_text_event(text=message.text, role=message.role)

        # Send initial context (system instruction and conversation history)
        # TODO: finish implementing
        # - pass additional message(s)
        # - merge init-passed system instruction + context instruction (latter takes precedence)
        # - merge init-passed tools + context tools (latter takes precedence)
        await self._send_text_event(text=self._instruction, role=Role.SYSTEM)

        # Start audio input
        await self._send_audio_input_start_event()

        # Start receiving events
        self._receive_task = self.create_task(self._receive_task_handler())

    async def _disconnect(self):
        try:
            # Clean up receive task
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None

            # Clean up client
            if self._client:
                await self._send_session_end_events()
                self._client = None

            # Clean up stream
            if self._stream:
                await self._stream.input_stream.close()
                self._stream = None

            # Reset remaining connection-specific state
            self._prompt_name = None
            self._input_audio_content_name = None
            self._content_being_received = None
            self._assistant_is_responding = False
            self._context_available = False
            self._ready_to_send_context = False
        except Exception as e:
            logger.error(f"{self} error disconnecting: {e}")

    def _create_client(self) -> BedrockRuntimeClient:
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._region}.amazonaws.com",
            region=self._region,
            aws_credentials_identity_resolver=StaticCredentialsResolver(
                credentials=AWSCredentialsIdentity(
                    access_key_id=self._access_key_id,
                    secret_access_key=self._secret_access_key,
                    # TODO: add additional stuff like aws_session_token
                )
            ),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        return BedrockRuntimeClient(config=config)

    #
    # LLM communication: input events (pipecat -> LLM)
    #

    # TODO: make params configurable?
    async def _send_session_start_event(self):
        session_start = """
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        """
        await self._send_client_event(session_start)

    async def _send_prompt_start_event(self, tools: List[Any]):
        tools_config = (
            f""",
        "toolUseOutputConfiguration": {{
          "mediaType": "application/json"
        }},
        "toolConfiguration": {{
          "tools": {json.dumps(tools)}
        }}
        """
            if tools
            else ""
        )

        prompt_start = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self._prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "{self._voice_id}",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}{tools_config}
            }}
          }}
        }}
        '''
        await self._send_client_event(prompt_start)

    async def _send_audio_input_start_event(self):
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{self._input_audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(audio_content_start)

    async def _send_text_event(self, text: str, role: Role):
        if not self._stream:
            return

        content_name = str(uuid.uuid4())

        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}",
                    "type": "TEXT",
                    "interactive": true,
                    "role": "{role.value}",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(text_content_start)

        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}",
                    "content": "{text}"
                }}
            }}
        }}
        '''
        await self._send_client_event(text_input)

        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}"
                }}
            }}
        }}
        '''
        await self._send_client_event(text_content_end)

    async def _send_user_audio_event(self, frame: InputAudioRawFrame):
        if not self._stream:
            return

        blob = base64.b64encode(frame.audio)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{self._input_audio_content_name}",
                    "content": "{blob.decode("utf-8")}"
                }}
            }}
        }}
        '''
        await self._send_client_event(audio_event)

    async def _send_session_end_events(self):
        if not self._stream:
            return

        prompt_end = f'''
        {{
            "event": {{
                "promptEnd": {{
                    "promptName": "{self._prompt_name}"
                }}
            }}
        }}
        '''
        await self._send_client_event(prompt_end)

        session_end = """
        {
            "event": {
                "sessionEnd": {}
            }
        }
        """
        await self._send_client_event(session_end)

    async def _send_tool_result(self, tool_call_id, result):
        if not self._stream:
            return

        # print(f"[pk] sending tool result. tool call ID: {tool_call_id}, result: {result}")

        content_name = str(uuid.uuid4())

        result_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}",
                    "interactive": false,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {{
                        "toolUseId": "{tool_call_id}",
                        "type": "TEXT",
                        "textInputConfiguration": {{
                            "mediaType": "text/plain"
                        }}
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(result_content_start)

        result_content = json.dumps(
            {
                "event": {
                    "toolResult": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "content": json.dumps(result) if isinstance(result, dict) else result,
                    }
                }
            }
        )
        await self._send_client_event(result_content)

        result_content_end = f"""
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}"
                }}
            }}
        }}
        """
        await self._send_client_event(result_content_end)

    async def _send_client_event(self, event_json: str):
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self._stream.input_stream.send(event)

    #
    # LLM communication: output events (LLM -> pipecat)
    #

    # Receive the ongoing LLM "completion".
    # There is generally a single completion per session.
    # In a completion, a few different kinds of content can be delivered:
    # - Transcription of user audio
    # - Tool use
    # - Text preview of planned response speech before audio delivered
    # - User interruption notification
    # - Text of response speech that whose audio was actually delivered
    # - Audio of response speech
    # Each piece of content is wrapped by "contentStart" and "contentEnd" events. The content is
    # delivered sequentially: one piece of content will end before another starts.
    # The overall completion is wrapped by "completionStart" and "completionEnd" events.
    async def _receive_task_handler(self):
        try:
            while self._client:
                output = await self._stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode("utf-8")
                    json_data = json.loads(response_data)

                if "event" in json_data:
                    event_json = json_data["event"]
                    if "completionStart" in event_json:
                        # Handle the LLM completion starting
                        await self._handle_completion_start_event(event_json)
                    elif "contentStart" in event_json:
                        # Handle a piece of content starting
                        await self._handle_content_start_event(event_json)
                    elif "textOutput" in event_json:
                        # Handle text output content
                        await self._handle_text_output_event(event_json)
                    elif "audioOutput" in event_json:
                        # Handle audio output content
                        await self._handle_audio_output_event(event_json)
                    elif "toolUse" in event_json:
                        # Handle tool use
                        await self._handle_tool_use_event(event_json)
                    elif "contentEnd" in event_json:
                        # Handle a piece of content ending
                        await self._handle_content_end_event(event_json)
                    elif "completionStart" in event_json:
                        # Handle the LLM completion ending
                        await self._handle_completion_end_event(event_json)

        except Exception as e:
            logger.error(f"{self} error processing responses: {e}")

    async def _handle_completion_start_event(self, event_json):
        # print("[pk] completion start")
        pass

    async def _handle_content_start_event(self, event_json):
        content_start = event_json["contentStart"]
        type = content_start["type"]
        role = content_start["role"]
        generation_stage = None
        if "additionalModelFields" in content_start:
            additional_model_fields = json.loads(content_start["additionalModelFields"])
            generation_stage = additional_model_fields.get("generationStage")

        # Bookkeeping: track current content being received
        content = CurrentContent(
            type=ContentType(type),
            role=Role(role),
            text_stage=TextStage(generation_stage) if generation_stage else None,
            text_content=None,
        )
        self._content_being_received = content

        # print(f"[pk] content start: {content}")
        # if content.role == Role.ASSISTANT:
        #     print(f"[pk] assistant content start: {content}")

        if content.role == Role.ASSISTANT:
            if content.type == ContentType.AUDIO:
                # Note that an assistant response can comprise of multiple audio blocks
                if not self._assistant_is_responding:
                    # The assistant has started responding.
                    self._assistant_is_responding = True
                    await self._report_assistant_response_started()

    async def _handle_text_output_event(self, event_json):
        # This should never happen
        if not self._content_being_received:
            return
        content = self._content_being_received

        text_content = event_json["textOutput"]["content"]
        # print(f"[pk] text output. content: {text_content}")
        # if content.role == Role.ASSISTANT:
        #     print(f"[pk] assistant text output. content: {text_content}")

        # Bookkeeping: augment the current content being received with text
        # Assumption: only one text content per content block
        content.text_content = text_content

    async def _handle_audio_output_event(self, event_json):
        # This should never happen
        if not self._content_being_received:
            return

        # Get audio
        audio_content = event_json["audioOutput"]["content"]
        # print(f"[pk] audio output. content: {len(audio_content)}")

        # Push audio frame
        audio = base64.b64decode(audio_content)
        # TODO: make sample rate + channels (used in multiple places) consts
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=24000,
            num_channels=1,
        )
        await self.push_frame(frame)

    async def _handle_tool_use_event(self, event_json):
        # This should never happen
        if not self._content_being_received:
            return

        # Get tool use details
        tool_use = event_json["toolUse"]
        function_name = tool_use["toolName"]
        tool_call_id = tool_use["toolUseId"]
        arguments = json.loads(tool_use["content"])

        # print(
        #     f"[pk] tool use - function_name: {function_name}, tool_call_id: {tool_call_id}, arguments: {arguments}"
        # )

        # Call tool function
        if self.has_function(function_name):
            if function_name in self._functions.keys():
                await self.call_function(
                    context=self._context,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    arguments=arguments,
                )
            elif None in self._functions.keys():
                await self.call_function(
                    context=self._context,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    arguments=arguments,
                )
        else:
            raise AWSNovaSonicUnhandledFunctionException(
                f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
            )

    async def _handle_content_end_event(self, event_json):
        # This should never happen
        if not self._content_being_received:
            return
        content = self._content_being_received

        content_end = event_json["contentEnd"]
        stop_reason = content_end["stopReason"]
        # print(f"[pk] content end: {content}.\n  stop_reason: {stop_reason}")
        # if content.role == Role.ASSISTANT:
        #     print(f"[pk] assistant content end: {content}.\n  stop_reason: {stop_reason}")

        # Bookkeeping: clear current content being received
        self._content_being_received = None

        if content.role == Role.ASSISTANT:
            if content.type == ContentType.TEXT:
                # Ignore non-final text, and the "interrupted" message (which isn't meaningful text)
                if content.text_stage == TextStage.FINAL and stop_reason != "INTERRUPTED":
                    if self._assistant_is_responding:
                        # Text added to the ongoing assistant response
                        await self._report_assistant_response_text_added(content.text_content)
        elif content.role == Role.USER:
            if content.type == ContentType.TEXT:
                if content.text_stage == TextStage.FINAL:
                    # User transcription text added
                    await self._report_user_transcription_text_added(content.text_content)

        self._content_being_received = False

    async def _handle_completion_end_event(self, event_json):
        # print("[pk] completion end")
        pass

    async def _report_assistant_response_started(self):
        # Report that the assistant has started their response.
        print("[pk] LLM full response started")
        await self.push_frame(LLMFullResponseStartFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) started
        print("[pk] TTS started")
        await self.push_frame(TTSStartedFrame())

    async def _report_assistant_response_text_added(self, text):
        # Report some text added to the ongoing assistant response
        print(f"[pk] LLM text: {text}")
        await self.push_frame(LLMTextFrame(text))

        # Report some text added to the *equivalent* of TTS (this is a speech-to-speech model)
        print(f"[pk] TTS text: {text}")
        await self.push_frame(TTSTextFrame(text))

    async def _report_assistant_response_ended(self):
        # Report that the assistant has finished their response.
        print("[pk] LLM full response ended")
        await self.push_frame(LLMFullResponseEndFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) stopped.
        print("[pk] TTS stopped")
        await self.push_frame(TTSStoppedFrame())

    async def _report_user_transcription_text_added(self, text):
        print(f"[pk] transcription: {text}")
        # Manually add new user transcription text to context.
        # We can't rely on the user context aggregator to do this since it's upstream from the LLM.
        self._context.add_user_transcription_text_as_message(text)

        # Report that some new user transcription text is available.
        if self._send_transcription_frames:
            await self.push_frame(
                TranscriptionFrame(text=text, user_id="", timestamp=time_now_iso8601())
            )

    #
    # Context
    #

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> AWSNovaSonicContextAggregatorPair:
        context.set_llm_adapter(self.get_llm_adapter())

        user = AWSNovaSonicUserContextAggregator(context=context, params=user_params)
        assistant = AWSNovaSonicAssistantContextAggregator(context=context, params=assistant_params)

        return AWSNovaSonicContextAggregatorPair(user, assistant)

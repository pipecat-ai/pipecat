#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic LLM service implementation for Pipecat AI framework.

This module provides a speech-to-speech LLM service using AWS Nova Sonic, which supports
bidirectional audio streaming, text generation, and function calling capabilities.
"""

import asyncio
import base64
import json
import time
import uuid
import wave
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.aws_nova_sonic_adapter import AWSNovaSonicLLMAdapter
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
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

try:
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
    from smithy_aws_core.credentials_resolvers.static import StaticCredentialsResolver
    from smithy_aws_core.identity import AWSCredentialsIdentity
    from smithy_core.aio.eventstream import DuplexEventStream
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AWS services, you need to `pip install pipecat-ai[aws-nova-sonic]`."
    )
    raise Exception(f"Missing module: {e}")


class AWSNovaSonicUnhandledFunctionException(Exception):
    """Exception raised when the LLM attempts to call an unregistered function."""

    pass


class ContentType(Enum):
    """Content types supported by AWS Nova Sonic.

    Parameters:
        AUDIO: Audio content type.
        TEXT: Text content type.
        TOOL: Tool content type.
    """

    AUDIO = "AUDIO"
    TEXT = "TEXT"
    TOOL = "TOOL"


class TextStage(Enum):
    """Text generation stages in AWS Nova Sonic responses.

    Parameters:
        FINAL: Final text that has been fully generated.
        SPECULATIVE: Speculative text that is still being generated.
    """

    FINAL = "FINAL"  # what has been said
    SPECULATIVE = "SPECULATIVE"  # what's planned to be said


@dataclass
class CurrentContent:
    """Represents content currently being received from AWS Nova Sonic.

    Parameters:
        type: The type of content (audio, text, or tool).
        role: The role generating the content (user, assistant, etc.).
        text_stage: The stage of text generation (final or speculative).
        text_content: The actual text content if applicable.
    """

    type: ContentType
    role: Role
    text_stage: TextStage  # None if not text
    text_content: str  # starts as None, then fills in if text

    def __str__(self):
        """String representation of the current content."""
        return (
            f"CurrentContent(\n"
            f"  type={self.type.name},\n"
            f"  role={self.role.name},\n"
            f"  text_stage={self.text_stage.name if self.text_stage else 'None'}\n"
            f")"
        )


class Params(BaseModel):
    """Configuration parameters for AWS Nova Sonic.

    Parameters:
        input_sample_rate: Audio input sample rate in Hz.
        input_sample_size: Audio input sample size in bits.
        input_channel_count: Number of input audio channels.
        output_sample_rate: Audio output sample rate in Hz.
        output_sample_size: Audio output sample size in bits.
        output_channel_count: Number of output audio channels.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling parameter.
        temperature: Sampling temperature for text generation.
    """

    # Audio input
    input_sample_rate: Optional[int] = Field(default=16000)
    input_sample_size: Optional[int] = Field(default=16)
    input_channel_count: Optional[int] = Field(default=1)

    # Audio output
    output_sample_rate: Optional[int] = Field(default=24000)
    output_sample_size: Optional[int] = Field(default=16)
    output_channel_count: Optional[int] = Field(default=1)

    # Inference
    max_tokens: Optional[int] = Field(default=1024)
    top_p: Optional[float] = Field(default=0.9)
    temperature: Optional[float] = Field(default=0.7)


class AWSNovaSonicLLMService(LLMService):
    """AWS Nova Sonic speech-to-speech LLM service.

    Provides bidirectional audio streaming, real-time transcription, text generation,
    and function calling capabilities using AWS Nova Sonic model.
    """

    # Override the default adapter to use the AWSNovaSonicLLMAdapter one
    adapter_class = AWSNovaSonicLLMAdapter

    def __init__(
        self,
        *,
        secret_access_key: str,
        access_key_id: str,
        session_token: Optional[str] = None,
        region: str,
        model: str = "amazon.nova-sonic-v1:0",
        voice_id: str = "matthew",  # matthew, tiffany, amy
        params: Optional[Params] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[ToolsSchema] = None,
        send_transcription_frames: bool = True,
        **kwargs,
    ):
        """Initializes the AWS Nova Sonic LLM service.

        Args:
            secret_access_key: AWS secret access key for authentication.
            access_key_id: AWS access key ID for authentication.
            session_token: AWS session token for authentication.
            region: AWS region where the service is hosted.
            model: Model identifier. Defaults to "amazon.nova-sonic-v1:0".
            voice_id: Voice ID for speech synthesis. Options: matthew, tiffany, amy.
            params: Model parameters for audio configuration and inference.
            system_instruction: System-level instruction for the model.
            tools: Available tools/functions for the model to use.
            send_transcription_frames: Whether to emit transcription frames.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        super().__init__(**kwargs)
        self._secret_access_key = secret_access_key
        self._access_key_id = access_key_id
        self._session_token = session_token
        self._region = region
        self._model = model
        self._client: Optional[BedrockRuntimeClient] = None
        self._voice_id = voice_id
        self._params = params or Params()
        self._system_instruction = system_instruction
        self._tools = tools
        self._send_transcription_frames = send_transcription_frames
        self._context: Optional[AWSNovaSonicLLMContext] = None
        self._stream: Optional[
            DuplexEventStream[
                InvokeModelWithBidirectionalStreamInput,
                InvokeModelWithBidirectionalStreamOutput,
                InvokeModelWithBidirectionalStreamOperationOutput,
            ]
        ] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._prompt_name: Optional[str] = None
        self._input_audio_content_name: Optional[str] = None
        self._content_being_received: Optional[CurrentContent] = None
        self._assistant_is_responding = False
        self._ready_to_send_context = False
        self._handling_bot_stopped_speaking = False
        self._triggering_assistant_response = False
        self._assistant_response_trigger_audio: Optional[bytes] = (
            None  # Not cleared on _disconnect()
        )
        self._disconnecting = False
        self._connected_time: Optional[float] = None
        self._wants_connection = False

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        """Start the service and initiate connection to AWS Nova Sonic.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)
        self._wants_connection = True
        await self._start_connecting()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connections.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        self._wants_connection = False
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close connections.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        self._wants_connection = False
        await self._disconnect()

    #
    # conversation resetting
    #

    async def reset_conversation(self):
        """Reset the conversation state while preserving context.

        Handles bot stopped speaking event, disconnects from the service,
        and reconnects with the preserved context.
        """
        logger.debug("Resetting conversation")
        await self._handle_bot_stopped_speaking(delay_to_catch_trailing_assistant_text=False)

        # Carry over previous context through disconnect
        context = self._context
        await self._disconnect()
        self._context = context

        await self._start_connecting()

    #
    # frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle service-specific logic.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio_frame(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(delay_to_catch_trailing_assistant_text=True)
        elif isinstance(frame, AWSNovaSonicFunctionCallResultFrame):
            await self._handle_function_call_result(frame)

        await self.push_frame(frame, direction)

    async def _handle_context(self, context: OpenAILLMContext):
        if not self._context:
            # We got our initial context - try to finish connecting
            self._context = AWSNovaSonicLLMContext.upgrade_to_nova_sonic(
                context, self._system_instruction
            )
            await self._finish_connecting_if_context_available()

    async def _handle_input_audio_frame(self, frame: InputAudioRawFrame):
        # Wait until we're done sending the assistant response trigger audio before sending audio
        # from the user's mic
        if self._triggering_assistant_response:
            return

        await self._send_user_audio_event(frame.audio)

    async def _handle_bot_stopped_speaking(self, delay_to_catch_trailing_assistant_text: bool):
        # Protect against back-to-back BotStoppedSpeaking calls, which I've observed
        if self._handling_bot_stopped_speaking:
            return
        self._handling_bot_stopped_speaking = True

        async def finalize_assistant_response():
            if self._assistant_is_responding:
                # Consider the assistant finished with their response (possibly after a short delay,
                # to allow for any trailing FINAL assistant text block to come in that need to make
                # it into context).
                #
                # TODO: ideally we could base this solely on the LLM output events, but I couldn't
                # figure out a reliable way to determine when we've gotten our last FINAL text block
                # after the LLM is done talking.
                #
                # First I looked at stopReason, but it doesn't seem like the last FINAL text block
                # is reliably marked END_TURN (sometimes the *first* one is, but not the last...
                # bug?)
                #
                # Then I considered schemes where we tally or match up SPECULATIVE text blocks with
                # FINAL text blocks to know how many or which FINAL blocks to expect, but user
                # interruptions throw a wrench in these schemes: depending on the exact timing of
                # the interruption, we should or shouldn't expect some FINAL blocks.
                if delay_to_catch_trailing_assistant_text:
                    # This delay length is a balancing act between "catching" trailing assistant
                    # text that is quite delayed but not waiting so long that user text comes in
                    # first and results in a bit of context message order scrambling.
                    await asyncio.sleep(1.25)
                self._assistant_is_responding = False
                await self._report_assistant_response_ended()

            self._handling_bot_stopped_speaking = False

        # Finalize the assistant response, either now or after a delay
        if delay_to_catch_trailing_assistant_text:
            self.create_task(finalize_assistant_response())
        else:
            await finalize_assistant_response()

    async def _handle_function_call_result(self, frame: AWSNovaSonicFunctionCallResultFrame):
        result = frame.result_frame
        await self._send_tool_result(tool_call_id=result.tool_call_id, result=result.result)

    #
    # LLM communication: lifecycle
    #

    async def _start_connecting(self):
        try:
            logger.info("Connecting...")

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
        if not (self._context and self._ready_to_send_context):
            return

        logger.info("Finishing connecting (setting up session)...")

        # Read context
        history = self._context.get_messages_for_initializing_history()

        # Send prompt start event, specifying tools.
        # Tools from context take priority over self._tools.
        tools = (
            self._context.tools
            if self._context.tools
            else self.get_llm_adapter().from_standard_tools(self._tools)
        )
        logger.debug(f"Using tools: {tools}")
        await self._send_prompt_start_event(tools)

        # Send system instruction.
        # Instruction from context takes priority over self._system_instruction.
        # (NOTE: this prioritizing occurred automatically behind the scenes: the context was
        # initialized with self._system_instruction and then updated itself from its messages when
        # get_messages_for_initializing_history() was called).
        logger.debug(f"Using system instruction: {history.system_instruction}")
        if history.system_instruction:
            await self._send_text_event(text=history.system_instruction, role=Role.SYSTEM)

        # Send conversation history
        for message in history.messages:
            await self._send_text_event(text=message.text, role=message.role)

        # Start audio input
        await self._send_audio_input_start_event()

        # Start receiving events
        self._receive_task = self.create_task(self._receive_task_handler())

        # Record finished connecting time (must be done before sending assistant response trigger)
        self._connected_time = time.time()

        logger.info("Finished connecting")

        # If we need to, send assistant response trigger (depends on self._connected_time)
        if self._triggering_assistant_response:
            await self._send_assistant_response_trigger()

    async def _disconnect(self):
        try:
            logger.info("Disconnecting...")

            # NOTE: see explanation of HACK, below
            self._disconnecting = True

            # Clean up client
            if self._client:
                await self._send_session_end_events()
                self._client = None

            # Clean up stream
            if self._stream:
                await self._stream.input_stream.close()
                self._stream = None

            # NOTE: see explanation of HACK, below
            await asyncio.sleep(1)

            # Clean up receive task
            # HACK: we should ideally be able to cancel the receive task before stopping the input
            # stream, above (meaning we wouldn't need self._disconnecting). But for some reason if
            # we don't close the input stream and wait a second first, we're getting an error a lot
            # like this one: https://github.com/awslabs/amazon-transcribe-streaming-sdk/issues/61.
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None

            # Reset remaining connection-specific state
            self._prompt_name = None
            self._input_audio_content_name = None
            self._content_being_received = None
            self._assistant_is_responding = False
            self._ready_to_send_context = False
            self._handling_bot_stopped_speaking = False
            self._triggering_assistant_response = False
            self._disconnecting = False
            self._connected_time = None

            logger.info("Finished disconnecting")
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
                    session_token=self._session_token,
                )
            ),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        return BedrockRuntimeClient(config=config)

    #
    # LLM communication: input events (pipecat -> LLM)
    #

    async def _send_session_start_event(self):
        session_start = f"""
        {{
          "event": {{
            "sessionStart": {{
              "inferenceConfiguration": {{
                "maxTokens": {self._params.max_tokens},
                "topP": {self._params.top_p},
                "temperature": {self._params.temperature}
              }}
            }}
          }}
        }}
        """
        await self._send_client_event(session_start)

    async def _send_prompt_start_event(self, tools: List[Any]):
        if not self._prompt_name:
            return

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
                "sampleRateHertz": {self._params.output_sample_rate},
                "sampleSizeBits": {self._params.output_sample_size},
                "channelCount": {self._params.output_channel_count},
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
        if not self._prompt_name:
            return

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
                        "sampleRateHertz": {self._params.input_sample_rate},
                        "sampleSizeBits": {self._params.input_sample_size},
                        "channelCount": {self._params.input_channel_count},
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(audio_content_start)

    async def _send_text_event(self, text: str, role: Role):
        if not self._stream or not self._prompt_name or not text:
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

        escaped_text = json.dumps(text)  # includes quotes
        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}",
                    "content": {escaped_text}
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

    async def _send_user_audio_event(self, audio: bytes):
        if not self._stream:
            return

        blob = base64.b64encode(audio)
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
        if not self._stream or not self._prompt_name:
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
        if not self._stream or not self._prompt_name:
            return

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
        if not self._stream:  # should never happen
            return

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self._stream.input_stream.send(event)

    #
    # LLM communication: output events (LLM -> pipecat)
    #

    # Receive events for the session.
    # A few different kinds of content can be delivered:
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
            while self._stream and not self._disconnecting:
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
                        elif "completionEnd" in event_json:
                            # Handle the LLM completion ending
                            await self._handle_completion_end_event(event_json)
        except Exception as e:
            logger.error(f"{self} error processing responses: {e}")
            if self._wants_connection:
                await self.reset_conversation()

    async def _handle_completion_start_event(self, event_json):
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

        if content.role == Role.ASSISTANT:
            if content.type == ContentType.AUDIO:
                # Note that an assistant response can comprise of multiple audio blocks
                if not self._assistant_is_responding:
                    # The assistant has started responding.
                    self._assistant_is_responding = True
                    await self._report_user_transcription_ended()  # Consider user turn over
                    await self._report_assistant_response_started()

    async def _handle_text_output_event(self, event_json):
        if not self._content_being_received:  # should never happen
            return
        content = self._content_being_received

        text_content = event_json["textOutput"]["content"]

        # Bookkeeping: augment the current content being received with text
        # Assumption: only one text content per content block
        content.text_content = text_content

    async def _handle_audio_output_event(self, event_json):
        if not self._content_being_received:  # should never happen
            return

        # Get audio
        audio_content = event_json["audioOutput"]["content"]

        # Push audio frame
        audio = base64.b64decode(audio_content)
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=self._params.output_sample_rate,
            num_channels=self._params.output_channel_count,
        )
        await self.push_frame(frame)

    async def _handle_tool_use_event(self, event_json):
        if not self._content_being_received or not self._context:  # should never happen
            return

        # Consider user turn over
        await self._report_user_transcription_ended()

        # Get tool use details
        tool_use = event_json["toolUse"]
        function_name = tool_use["toolName"]
        tool_call_id = tool_use["toolUseId"]
        arguments = json.loads(tool_use["content"])

        # Call tool function
        if self.has_function(function_name):
            if function_name in self._functions.keys() or None in self._functions.keys():
                function_calls_llm = [
                    FunctionCallFromLLM(
                        context=self._context,
                        tool_call_id=tool_call_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                ]

                await self.run_function_calls(function_calls_llm)
        else:
            raise AWSNovaSonicUnhandledFunctionException(
                f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
            )

    async def _handle_content_end_event(self, event_json):
        if not self._content_being_received:  # should never happen
            return
        content = self._content_being_received

        content_end = event_json["contentEnd"]
        stop_reason = content_end["stopReason"]

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

    async def _handle_completion_end_event(self, event_json):
        pass

    #
    # assistant response reporting
    #
    # 1. Started
    # 2. Text added
    # 3. Ended
    #

    async def _report_assistant_response_started(self):
        logger.debug("Assistant response started")

        # Report that the assistant has started their response.
        await self.push_frame(LLMFullResponseStartFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) started
        await self.push_frame(TTSStartedFrame())

    async def _report_assistant_response_text_added(self, text):
        if not self._context:  # should never happen
            return

        logger.debug(f"Assistant response text added: {text}")

        # Report some text added to the ongoing assistant response
        await self.push_frame(LLMTextFrame(text))

        # Report some text added to the *equivalent* of TTS (this is a speech-to-speech model)
        await self.push_frame(TTSTextFrame(text))

        # TODO: this is a (hopefully temporary) HACK. Here we directly manipulate the context rather
        # than relying on the frames pushed to the assistant context aggregator. The pattern of
        # receiving full-sentence text after the assistant has spoken does not easily fit with the
        # Pipecat expectation of chunks of text streaming in while the assistant is speaking.
        # Interruption handling was especially challenging. Rather than spend days trying to fit a
        # square peg in a round hole, I decided on this hack for the time being. We can most cleanly
        # abandon this hack if/when AWS Nova Sonic implements streaming smaller text chunks
        # interspersed with audio. Note that when we move away from this hack, we need to make sure
        # that on an interruption we avoid sending LLMFullResponseEndFrame, which gets the
        # LLMAssistantContextAggregator into a bad state.
        self._context.buffer_assistant_text(text)

    async def _report_assistant_response_ended(self):
        if not self._context:  # should never happen
            return

        logger.debug("Assistant response ended")

        # Report that the assistant has finished their response.
        await self.push_frame(LLMFullResponseEndFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) stopped.
        await self.push_frame(TTSStoppedFrame())

        # For an explanation of this hack, see _report_assistant_response_text_added.
        self._context.flush_aggregated_assistant_text()

    #
    # user transcription reporting
    #
    # 1. Text added
    # 2. Ended
    #
    # Note: "started" does not need to be reported
    #

    async def _report_user_transcription_text_added(self, text):
        if not self._context:  # should never happen
            return

        logger.debug(f"User transcription text added: {text}")

        # Manually add new user transcription text to context.
        # We can't rely on the user context aggregator to do this since it's upstream from the LLM.
        self._context.buffer_user_text(text)

        # Report that some new user transcription text is available.
        if self._send_transcription_frames:
            await self.push_frame(
                InterimTranscriptionFrame(text=text, user_id="", timestamp=time_now_iso8601())
            )

    async def _report_user_transcription_ended(self):
        if not self._context:  # should never happen
            return

        # Manually add user transcription to context (if any has been buffered).
        # We can't rely on the user context aggregator to do this since it's upstream from the LLM.
        transcription = self._context.flush_aggregated_user_text()

        if not transcription:
            return

        logger.debug(f"User transcription ended")

        if self._send_transcription_frames:
            await self.push_frame(
                TranscriptionFrame(text=transcription, user_id="", timestamp=time_now_iso8601())
            )

    #
    # context
    #

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> AWSNovaSonicContextAggregatorPair:
        """Create context aggregator pair for managing conversation context.

        Args:
            context: The OpenAI LLM context to upgrade.
            user_params: Parameters for the user context aggregator.
            assistant_params: Parameters for the assistant context aggregator.

        Returns:
            A pair of user and assistant context aggregators.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        user = AWSNovaSonicUserContextAggregator(context=context, params=user_params)
        assistant = AWSNovaSonicAssistantContextAggregator(context=context, params=assistant_params)

        return AWSNovaSonicContextAggregatorPair(user, assistant)

    #
    # assistant response trigger (HACK)
    #

    # Class variable
    AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION = (
        "Start speaking when you hear the user say 'ready', but don't consider that 'ready' to be "
        "a meaningful part of the conversation other than as a trigger for you to start speaking."
    )

    async def trigger_assistant_response(self):
        """Trigger an assistant response by sending audio cue.

        Sends a pre-recorded "ready" audio trigger to prompt the assistant
        to start speaking. This is useful for controlling conversation flow.

        Returns:
            False if already triggering a response, True otherwise.
        """
        if self._triggering_assistant_response:
            return False

        self._triggering_assistant_response = True

        # Read audio bytes, if we don't already have them cached
        if not self._assistant_response_trigger_audio:
            file_path = files("pipecat.services.aws_nova_sonic").joinpath("ready.wav")
            with wave.open(file_path.open("rb"), "rb") as wav_file:
                self._assistant_response_trigger_audio = wav_file.readframes(wav_file.getnframes())

        # Send the trigger audio, if we're fully connected and set up
        if self._connected_time is not None:
            await self._send_assistant_response_trigger()

    async def _send_assistant_response_trigger(self):
        if (
            not self._assistant_response_trigger_audio or self._connected_time is None
        ):  # should never happen
            return

        try:
            logger.debug("Sending assistant response trigger...")

            chunk_duration = 0.02  # what we might get from InputAudioRawFrame
            chunk_size = int(
                chunk_duration
                * self._params.input_sample_rate
                * self._params.input_channel_count
                * (self._params.input_sample_size / 8)
            )  # e.g. 0.02 seconds of 16-bit (2-byte) PCM mono audio at 16kHz is 640 bytes

            # Lead with a bit of blank audio, if needed.
            # It seems like the LLM can't quite "hear" the first little bit of audio sent on a
            # connection.
            current_time = time.time()
            max_blank_audio_duration = 0.5
            blank_audio_duration = (
                max_blank_audio_duration - (current_time - self._connected_time)
                if self._connected_time is not None
                and (current_time - self._connected_time) < max_blank_audio_duration
                else None
            )
            if blank_audio_duration:
                logger.debug(
                    f"Leading assistant response trigger with {blank_audio_duration}s of blank audio"
                )
                blank_audio_chunk = b"\x00" * chunk_size
                num_chunks = int(blank_audio_duration / chunk_duration)
                for _ in range(num_chunks):
                    await self._send_user_audio_event(blank_audio_chunk)
                    await asyncio.sleep(chunk_duration)

            # Send trigger audio
            # NOTE: this audio *will* be transcribed and eventually make it into the context. That's OK:
            # if we ever need to seed this service again with context it would make sense to include it
            # since the instruction (i.e. the "wait for the trigger" instruction) will be part of the
            # context as well.
            audio_chunks = [
                self._assistant_response_trigger_audio[i : i + chunk_size]
                for i in range(0, len(self._assistant_response_trigger_audio), chunk_size)
            ]
            for chunk in audio_chunks:
                await self._send_user_audio_event(chunk)
                await asyncio.sleep(chunk_duration)
        finally:
            # We need to clean up in case sending the trigger was cancelled, e.g. in the case of a user interruption.
            # (An asyncio.CancelledError would be raised in that case.)
            self._triggering_assistant_response = False

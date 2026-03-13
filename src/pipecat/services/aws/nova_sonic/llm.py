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
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import Any, Deque, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.aws_nova_sonic_adapter import AWSNovaSonicLLMAdapter, Role
from pipecat.frames.frames import (
    AggregationType,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
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
from pipecat.services.llm_service import LLMService
from pipecat.utils.time import time_now_iso8601

try:
    from aws_sdk_bedrock_runtime.client import (
        BedrockRuntimeClient,
        InvokeModelWithBidirectionalStreamOperationInput,
    )
    from aws_sdk_bedrock_runtime.config import Config
    from aws_sdk_bedrock_runtime.models import (
        BidirectionalInputPayloadPart,
        InvokeModelWithBidirectionalStreamInput,
        InvokeModelWithBidirectionalStreamInputChunk,
        InvokeModelWithBidirectionalStreamOperationOutput,
        InvokeModelWithBidirectionalStreamOutput,
    )
    from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
    from smithy_aws_core.identity.static import StaticCredentialsResolver
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
        endpointing_sensitivity: Controls how quickly Nova Sonic decides the
            user has stopped speaking. Can be "LOW", "MEDIUM", or "HIGH", with
            "HIGH" being the most sensitive (i.e., causing the model to respond
            most quickly).
            If not set, uses the model's default behavior.
            Only supported with Nova 2 Sonic (the default model).
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

    # Turn-taking
    endpointing_sensitivity: Optional[str] = Field(default=None)


class SessionContinuationParams(BaseModel):
    """Configuration for automatic session continuation.

    Nova Sonic sessions have an AWS-imposed time limit (~8 minutes). When enabled,
    session continuation proactively creates a new session in the background before
    the limit is reached, buffers user audio during the transition, and seamlessly
    hands off — preserving conversation context with no user-perceptible gap.

    Parameters:
        enabled: Whether automatic session continuation is enabled.
        transition_threshold_seconds: How many seconds into a session to begin
            monitoring for a transition opportunity. The transition will occur
            when the assistant next starts speaking after this threshold.
            Must not exceed 420 seconds (7 minutes) to ensure the transition
            completes before the hard ~8-minute AWS limit.
        audio_buffer_duration_seconds: Duration of the rolling audio buffer
            (in seconds) that captures user audio during the transition window.
            This audio is replayed into the new session so no user input is lost.
        audio_start_timeout_seconds: Maximum time to wait for the assistant to
            start speaking after the threshold is reached. If no assistant audio
            arrives within this window, the transition is forced. Set to 0 to
            disable the timeout (wait indefinitely).
    """

    enabled: bool = Field(default=True)
    transition_threshold_seconds: float = Field(default=360.0, le=420.0)
    audio_buffer_duration_seconds: float = Field(default=3.0)
    audio_start_timeout_seconds: float = Field(default=80.0)


@dataclass
class _NextSession:
    """Holds pre-created resources for the next session during a transition.

    This is an internal data structure — not part of the public API.
    """

    client: Any  # BedrockRuntimeClient
    stream: Any  # DuplexEventStream
    prompt_name: str
    input_audio_content_name: str


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
        model: str = "amazon.nova-2-sonic-v1:0",
        voice_id: str = "matthew",
        params: Optional[Params] = None,
        session_continuation: Optional[SessionContinuationParams] = None,
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
                Supported regions:
                - Nova 2 Sonic (the default model): "us-east-1", "us-west-2", "ap-northeast-1"
                - Nova Sonic (the older model): "us-east-1", "ap-northeast-1"
            model: Model identifier. Defaults to "amazon.nova-2-sonic-v1:0".
            voice_id: Voice ID for speech synthesis.
                Note that some voices are designed for use with a specific language.
                Options:
                - Nova 2 Sonic (the default model): see https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html
                - Nova Sonic (the older model): see https://docs.aws.amazon.com/nova/latest/userguide/available-voices.html.
            params: Model parameters for audio configuration and inference.
            session_continuation: Configuration for automatic session continuation.
                When enabled, sessions are seamlessly rotated before the AWS time
                limit (~8 minutes) with no user-perceptible interruption.
            system_instruction: System-level instruction for the model.
            tools: Available tools/functions for the model to use.
            send_transcription_frames: Whether to emit transcription frames.

                .. deprecated:: 0.0.91
                    This parameter is deprecated and will be removed in a future version.
                    Transcription frames are always sent.

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

        # Validate endpointing_sensitivity parameter
        if (
            self._params.endpointing_sensitivity
            and not self._is_endpointing_sensitivity_supported()
        ):
            logger.warning(
                f"endpointing_sensitivity is not supported for model '{model}' and will be ignored. "
                "This parameter is only supported starting with Nova 2 Sonic (amazon.nova-2-sonic-v1:0)."
            )
            self._params.endpointing_sensitivity = None

        if not send_transcription_frames:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`send_transcription_frames` is deprecated and will be removed in a future version. "
                    "Transcription frames are always sent.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self._context: Optional[LLMContext] = None
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
        self._may_need_repush_assistant_text = False
        self._ready_to_send_context = False
        self._handling_bot_stopped_speaking = False
        self._triggering_assistant_response = False
        self._waiting_for_trigger_transcription = False
        self._disconnecting = False
        self._connected_time: Optional[float] = None
        self._wants_connection = False
        self._user_text_buffer = ""
        self._assistant_text_buffer = ""
        self._completed_tool_calls = set()

        # Session continuation state
        self._session_continuation = session_continuation or SessionContinuationParams()
        self._next_session: Optional[_NextSession] = None
        self._sc_audio_buffer: Deque[bytes] = deque()
        self._sc_audio_buffer_max_bytes: int = int(
            self._session_continuation.audio_buffer_duration_seconds
            * self._params.input_sample_rate
            * (self._params.input_sample_size / 8)
            * self._params.input_channel_count
        )
        self._sc_is_buffering = False
        self._sc_waiting_for_audio_start = False
        self._sc_waiting_for_completion = False
        self._sc_handoff_in_progress = False
        self._sc_speculative_text_count = 0
        self._sc_final_text_count = 0
        self._sc_audio_start_wait_time: Optional[float] = None
        self._sc_next_session_created_time: Optional[float] = None
        self._sc_barge_in_detected = False
        self._sc_conversation_history: List[Dict[str, str]] = []  # {"role": ..., "text": ...}
        self._sc_max_single_message_bytes = 1024
        self._sc_max_chat_history_bytes = 40960
        # NOTE: This parallel history list duplicates state that pipecat's LLMContext
        # already tracks via the pipeline aggregator. However, the context is updated
        # asynchronously (frames travel through the pipeline), so it can lag behind
        # the actual conversation at handoff time.
        self._sc_monitor_task: Optional[asyncio.Task] = None

        file_path = files("pipecat.services.aws.nova_sonic").joinpath("ready.wav")
        with wave.open(file_path.open("rb"), "rb") as wav_file:
            self._assistant_response_trigger_audio = wav_file.readframes(wav_file.getnframes())

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
        self._sc_start_monitor()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connections.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        self._wants_connection = False
        await self._sc_stop_monitor()
        await self._sc_cleanup_next_session()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close connections.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        self._wants_connection = False
        await self._sc_stop_monitor()
        await self._sc_cleanup_next_session()
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

        # Grab context to carry through disconnect/reconnect
        context = self._context

        await self._disconnect()
        await self._start_connecting()
        await self._handle_context(context)

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

        if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            context = (
                frame.context
                if isinstance(frame, LLMContextFrame)
                else LLMContext.from_openai_context(frame.context)
            )
            await self._handle_context(context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio_frame(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(delay_to_catch_trailing_assistant_text=True)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption_frame()

        await self.push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        if self._disconnecting:
            return

        if not self._context:
            # We got our initial context
            # Try to finish connecting
            self._context = context
            await self._finish_connecting_if_context_available()
        else:
            # We got an updated context
            # Send results for any newly-completed function calls
            await self._process_completed_function_calls(send_new_results=True)

    async def _handle_input_audio_frame(self, frame: InputAudioRawFrame):
        # Wait until we're done sending the assistant response trigger audio before sending audio
        # from the user's mic
        if self._triggering_assistant_response:
            return

        # Buffer audio for session continuation if we're in the transition window
        if self._sc_is_buffering:
            self._sc_audio_buffer.append(frame.audio)
            # Enforce rolling buffer size limit
            total = sum(len(c) for c in self._sc_audio_buffer)
            while total > self._sc_audio_buffer_max_bytes and self._sc_audio_buffer:
                removed = self._sc_audio_buffer.popleft()
                total -= len(removed)

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

    async def _handle_interruption_frame(self):
        if self._assistant_is_responding:
            self._may_need_repush_assistant_text = True

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
            await self.push_error(error_msg=f"Initialization error: {e}", exception=e)
            await self._disconnect()

    async def _process_completed_function_calls(self, send_new_results: bool):
        # Check for set of completed function calls in the context
        for message in self._context.get_messages():
            if message.get("role") and message.get("content") not in ["IN_PROGRESS", "CANCELLED"]:
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in self._completed_tool_calls:
                    # Found a newly-completed function call - send the result to the service
                    if send_new_results:
                        await self._send_tool_result(tool_call_id, message.get("content"))
                    self._completed_tool_calls.add(tool_call_id)

    async def _finish_connecting_if_context_available(self):
        # We can only finish connecting once we've gotten our initial context and we're ready to
        # send it
        if not (self._context and self._ready_to_send_context):
            return

        logger.info("Finishing connecting (setting up session)...")

        # Initialize our bookkeeping of already-completed tool calls in the
        # context
        await self._process_completed_function_calls(send_new_results=False)

        # Read context
        adapter: AWSNovaSonicLLMAdapter = self.get_llm_adapter()
        llm_connection_params = adapter.get_llm_invocation_params(self._context)

        # Send prompt start event, specifying tools.
        # Tools from context take priority over self._tools.
        tools = (
            llm_connection_params["tools"]
            if llm_connection_params["tools"]
            else adapter.from_standard_tools(self._tools)
        )
        logger.debug(f"Using tools: {tools}")
        await self._send_prompt_start_event(tools)

        # Send system instruction.
        # Instruction from context takes priority over self._system_instruction.
        system_instruction = (
            llm_connection_params["system_instruction"]
            if llm_connection_params["system_instruction"]
            else self._system_instruction
        )
        logger.debug(f"Using system instruction: {system_instruction}")
        if system_instruction:
            await self._send_text_event(text=system_instruction, role=Role.SYSTEM)

        # Send conversation history
        for message in llm_connection_params["messages"]:
            # logger.debug(f"Seeding conversation history with message: {message}")
            await self._send_text_event(text=message.text, role=message.role)
            # Seed session continuation history with initial context messages
            if self._session_continuation.enabled:
                self._sc_add_to_history(message.role.value, message.text)

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

            # Clean up context
            self._context = None

            # Clean up stream
            if self._stream:
                await self._stream.close()
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
            # Should be all private state except:
            # - _wants_connection
            # - _assistant_response_trigger_audio
            # - _session_continuation (config)
            # - _sc_monitor_task (managed separately)
            self._prompt_name = None
            self._input_audio_content_name = None
            self._content_being_received = None
            self._assistant_is_responding = False
            self._may_need_repush_assistant_text = False
            self._ready_to_send_context = False
            self._handling_bot_stopped_speaking = False
            self._triggering_assistant_response = False
            self._waiting_for_trigger_transcription = False
            self._disconnecting = False
            self._connected_time = None
            self._user_text_buffer = ""
            self._assistant_text_buffer = ""
            self._completed_tool_calls = set()
            self._sc_waiting_for_audio_start = False
            self._sc_handoff_in_progress = False
            self._sc_is_buffering = False
            self._sc_barge_in_detected = False
            self._sc_speculative_text_count = 0
            self._sc_final_text_count = 0
            self._sc_audio_buffer.clear()

            logger.info("Finished disconnecting")
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)

    def _create_client(self) -> BedrockRuntimeClient:
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._region}.amazonaws.com",
            region=self._region,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            aws_session_token=self._session_token,
            aws_credentials_identity_resolver=StaticCredentialsResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
        )
        return BedrockRuntimeClient(config=config)

    def _is_first_generation_sonic_model(self) -> bool:
        # Nova Sonic (the older model) is identified by "amazon.nova-sonic-v1:0"
        return self._model == "amazon.nova-sonic-v1:0"

    def _is_endpointing_sensitivity_supported(self) -> bool:
        # endpointing_sensitivity is only supported with Nova 2 Sonic (and,
        # presumably, future models)
        return not self._is_first_generation_sonic_model()

    def _is_assistant_response_trigger_needed(self) -> bool:
        # Assistant response trigger audio is only needed with the older model
        return self._is_first_generation_sonic_model()

    #
    # LLM communication: input events (pipecat -> LLM)
    #

    async def _send_session_start_event(self):
        await self._send_client_event(self._build_session_start_event_json())

    async def _send_prompt_start_event(self, tools: List[Any], *, prompt_name=None, stream=None):
        """Send promptStart event. Defaults to current session."""
        target_prompt = prompt_name or self._prompt_name
        if not target_prompt:
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
              "promptName": "{target_prompt}",
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
        await self._send_client_event(prompt_start, stream)

    async def _send_audio_input_start_event(
        self, *, prompt_name=None, content_name=None, stream=None
    ):
        """Send audio input contentStart. Defaults to current session."""
        target_prompt = prompt_name or self._prompt_name
        target_content = content_name or self._input_audio_content_name
        if not target_prompt:
            return

        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{target_prompt}",
                    "contentName": "{target_content}",
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
        await self._send_client_event(audio_content_start, stream)

    async def _send_text_event(
        self, text: str, role: Role, *, prompt_name=None, stream=None, interactive: bool = True
    ):
        """Send a text content block. Defaults to current session."""
        target_prompt = prompt_name or self._prompt_name
        if not text or not target_prompt:
            return

        content_name = str(uuid.uuid4())
        interactive_str = "true" if interactive else "false"

        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{target_prompt}",
                    "contentName": "{content_name}",
                    "type": "TEXT",
                    "interactive": {interactive_str},
                    "role": "{role.value}",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(text_content_start, stream)

        escaped_text = json.dumps(text)
        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{target_prompt}",
                    "contentName": "{content_name}",
                    "content": {escaped_text}
                }}
            }}
        }}
        '''
        await self._send_client_event(text_input, stream)

        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{target_prompt}",
                    "contentName": "{content_name}"
                }}
            }}
        }}
        '''
        await self._send_client_event(text_content_end, stream)

    async def _send_user_audio_event(
        self, audio: bytes, *, prompt_name=None, content_name=None, stream=None
    ):
        """Send an audio chunk. Defaults to current session."""
        target_stream = stream or self._stream
        if not target_stream:
            return

        target_prompt = prompt_name or self._prompt_name
        target_content = content_name or self._input_audio_content_name

        blob = base64.b64encode(audio)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{target_prompt}",
                    "contentName": "{target_content}",
                    "content": "{blob.decode("utf-8")}"
                }}
            }}
        }}
        '''
        await self._send_client_event(audio_event, target_stream)

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

    async def _send_client_event(self, event_json: str, stream=None):
        """Send an event to a stream. Defaults to the current session's stream."""
        target_stream = stream or self._stream
        if not target_stream:
            return

        # Log event type for debugging (extract first key from the event JSON)
        try:
            parsed = json.loads(event_json)
            if "event" in parsed:
                event_type = next(iter(parsed["event"]), "unknown")
                is_next = stream is not None and stream is not self._stream
                session_label = "next" if is_next else "current"
                # Skip noisy audioInput events
                if event_type != "audioInput":
                    logger.debug(f"Sending {event_type} to {session_label} session")
        except Exception:
            pass

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await target_stream.input_stream.send(event)

    async def _receive_task_handler(self):
        # Capture the stream reference so session continuation swaps don't
        # cause us to read from a different stream mid-loop.
        stream = self._stream
        try:
            while stream and not self._disconnecting:
                output = await stream.await_output()
                result = await output[1].receive()

                # Guard against None results from cancelled CRT futures
                # (happens during session continuation handoff)
                if not result or not result.value or not result.value.bytes_:
                    continue

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
        except asyncio.CancelledError:
            # Expected during session continuation handoff
            return
        except Exception as e:
            # If this receive loop's stream has been replaced by session continuation,
            # exit silently — the new receive loop is handling the new stream.
            if stream is not self._stream:
                return
            if self._disconnecting or self._sc_handoff_in_progress:
                # Errors are expected while disconnecting or during handoff
                return
            await self.push_error(error_msg=f"Error processing responses: {e}", exception=e)
            if self._wants_connection:
                await self.reset_conversation()

    async def _handle_completion_start_event(self, event_json):
        completion_start = event_json.get("completionStart", {})
        session_id = completion_start.get("sessionId", "unknown")
        logger.info(f"completionStart received (sessionId={session_id})")

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

                # Session continuation: assistant audio start is our trigger to begin
                # buffering. Wait for a completion signal (text counts match, barge-in, etc.)
                if self._sc_waiting_for_audio_start and not self._sc_handoff_in_progress:
                    self._sc_waiting_for_audio_start = False
                    self._sc_audio_start_wait_time = None
                    self._sc_is_buffering = True
                    self._sc_waiting_for_completion = True
                    logger.info(
                        "Session continuation: assistant audio started, "
                        "buffering user audio and creating next session"
                    )
                    if not self._next_session:
                        try:
                            await self._sc_prepare_next_session()
                        except Exception as e:
                            logger.error(
                                f"Session continuation: failed to prepare next session: {e}"
                            )
                            self._sc_is_buffering = False
                            self._sc_waiting_for_completion = False

        # Session continuation: user contentStart during transition means the user
        # is speaking — treat as barge-in and trigger handoff immediately if we're
        # in a forced transition (no assistant text yet)
        if (
            content.role == Role.USER
            and self._sc_waiting_for_completion
            and not self._sc_handoff_in_progress
            and self._next_session
        ):
            if self._sc_final_text_count == 0:
                logger.info(
                    "Session continuation: user spoke during forced transition "
                    "(no assistant response yet) — completing handoff immediately"
                )
                self._sc_waiting_for_completion = False
                self.create_task(self._sc_execute_handoff(), name="sc_execute_handoff")

    async def _handle_text_output_event(self, event_json):
        if not self._content_being_received:  # should never happen
            return
        content = self._content_being_received

        text_content = event_json["textOutput"]["content"]

        # Bookkeeping: augment the current content being received with text
        # Assumption: only one text content per content block
        content.text_content = text_content

        # Session continuation: track speculative/final text counts across the entire
        # session (not just during transition). The reference solution counts all text
        # events, and only checks the equality when waiting_for_completion is True.
        # This ensures speculative texts that arrive before the transition flag is set
        # are properly counted.
        if (
            self._session_continuation.enabled
            and content.role == Role.ASSISTANT
            and content.type == ContentType.TEXT
        ):
            if content.text_stage == TextStage.SPECULATIVE:
                self._sc_speculative_text_count += 1
                logger.debug(
                    f"Session continuation: SPECULATIVE text #{self._sc_speculative_text_count}"
                )
            elif content.text_stage == TextStage.FINAL:
                self._sc_final_text_count += 1
                logger.debug(
                    f"Session continuation: FINAL text #{self._sc_final_text_count} "
                    f"(speculative={self._sc_speculative_text_count})"
                )

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
                    # Session continuation: track FINAL assistant text in history
                    if content.text_content and self._session_continuation.enabled:
                        self._sc_add_to_history("ASSISTANT", content.text_content)

                    # Session continuation: check if text counts match AFTER adding
                    # to history, so the latest text is included in the handoff.
                    if (
                        self._sc_waiting_for_completion
                        and self._sc_speculative_text_count > 0
                        and self._sc_final_text_count > 0
                        and self._sc_speculative_text_count == self._sc_final_text_count
                        and not self._sc_handoff_in_progress
                    ):
                        logger.info(
                            f"Session continuation: completion signal — text pairs matched "
                            f"({self._sc_speculative_text_count}={self._sc_final_text_count})"
                        )
                        self._sc_waiting_for_completion = False
                        self.create_task(self._sc_execute_handoff(), name="sc_execute_handoff")

                # Session continuation: track barge-in for count recovery.
                # When a barge-in occurs, speculative texts may not get their matching
                # FINAL texts. We record this so that when the user next speaks, we can
                # equalize the counts (matching the reference solution's logic).
                if stop_reason == "INTERRUPTED" and self._session_continuation.enabled:
                    self._sc_barge_in_detected = True

                # Session continuation: TEXT contentEnd with INTERRUPTED is a completion
                # signal (barge-in). Trigger handoff immediately.
                if (
                    stop_reason == "INTERRUPTED"
                    and self._sc_waiting_for_completion
                    and not self._sc_handoff_in_progress
                ):
                    logger.info(
                        "Session continuation: completion signal — TEXT INTERRUPTED (barge-in)"
                    )
                    self._sc_waiting_for_completion = False
                    self.create_task(self._sc_execute_handoff(), name="sc_execute_handoff")

        elif content.role == Role.USER:
            if content.type == ContentType.TEXT:
                if content.text_stage == TextStage.FINAL:
                    # User transcription text added
                    await self._report_user_transcription_text_added(content.text_content)
                    # Session continuation: track user text in history
                    if content.text_content and self._session_continuation.enabled:
                        self._sc_add_to_history("USER", content.text_content)

                    # Session continuation: barge-in count recovery.
                    # When a barge-in interrupted an assistant response, some speculative
                    # texts may never get matching FINAL texts. When the user speaks next
                    # (confirming no more FINAL texts will arrive), equalize the counts so
                    # the next transition's text-pair matching works correctly.
                    if (
                        self._sc_barge_in_detected
                        and not self._sc_waiting_for_completion
                        and not self._sc_handoff_in_progress
                        and self._sc_speculative_text_count > self._sc_final_text_count
                    ):
                        logger.debug(
                            f"Session continuation: barge-in recovery — setting "
                            f"final={self._sc_speculative_text_count} "
                            f"(was {self._sc_final_text_count})"
                        )
                        self._sc_final_text_count = self._sc_speculative_text_count
                        self._sc_barge_in_detected = False

    async def _handle_completion_end_event(self, event_json):
        # Session continuation: completionEnd is a fallback completion signal.
        # If we're still waiting for completion when the entire turn ends,
        # trigger handoff now — the assistant is done.
        if self._sc_waiting_for_completion and not self._sc_handoff_in_progress:
            logger.info("Session continuation: completion signal — completionEnd (fallback)")
            self._sc_waiting_for_completion = False
            self.create_task(self._sc_execute_handoff(), name="sc_execute_handoff")

    #
    # assistant response reporting
    #
    # 1. Started
    # 2. Text added
    # 3. Ended
    #

    async def _report_assistant_response_started(self):
        logger.debug("Assistant response started")

        # Report the start of the assistant response.
        await self.push_frame(LLMFullResponseStartFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) started
        await self.push_frame(TTSStartedFrame())

    async def _report_assistant_response_text_added(self, text):
        if not self._context:  # should never happen
            return

        logger.debug(f"Assistant response text added: {text}")

        # Report the text of the assistant response.
        frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        frame.includes_inter_frame_spaces = True
        await self.push_frame(frame)

        # HACK: here we're also buffering the assistant text ourselves as a
        # backup rather than relying solely on the assistant context aggregator
        # to do it, because the text arrives from Nova Sonic only after all the
        # assistant audio frames have been pushed, meaning that if an
        # interruption frame were to arrive we would lose all of it (the text
        # frames sitting in the queue would be wiped).
        self._assistant_text_buffer += text

    async def _report_assistant_response_ended(self):
        if not self._context:  # should never happen
            return

        logger.debug("Assistant response ended")

        # If an interruption frame arrived while the assistant was responding
        # we may have lost all of the assistant text (see HACK, above), so
        # re-push it downstream to the aggregator now.
        if self._may_need_repush_assistant_text:
            # Just in case, check that assistant text hasn't already made it
            # into the context (sometimes it does, despite the interruption).
            messages = self._context.get_messages()
            last_message = messages[-1] if messages else None
            if (
                not last_message
                or last_message.get("role") != "assistant"
                or last_message.get("content") != self._assistant_text_buffer
            ):
                # We also need to re-push the LLMFullResponseStartFrame since the
                # TTSTextFrame would be ignored otherwise (the interruption frame
                # would have cleared the assistant aggregator state).
                await self.push_frame(LLMFullResponseStartFrame())
                frame = TTSTextFrame(
                    self._assistant_text_buffer, aggregated_by=AggregationType.SENTENCE
                )
                frame.includes_inter_frame_spaces = True
                await self.push_frame(frame)
            self._may_need_repush_assistant_text = False

        # Report the end of the assistant response.
        await self.push_frame(LLMFullResponseEndFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) stopped.
        await self.push_frame(TTSStoppedFrame())

        # Clear out the buffered assistant text
        self._assistant_text_buffer = ""

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

        # HACK: here we're buffering the user text ourselves rather than
        # relying on the upstream user context aggregator to do it, because the
        # text arrives in fairly large chunks spaced fairly far apart in time.
        # That means the user text would be split between different messages in
        # context. Even if we sent placeholder InterimTranscriptionFrames in
        # between each TranscriptionFrame to tell the aggregator to hold off on
        # finalizing the user message, the aggregator would likely get the last
        # chunk too late.
        self._user_text_buffer += f" {text}" if self._user_text_buffer else text

    async def _report_user_transcription_ended(self):
        if not self._context:  # should never happen
            return

        logger.debug(f"User transcription ended")

        # Report to the upstream user context aggregator that some new user
        # transcription text is available.

        # HACK: Check if this transcription was triggered by our own
        # assistant response trigger. If so, we need to wrap it with
        # UserStarted/StoppedSpeakingFrames; otherwise the user aggregator
        # would fire an EmulatedUserStartedSpeakingFrame, which would
        # trigger an interruption, which would prevent us from writing the
        # assistant response to context.
        #
        # Sending an EmulateUserStartedSpeakingFrame ourselves doesn't
        # work: it just causes the interruption we're trying to avoid.
        #
        # Setting enable_emulated_vad_interruptions also doesn't work: at
        # the time the user aggregator receives the TranscriptionFrame, it
        # doesn't yet know the assistant has started responding, so it
        # doesn't know that emulating the user starting to speak would
        # cause an interruption.
        should_wrap_in_user_started_stopped_speaking_frames = (
            self._waiting_for_trigger_transcription
            and self._user_text_buffer.strip().lower() == "ready"
        )

        # Start wrapping the upstream transcription in UserStarted/StoppedSpeakingFrames if needed
        if should_wrap_in_user_started_stopped_speaking_frames:
            logger.debug(
                "Wrapping assistant response trigger transcription with upstream UserStarted/StoppedSpeakingFrames"
            )
            await self.push_frame(UserStartedSpeakingFrame(), direction=FrameDirection.UPSTREAM)

        # Send the transcription upstream for the user context aggregator
        frame = TranscriptionFrame(
            text=self._user_text_buffer, user_id="", timestamp=time_now_iso8601()
        )
        await self.push_frame(frame, direction=FrameDirection.UPSTREAM)

        # Finish wrapping the upstream transcription in UserStarted/StoppedSpeakingFrames if needed
        if should_wrap_in_user_started_stopped_speaking_frames:
            await self.push_frame(UserStoppedSpeakingFrame(), direction=FrameDirection.UPSTREAM)

        # Clear out the buffered user text
        self._user_text_buffer = ""

        # We're no longer waiting for a trigger transcription
        self._waiting_for_trigger_transcription = False

    #
    # session continuation
    #
    # Proactively rotates sessions before the AWS ~8-minute limit.
    # Flow:
    #   1. Monitor task detects session age >= threshold
    #   2. Sets _sc_waiting_for_audio_start = True
    #   3. When assistant AUDIO contentStart arrives, starts buffering user audio
    #      and kicks off _sc_execute_handoff()
    #   4. _sc_execute_handoff waits for the next session to be ready, then:
    #      a. Sends conversation history to the new session
    #      b. Sends buffered user audio to the new session
    #      c. Promotes the new session to current
    #      d. Closes the old session in the background
    #

    def _sc_start_monitor(self):
        """Start the session duration monitor if session continuation is enabled."""
        if not self._session_continuation.enabled:
            return
        if self._sc_monitor_task:
            return
        self._sc_monitor_task = self.create_task(self._sc_monitor_loop(), name="sc_monitor_loop")

    async def _sc_stop_monitor(self):
        """Stop the session duration monitor."""
        if self._sc_monitor_task:
            await self.cancel_task(self._sc_monitor_task, timeout=1.0)
            self._sc_monitor_task = None

    async def _sc_monitor_loop(self):
        """Periodically check session age and pre-create the next session when threshold is reached."""
        try:
            while self._wants_connection:
                await asyncio.sleep(1)

                if not self._connected_time or self._sc_handoff_in_progress:
                    continue

                session_age = time.time() - self._connected_time
                threshold = self._session_continuation.transition_threshold_seconds

                if (
                    session_age >= threshold
                    and not self._sc_waiting_for_audio_start
                    and not self._next_session
                ):
                    logger.info(
                        f"Session continuation: session age {session_age:.0f}s >= "
                        f"threshold {threshold:.0f}s, preparing next session"
                    )
                    self._sc_waiting_for_audio_start = True
                    self._sc_audio_start_wait_time = time.time()

                # Timeout: if we've been waiting for assistant audio too long, force transition
                audio_start_timeout = self._session_continuation.audio_start_timeout_seconds
                if (
                    self._sc_waiting_for_audio_start
                    and self._sc_audio_start_wait_time
                    and audio_start_timeout > 0
                    and (time.time() - self._sc_audio_start_wait_time) > audio_start_timeout
                ):
                    logger.info(
                        f"Session continuation: TIMEOUT — no assistant audio after "
                        f"{audio_start_timeout:.0f}s, forcing transition"
                    )
                    self._sc_waiting_for_audio_start = False
                    self._sc_audio_start_wait_time = None
                    self._sc_is_buffering = True
                    self._sc_waiting_for_completion = False
                    try:
                        if not self._next_session:
                            await self._sc_prepare_next_session()
                        self.create_task(self._sc_execute_handoff(), name="sc_execute_handoff")
                    except Exception as e:
                        logger.error(f"Session continuation: forced transition failed: {e}")
                        self._sc_is_buffering = False

                # Dead session detection: if the next session has been alive too long
                # without a handoff, it may have timed out on the server side. Recreate it.
                # Matches the reference's _monitor_next_session_readiness / _recreate_next_session.
                next_session_timeout = 30.0
                if (
                    self._next_session
                    and self._sc_next_session_created_time
                    and not self._sc_handoff_in_progress
                    and (time.time() - self._sc_next_session_created_time) > next_session_timeout
                ):
                    logger.warning(
                        f"Session continuation: next session idle for "
                        f">{next_session_timeout:.0f}s, recreating"
                    )
                    await self._sc_cleanup_next_session()
                    try:
                        await self._sc_prepare_next_session()
                    except Exception as e:
                        logger.error(f"Session continuation: failed to recreate next session: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Session continuation monitor error: {e}")

    async def _sc_prepare_next_session(self):
        """Create a new Bedrock client/stream and send setup (session, prompt, system instruction).

        Conversation history is handled at handoff time so it includes the latest exchange.
        audioInputStart and audio bytes are also deferred to handoff time.
        """
        prompt_name = str(uuid.uuid4())
        input_audio_content_name = str(uuid.uuid4())

        client = self._create_client()
        stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model)
        )

        self._next_session = _NextSession(
            client=client,
            stream=stream,
            prompt_name=prompt_name,
            input_audio_content_name=input_audio_content_name,
        )
        self._sc_next_session_created_time = time.time()

        ns = self._next_session

        # Send sessionStart
        await self._send_client_event(self._build_session_start_event_json(), ns.stream)

        # Build context params from current context
        if self._context:
            adapter: AWSNovaSonicLLMAdapter = self.get_llm_adapter()
            llm_params = adapter.get_llm_invocation_params(self._context)

            # Send promptStart with tools
            tools = (
                llm_params["tools"]
                if llm_params["tools"]
                else adapter.from_standard_tools(self._tools)
            )
            await self._send_prompt_start_event(tools, prompt_name=ns.prompt_name, stream=ns.stream)

            # Send system instruction
            system_instruction = (
                llm_params["system_instruction"]
                if llm_params["system_instruction"]
                else self._system_instruction
            )
            if system_instruction:
                await self._send_text_event(
                    system_instruction,
                    Role.SYSTEM,
                    prompt_name=ns.prompt_name,
                    stream=ns.stream,
                    interactive=False,
                )

            # NOTE: conversation history is NOT sent here — it's deferred to
            # handoff time (_sc_execute_handoff) so it includes the latest exchange.
        else:
            logger.warning("Session continuation: no context available during preparation")

    async def _sc_execute_handoff(self):
        """Execute the session handoff: send audioInputStart + buffered audio, then promote.

        We promote immediately after sending audio. The API sends completionStart
        only after audio bytes arrive, and it will arrive naturally on the new session's
        receive loop after promotion.
        """
        if self._sc_handoff_in_progress:
            return
        self._sc_handoff_in_progress = True

        try:
            ns = self._next_session
            if not ns:
                logger.warning("Session continuation: no next session available for handoff")
                return

            logger.info("Session continuation: executing handoff")

            # Send conversation history to the next session. This uses our own
            # real-time history (updated as FINAL text events arrive), not the
            # pipeline context which may lag.
            if self._sc_conversation_history:
                logger.info(
                    f"Session continuation: sending {len(self._sc_conversation_history)} "
                    f"history messages to new session"
                )
                for msg in self._sc_conversation_history:
                    logger.debug(
                        f"Session continuation: history [{msg['role']}]: "
                        f"{msg['text'][:80]}{'...' if len(msg['text']) > 80 else ''}"
                    )
                    await self._send_text_event(
                        msg["text"],
                        Role(msg["role"]),
                        prompt_name=ns.prompt_name,
                        stream=ns.stream,
                        interactive=False,
                    )

            # Send audio input start to the next session
            await self._send_audio_input_start_event(
                prompt_name=ns.prompt_name,
                content_name=ns.input_audio_content_name,
                stream=ns.stream,
            )

            # Send buffered audio to the new session
            buffer_chunks = list(self._sc_audio_buffer)
            buffer_duration = sum(len(c) for c in buffer_chunks) / (
                self._params.input_sample_rate
                * (self._params.input_sample_size / 8)
                * self._params.input_channel_count
            )
            logger.info(
                f"Session continuation: sending {len(buffer_chunks)} buffered audio chunks "
                f"(~{buffer_duration:.1f}s) to new session"
            )
            for chunk in buffer_chunks:
                await self._send_user_audio_event(
                    chunk,
                    prompt_name=ns.prompt_name,
                    content_name=ns.input_audio_content_name,
                    stream=ns.stream,
                )

            # === Promote: swap the new session into place immediately ===
            # Promote FIRST, then close old session in background.
            # Do NOT cancel the old receive task here — it captured the stream locally
            # and will exit naturally when the old stream is closed in the background.
            logger.info("Session continuation: promoting new session")

            # Save old session resources for background cleanup
            old_client = self._client
            old_stream = self._stream
            old_receive_task = self._receive_task
            old_prompt_name = self._prompt_name
            old_input_audio_content_name = self._input_audio_content_name

            # Swap in the new session
            self._client = ns.client
            self._stream = ns.stream
            self._prompt_name = ns.prompt_name
            self._input_audio_content_name = ns.input_audio_content_name
            self._connected_time = time.time()
            self._next_session = None

            # Start the main receive loop on the new stream
            self._receive_task = self.create_task(self._receive_task_handler())

            # Stop buffering
            self._sc_is_buffering = False
            self._sc_audio_buffer.clear()

            logger.info("Session continuation: handoff complete, new session is active")

            # Close old session in background — send end events, close stream,
            # and the old receive task will exit naturally via exception path
            self.create_task(
                self._sc_close_old_session(
                    old_client,
                    old_stream,
                    old_receive_task,
                    old_prompt_name,
                    old_input_audio_content_name,
                ),
                name="sc_close_old",
            )

        except Exception as e:
            logger.error(f"Session continuation: handoff error: {e}")
            await self._sc_cleanup_next_session()
        finally:
            self._sc_handoff_in_progress = False
            self._sc_waiting_for_audio_start = False
            self._sc_waiting_for_completion = False
            self._sc_barge_in_detected = False
            self._sc_speculative_text_count = 0
            self._sc_final_text_count = 0
            self._sc_audio_start_wait_time = None
            self._sc_next_session_created_time = None

    async def _sc_close_old_session(
        self, client, stream, receive_task, prompt_name, input_audio_content_name
    ):
        """Close the old session's resources in the background.

        Send end events, close the input stream, and let the receive loop exit naturally.
        The receive loop captured the stream locally and will get an error when the stream is closed,
        exiting via the exception handler that checks _sc_handoff_in_progress.
        """
        try:
            # Send session end events on the old stream (best effort).
            # The correct shutdown sequence is: audio contentEnd → promptEnd → sessionEnd
            # (matching the reference solution's _close_stream_in_background).
            if stream and prompt_name:
                try:
                    # Close the audio input content first
                    if input_audio_content_name:
                        audio_content_end_json = json.dumps(
                            {
                                "event": {
                                    "contentEnd": {
                                        "promptName": prompt_name,
                                        "contentName": input_audio_content_name,
                                    }
                                }
                            }
                        )
                        event = InvokeModelWithBidirectionalStreamInputChunk(
                            value=BidirectionalInputPayloadPart(
                                bytes_=audio_content_end_json.encode("utf-8")
                            )
                        )
                        await stream.input_stream.send(event)

                    prompt_end_json = json.dumps(
                        {"event": {"promptEnd": {"promptName": prompt_name}}}
                    )
                    session_end_json = json.dumps({"event": {"sessionEnd": {}}})
                    event = InvokeModelWithBidirectionalStreamInputChunk(
                        value=BidirectionalInputPayloadPart(bytes_=prompt_end_json.encode("utf-8"))
                    )
                    await stream.input_stream.send(event)
                    event = InvokeModelWithBidirectionalStreamInputChunk(
                        value=BidirectionalInputPayloadPart(bytes_=session_end_json.encode("utf-8"))
                    )
                    await stream.input_stream.send(event)
                except Exception:
                    pass

            # Close the input stream — this causes the old receive loop's
            # await_output() to error out and exit naturally
            if stream:
                try:
                    await asyncio.wait_for(stream.input_stream.close(), timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass

            # Give the old receive loop a moment to exit naturally
            await asyncio.sleep(0.5)

            # Cancel receive task as a fallback if it hasn't exited yet
            if receive_task:
                try:
                    await self.cancel_task(receive_task, timeout=1.0)
                except Exception:
                    pass

            logger.debug("Session continuation: old session closed")
        except Exception as e:
            logger.warning(f"Session continuation: error closing old session: {e}")

    async def _sc_cleanup_next_session(self):
        """Clean up the pre-created next session if it exists."""
        ns = self._next_session
        if not ns:
            return

        if ns.stream:
            try:
                await ns.stream.close()
            except Exception:
                pass

        self._next_session = None
        self._sc_is_buffering = False
        self._sc_audio_buffer.clear()

    def _sc_add_to_history(self, role: str, text: str):
        """Add a message to the session continuation conversation history.

        Applies the same byte-limit enforcement as the reference solution:
        - Truncate individual messages that exceed max_single_message_bytes.
        - Remove oldest messages when total history exceeds max_chat_history_bytes.
        """
        content_bytes = text.encode("utf-8")
        if len(content_bytes) > self._sc_max_single_message_bytes:
            truncation_marker = "... [truncated]"
            max_content_bytes = self._sc_max_single_message_bytes - len(
                truncation_marker.encode("utf-8")
            )
            text = content_bytes[:max_content_bytes].decode("utf-8", errors="ignore")
            text += truncation_marker

        self._sc_conversation_history.append({"role": role, "text": text})

        # Trim oldest messages to stay within total byte limit
        while self._sc_conversation_history:
            total = sum(
                len(m["text"].encode("utf-8")) + len(m["role"].encode("utf-8"))
                for m in self._sc_conversation_history
            )
            if total <= self._sc_max_chat_history_bytes:
                break
            self._sc_conversation_history.pop(0)

    # Session continuation: event helpers for the next session

    def _build_session_start_event_json(self) -> str:
        """Build the sessionStart event JSON (shared between current and next session)."""
        turn_detection_config = (
            f""",
              "turnDetectionConfiguration": {{
                "endpointingSensitivity": "{self._params.endpointing_sensitivity}"
              }}"""
            if self._params.endpointing_sensitivity
            else ""
        )
        return f"""
        {{
          "event": {{
            "sessionStart": {{
              "inferenceConfiguration": {{
                "maxTokens": {self._params.max_tokens},
                "topP": {self._params.top_p},
                "temperature": {self._params.temperature}
              }}{turn_detection_config}
            }}
          }}
        }}
        """

    #
    # context
    #

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> LLMContextAggregatorPair:
        """Create context aggregator pair for managing conversation context.

        NOTE: this method exists only for backward compatibility. New code
        should instead do::

            context = LLMContext(...)
            context_aggregator = LLMContextAggregatorPair(context)

        Args:
            context: The OpenAI LLM context.
            user_params: Parameters for the user context aggregator.
            assistant_params: Parameters for the assistant context aggregator.

        Returns:
            A pair of user and assistant context aggregators.
        """
        context = LLMContext.from_openai_context(context)
        return LLMContextAggregatorPair(
            context, user_params=user_params, assistant_params=assistant_params
        )

    #
    # assistant response trigger
    # HACK: only needed for the older Nova Sonic (as opposed to Nova 2 Sonic) model
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
        """
        if not self._is_assistant_response_trigger_needed():
            logger.warning(
                f"Assistant response trigger not needed for model '{self._model}'; skipping. "
                "An LLMRunFrame() should be sufficient to prompt the assistant to respond, "
                "assuming the context ends in a user message."
            )
            return

        if self._triggering_assistant_response:
            return

        self._triggering_assistant_response = True

        # Send the trigger audio, if we're fully connected and set up
        if self._connected_time:
            await self._send_assistant_response_trigger()

    async def _send_assistant_response_trigger(self):
        if not self._connected_time:
            # should never happen
            return

        try:
            logger.debug("Sending assistant response trigger...")

            self._waiting_for_trigger_transcription = True

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

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic LLM service implementation for Pipecat AI framework.

This module provides a speech-to-speech LLM service using AWS Nova Sonic, which supports
bidirectional audio streaming, text generation, and function calling capabilities.
"""

import asyncio
import base64
import concurrent.futures
import json
import time
import uuid
import wave
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.aws_nova_sonic_adapter import AWSNovaSonicLLMAdapter, Role
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.nova_sonic.session_continuation import (
    SessionContinuationHelper,
    SessionContinuationParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven, assert_given
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

    .. deprecated:: 0.0.105
        Use ``settings=AWSNovaSonicLLMService.Settings(...)`` for inference settings
        and ``audio_config=AudioConfig(...)`` for audio configuration.

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
    input_sample_rate: int | None = Field(default=16000)
    input_sample_size: int | None = Field(default=16)
    input_channel_count: int | None = Field(default=1)

    # Audio output
    output_sample_rate: int | None = Field(default=24000)
    output_sample_size: int | None = Field(default=16)
    output_channel_count: int | None = Field(default=1)

    # Inference
    max_tokens: int | None = Field(default=1024)
    top_p: float | None = Field(default=0.9)
    temperature: float | None = Field(default=0.7)

    # Turn-taking
    endpointing_sensitivity: str | None = Field(default=None)

    @property
    def audio_config(self) -> "AudioConfig":
        """Return an ``AudioConfig`` populated from this instance's audio fields."""
        return AudioConfig(
            input_sample_rate=self.input_sample_rate,
            input_sample_size=self.input_sample_size,
            input_channel_count=self.input_channel_count,
            output_sample_rate=self.output_sample_rate,
            output_sample_size=self.output_sample_size,
            output_channel_count=self.output_channel_count,
        )


class AudioConfig(BaseModel):
    """Audio configuration for AWS Nova Sonic.

    Parameters:
        input_sample_rate: Audio input sample rate in Hz.
        input_sample_size: Audio input sample size in bits.
        input_channel_count: Number of input audio channels.
        output_sample_rate: Audio output sample rate in Hz.
        output_sample_size: Audio output sample size in bits.
        output_channel_count: Number of output audio channels.
    """

    # Input
    input_sample_rate: int | None = Field(default=16000)
    input_sample_size: int | None = Field(default=16)
    input_channel_count: int | None = Field(default=1)

    # Output
    output_sample_rate: int | None = Field(default=24000)
    output_sample_size: int | None = Field(default=16)
    output_channel_count: int | None = Field(default=1)


@dataclass
class AWSNovaSonicLLMSettings(LLMSettings):
    """Settings for AWSNovaSonicLLMService.

    Parameters:
        voice: Voice identifier for speech synthesis.
        endpointing_sensitivity: Controls how quickly Nova Sonic decides the
            user has stopped speaking. Can be "LOW", "MEDIUM", or "HIGH".
    """

    voice: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpointing_sensitivity: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AWSNovaSonicLLMService(LLMService[AWSNovaSonicLLMAdapter]):
    """AWS Nova Sonic speech-to-speech LLM service.

    Provides bidirectional audio streaming, real-time transcription, text generation,
    and function calling capabilities using AWS Nova Sonic model.
    """

    Settings = AWSNovaSonicLLMSettings
    _settings: Settings

    # Override the default adapter to use the AWSNovaSonicLLMAdapter one
    adapter_class = AWSNovaSonicLLMAdapter

    def __init__(
        self,
        *,
        secret_access_key: str,
        access_key_id: str,
        session_token: str | None = None,
        region: str,
        model: str = "amazon.nova-2-sonic-v1:0",
        voice_id: str = "matthew",
        params: Params | None = None,
        audio_config: AudioConfig | None = None,
        settings: Settings | None = None,
        system_instruction: str | None = None,
        tools: ToolsSchema | None = None,
        session_continuation: SessionContinuationParams | None = None,
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

                .. deprecated:: 0.0.105
                    Use ``settings=AWSNovaSonicLLMService.Settings(model=...)`` instead.

            voice_id: Voice ID for speech synthesis.
                Note that some voices are designed for use with a specific language.
                Options:
                - Nova 2 Sonic (the default model): see https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html
                - Nova Sonic (the older model): see https://docs.aws.amazon.com/nova/latest/userguide/available-voices.html.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSNovaSonicLLMService.Settings(voice=...)`` instead.

            params: Model parameters for audio configuration and inference.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSNovaSonicLLMService.Settings(...)`` for inference
                    settings and ``audio_config=AudioConfig(...)`` for audio
                    configuration.

            audio_config: Audio configuration (sample rates, sample sizes,
                channel counts). If not provided, defaults are used.
            settings: AWS Nova Sonic LLM settings. If provided together with
                deprecated top-level parameters, the ``settings`` values take
                precedence.
            system_instruction: System-level instruction for the model.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSNovaSonicLLMService.Settings(system_instruction=...)`` instead.
            tools: Available tools/functions for the model to use.
            session_continuation: Configuration for automatic session continuation.
                When enabled (the default), sessions are seamlessly rotated before
                the AWS time limit (~8 minutes) with no user-perceptible interruption.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="amazon.nova-2-sonic-v1:0",
            system_instruction=None,
            voice="matthew",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            endpointing_sensitivity=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model != "amazon.nova-2-sonic-v1:0":
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id != "matthew":
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if system_instruction is not None:
            self._warn_init_param_moved_to_settings("system_instruction", "system_instruction")
            default_settings.system_instruction = system_instruction

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The `params` parameter is deprecated. "
                    "Use `settings=self.Settings(...)` for inference settings "
                    "(temperature, max_tokens, top_p, endpointing_sensitivity) "
                    "and `audio_config=AudioConfig(...)` for audio configuration "
                    "(sample rates, sample sizes, channel counts).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if not settings:
                default_settings.temperature = params.temperature
                default_settings.max_tokens = params.max_tokens
                default_settings.top_p = params.top_p
                default_settings.endpointing_sensitivity = params.endpointing_sensitivity

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )
        self._secret_access_key = secret_access_key
        self._access_key_id = access_key_id
        self._session_token = session_token
        self._region = region
        self._client: BedrockRuntimeClient | None = None

        # Audio I/O config (hardware settings, not runtime-tunable)
        # Priority: audio_config > params (deprecated) > defaults
        self._audio_config = audio_config or (
            params.audio_config if params is not None else AudioConfig()
        )
        self._tools = tools

        # Validate endpointing_sensitivity parameter
        if (
            self._settings.endpointing_sensitivity
            and not self._is_endpointing_sensitivity_supported()
        ):
            logger.warning(
                f"endpointing_sensitivity is not supported for model '{self._settings.model}' and will be ignored. "
                "This parameter is only supported starting with Nova 2 Sonic (amazon.nova-2-sonic-v1:0)."
            )
            self._settings.endpointing_sensitivity = None

        self._context: LLMContext | None = None
        self._stream: (
            DuplexEventStream[
                InvokeModelWithBidirectionalStreamInput,
                InvokeModelWithBidirectionalStreamOutput,
                InvokeModelWithBidirectionalStreamOperationOutput,
            ]
            | None
        ) = None
        self._receive_task: asyncio.Task | None = None
        self._prompt_name: str | None = None
        self._input_audio_content_name: str | None = None
        self._content_being_received: CurrentContent | None = None
        self._assistant_is_responding = False
        self._ready_to_send_context = False
        self._triggering_assistant_response = False
        self._waiting_for_trigger_transcription = False
        self._disconnecting = False
        self._connected_time: float | None = None
        self._wants_connection = False
        self._user_text_buffer = ""
        self._completed_tool_calls = set()
        self._audio_input_started = False

        # Session continuation helper. The service itself implements the
        # NovaSonicSessionSender protocol (see methods below) so the helper can
        # target either the current or next session without coupling to the
        # service's internal config.
        sc_params = session_continuation or SessionContinuationParams()
        self._sc = SessionContinuationHelper(
            sc_params,
            sender=self,
            create_task=lambda coro: self.create_task(coro),
            cancel_task=lambda task, timeout: self.cancel_task(task, timeout=timeout),
        )
        self._pending_speculative_text: str | None = None

        file_path = files("pipecat.services.aws.nova_sonic").joinpath("ready.wav")
        with wave.open(file_path.open("rb"), "rb") as wav_file:
            self._assistant_response_trigger_audio = wav_file.readframes(wav_file.getnframes())

    #
    # settings
    #

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored but not applied to the active connection.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # TODO: someday we could reconnect here to apply updated settings.
        # Code might look something like the below:
        # await self._disconnect()
        # await self._start_connecting()

        self._warn_unhandled_updated_settings(changed)

        return changed

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

        Cleans up any in-progress assistant response, disconnects from the
        service, and reconnects with the preserved context.
        """
        logger.debug("Resetting conversation")

        # Grab context to carry through disconnect/reconnect
        context = self._context
        if context is None:
            logger.warning(
                "reset_conversation called before an initial context was received; nothing to reset"
            )
            return

        if self._assistant_is_responding:
            self._assistant_is_responding = False
            await self._report_assistant_response_ended()

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

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio_frame(frame)
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

        # Session continuation: let the helper buffer audio during the transition window
        self._sc.on_audio_input(frame.audio)

        # Stop sending audio to the old stream once a handoff is in progress.
        # Audio is still being buffered above and will be replayed to the new
        # session. Matches reference: old session state set to CLOSING stops
        # audio routing before close events are sent.
        if self._sc.handoff_in_progress:
            return

        await self._send_user_audio_event(frame.audio)

    async def _handle_interruption_frame(self):
        pass

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
            self._client = self.create_client()

            # Start the bidirectional stream
            self._stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=assert_given(self._settings.model)
                )
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
        if not self._context:  # should never happen
            return
        # Check for set of completed function calls in the context
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
                        f"{self}: Nova Sonic does not support streamed async "
                        f"tool results; dropping intermediate result for "
                        f"tool_call_id={async_payload.tool_call_id}. Use a "
                        f"non-realtime LLM service if your tool needs to "
                        f"stream intermediate results."
                    )
                    await self.push_error(
                        error_msg="Nova Sonic does not support streamed async tool results.",
                    )
                    continue
                if async_payload.kind == "final":
                    # Deliver via the formal toolResult channel — same path
                    # as a synchronous tool result, just delayed.
                    if send_new_results:
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
            if message.get("role") == "tool" and message.get("content") not in [
                "IN_PROGRESS",
                "CANCELLED",
            ]:
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
        adapter = self.get_llm_adapter()
        llm_connection_params = adapter.get_llm_invocation_params(
            self._context, system_instruction=assert_given(self._settings.system_instruction)
        )

        # Send prompt start event, specifying tools.
        # Tools from context take priority over self._tools.
        tools = (
            llm_connection_params["tools"]
            if llm_connection_params["tools"]
            else (adapter.from_standard_tools(self._tools) or [])
        )
        logger.debug(f"Using tools: {tools}")
        await self._send_prompt_start_event(tools)

        # Send system instruction.
        # The adapter resolves conflicts between init-provided and
        # context-provided system instructions (preferring init-provided).
        system_instruction = llm_connection_params["system_instruction"]
        logger.debug(f"Using system instruction: {system_instruction}")
        if system_instruction:
            await self._send_text_event(text=system_instruction, role=Role.SYSTEM)

        # Send conversation history (except for the last message if it's from the
        # user, which we'll send as interactive after starting audio input)
        messages = llm_connection_params["messages"]
        last_user_message = None
        for i, message in enumerate(messages):
            # logger.debug(f"Seeding conversation history with message: {message}")
            is_last_message = i == len(messages) - 1
            if is_last_message and message.role == Role.USER:
                # Save for sending after audio input starts
                last_user_message = message
            else:
                await self._send_text_event(text=message.text, role=message.role)

        # Start audio input
        await self._send_audio_input_start_event()

        # Now send the last user message as interactive to trigger bot response
        if last_user_message:
            # logger.debug(
            #     f"Sending last user message as interactive to trigger bot response: {last_user_message}")
            await self._send_text_event(
                text=last_user_message.text, role=last_user_message.role, interactive=True
            )

        # Start receiving events (bound to the current stream)
        self._receive_task = self.create_task(self._receive_task_handler(stream=self._stream))

        # Record finished connecting time (must be done before sending assistant response trigger)
        self._connected_time = time.time()

        logger.info("Finished connecting")

        # Notify session continuation helper of connection and start monitoring
        self._sc.set_connected(self._connected_time)
        # Seed the helper's history with initial context messages (these wouldn't be
        # captured via real-time FINAL text events since they pre-date the session)
        for message in llm_connection_params["messages"]:
            self._sc.seed_history(message.role.value, message.text)
        self._sc.start_monitor()

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
            self._prompt_name = None
            self._input_audio_content_name = None
            self._content_being_received = None
            self._assistant_is_responding = False
            self._ready_to_send_context = False
            self._triggering_assistant_response = False
            self._waiting_for_trigger_transcription = False
            self._disconnecting = False
            self._connected_time = None
            self._user_text_buffer = ""
            self._completed_tool_calls = set()
            self._audio_input_started = False
            self._pending_speculative_text = None

            # Stop session continuation monitor and notify of disconnect
            await self._sc.stop_monitor()
            await self._sc.cleanup_next_session()
            self._sc.set_disconnected()

            logger.info("Finished disconnecting")
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)

    def create_client(self) -> BedrockRuntimeClient:
        """Create a new Bedrock runtime client (NovaSonicSessionSender protocol)."""
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

    @property
    def audio_config(self) -> AudioConfig:
        """Return the audio configuration (NovaSonicSessionSender protocol)."""
        return self._audio_config

    def _is_first_generation_sonic_model(self) -> bool:
        # Nova Sonic (the older model) is identified by "amazon.nova-sonic-v1:0"
        return self._settings.model == "amazon.nova-sonic-v1:0"

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
    # These methods operate on the current session. They're thin wrappers over
    # the NovaSonicSessionSender protocol methods (which accept an explicit
    # stream/prompt_name), reusing the same Nova Sonic wire-format serialization
    # for both the current session and next-session setup during a handoff.
    #

    async def _send_session_start_event(self):
        await self._send_client_event(self.build_session_start_json())

    async def _send_prompt_start_event(self, tools: list[Any]):
        if not self._prompt_name:
            return
        await self.send_prompt_start(tools, self._prompt_name, self._stream)

    async def _send_audio_input_start_event(self):
        if not self._prompt_name:
            return
        await self.send_audio_input_start(
            self._prompt_name, self._input_audio_content_name, self._stream
        )
        self._audio_input_started = True

    async def _send_text_event(self, text: str, role: Role, interactive: bool = False):
        """Send a text event to the LLM.

        Args:
            text: The text content to send.
            role: The role associated with the text (e.g., USER, ASSISTANT, SYSTEM).
            interactive: Whether the content is interactive. Defaults to False.
                False: conversation history or system instruction, sent prior to interactive audio
                True: text input sent during (or at the start of) interactive audio
        """
        if not self._stream or not self._prompt_name or not text:
            return
        await self.send_text(text, role.value, self._prompt_name, self._stream, interactive)

    async def _send_user_audio_event(self, audio: bytes):
        if not self._stream or not self._audio_input_started:
            return
        await self.send_audio(
            audio, self._prompt_name, self._input_audio_content_name, self._stream
        )

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

        logger.debug(f"Sending tool result to Nova Sonic for tool_call_id={tool_call_id}")

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
                        "content": json.dumps(result, ensure_ascii=False)
                        if isinstance(result, dict)
                        else result,
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
    # NovaSonicSessionSender protocol implementation
    #
    # These methods expose the Nova Sonic wire protocol to the session
    # continuation helper. Each accepts an explicit ``stream`` / ``prompt_name``
    # so the helper can target either the current session or a pre-created
    # next session during a handoff.
    #

    def build_session_start_json(self) -> str:
        """Build the ``sessionStart`` event JSON.

        Shared between the current and next session setup.
        """
        turn_detection_config = (
            f""",
              "turnDetectionConfiguration": {{
                "endpointingSensitivity": "{self._settings.endpointing_sensitivity}"
              }}"""
            if self._settings.endpointing_sensitivity
            else ""
        )
        return f"""
        {{
          "event": {{
            "sessionStart": {{
              "inferenceConfiguration": {{
                "maxTokens": {self._settings.max_tokens},
                "topP": {self._settings.top_p},
                "temperature": {self._settings.temperature}
              }}{turn_detection_config}
            }}
          }}
        }}
        """

    async def open_stream(self, client):
        """Open a bidirectional stream on the given client."""
        return await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(
                model_id=assert_given(self._settings.model)
            )
        )

    async def send_event(self, event_json: str, stream):
        """Send a raw event JSON to the given stream."""
        if not stream:
            return
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await stream.input_stream.send(event)

    async def send_text(
        self,
        text: str,
        role: str,
        prompt_name: str,
        stream,
        interactive: bool,
    ):
        """Send a text content block (contentStart/textInput/contentEnd) to the given stream."""
        if not text or not stream or not prompt_name:
            return
        content_name = str(uuid.uuid4())
        escaped_text = json.dumps(text)

        content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{prompt_name}",
                    "contentName": "{content_name}",
                    "type": "TEXT",
                    "interactive": {json.dumps(interactive)},
                    "role": "{role}",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(content_start, stream)

        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{prompt_name}",
                    "contentName": "{content_name}",
                    "content": {escaped_text}
                }}
            }}
        }}
        '''
        await self.send_event(text_input, stream)

        content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{prompt_name}",
                    "contentName": "{content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(content_end, stream)

    async def send_audio_input_start(self, prompt_name: str, content_name: str, stream):
        """Send an audio input ``contentStart`` to the given stream."""
        event_json = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{prompt_name}",
                    "contentName": "{content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": {self._audio_config.input_sample_rate},
                        "sampleSizeBits": {self._audio_config.input_sample_size},
                        "channelCount": {self._audio_config.input_channel_count},
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(event_json, stream)

    async def send_audio(self, audio: bytes, prompt_name: str, content_name: str, stream):
        """Send an ``audioInput`` event to the given stream."""
        blob = base64.b64encode(audio)
        event_json = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{prompt_name}",
                    "contentName": "{content_name}",
                    "content": "{blob.decode("utf-8")}"
                }}
            }}
        }}
        '''
        await self.send_event(event_json, stream)

    async def send_prompt_start(self, tools: list, prompt_name: str, stream):
        """Send a ``promptStart`` event to the given stream."""
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
        event_json = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": {self._audio_config.output_sample_rate},
                "sampleSizeBits": {self._audio_config.output_sample_size},
                "channelCount": {self._audio_config.output_channel_count},
                "voiceId": "{self._settings.voice}",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}{tools_config}
            }}
          }}
        }}
        '''
        await self.send_event(event_json, stream)

    def get_setup_params(self) -> tuple[str | None, list]:
        """Return ``(system_instruction, tools)`` for the next session setup."""
        if not self._context:
            return None, []
        adapter = self.get_llm_adapter()
        llm_params = adapter.get_llm_invocation_params(
            self._context, system_instruction=assert_given(self._settings.system_instruction)
        )
        tools = (
            llm_params["tools"]
            if llm_params["tools"]
            else (adapter.from_standard_tools(self._tools) or [])
        )
        return llm_params["system_instruction"], tools

    async def _run_sc_handoff(self):
        """Swap the current session with the pre-created next one."""
        # Snapshot the old session's resources before the helper swaps them out
        old_client = self._client
        old_stream = self._stream
        old_receive_task = self._receive_task
        old_prompt_name = self._prompt_name
        old_input_audio_content_name = self._input_audio_content_name

        next_session = await self._sc.execute_handoff()
        if not next_session:
            return

        # Swap in the new session's stream and names. The helper already sent
        # sessionStart, promptStart, system instruction, conversation history,
        # audioInputStart, and buffered audio to the new stream.
        self._client = next_session.client
        self._stream = next_session.stream
        self._prompt_name = next_session.prompt_name
        self._input_audio_content_name = next_session.input_audio_content_name
        self._connected_time = time.time()
        self._audio_input_started = True

        # Start the main receive loop on the new stream (bound to that stream)
        self._receive_task = self.create_task(self._receive_task_handler(stream=self._stream))

        # Update the helper's connected time so the threshold timer restarts
        self._sc.set_connected(self._connected_time)

        logger.info("Session continuation: swap complete, closing old session in background")

        # Close the old session in the background — do not block the pipeline
        self.create_task(
            self._sc.close_old_session(
                old_client,
                old_stream,
                old_receive_task,
                old_prompt_name,
                old_input_audio_content_name,
            ),
            name="sc_close_old_session",
        )

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
    async def _receive_task_handler(self, stream=None):
        # Bind to the specific stream given at creation time.
        # Do NOT re-read ``self._stream`` in the loop — during a session
        # continuation handoff, ``self._stream`` gets swapped to a new session,
        # and reading from the wrong stream here would cause two receive loops
        # to compete on the same stream (yielding "Invalid input request" from
        # the AWS event stream layer).
        if stream is None:
            stream = self._stream
        try:
            while stream and not self._disconnecting:
                try:
                    output = await stream.await_output()
                    result = await output[1].receive()
                except concurrent.futures.InvalidStateError:
                    break

                # After a session continuation handoff, this receive task
                # is stale — stop processing events so close_old_session
                # can drain the stream without interference.
                if stream is not self._stream:
                    return

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
            if self._disconnecting:
                return
            # If this receive task is for a stale (old) stream that was replaced
            # by a session continuation handoff, don't reset the conversation —
            # the new session is already active on self._stream.
            if stream is not self._stream:
                logger.debug(f"Session continuation: old receive task error (expected): {e}")
                return
            await self.push_error(error_msg=f"Error processing responses: {e}", exception=e)
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
            if content.type == ContentType.TEXT:
                if (
                    content.text_stage == TextStage.SPECULATIVE
                    and not self._assistant_is_responding
                ):
                    self._assistant_is_responding = True
                    await self._report_user_transcription_ended()  # Consider user turn over
                    await self._report_assistant_response_started()
            elif content.type == ContentType.AUDIO:
                # Session continuation: AUDIO contentStart from assistant is the
                # trigger to start buffering user audio and creating the next session
                # (if we're past the threshold).
                await self._sc.on_assistant_audio_started()
        elif content.role == Role.USER:
            # Session continuation: USER contentStart during a forced transition
            # (no assistant response yet) should complete the handoff immediately.
            if self._sc.on_user_content_started():
                self.create_task(self._run_sc_handoff(), name="sc_handoff")

    async def _handle_text_output_event(self, event_json):
        if not self._content_being_received:  # should never happen
            return
        content = self._content_being_received

        text_content = event_json["textOutput"]["content"]

        # Bookkeeping: augment the current content being received with text
        # Assumption: only one text content per content block
        content.text_content = text_content

        # Session continuation: track speculative/final text counts for completion signal
        self._sc.on_text_output(
            content.role.value,
            content.text_stage.value if content.text_stage else None,
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
            sample_rate=self._audio_config.output_sample_rate,
            num_channels=self._audio_config.output_channel_count,
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
                if stop_reason != "INTERRUPTED":
                    if content.text_stage == TextStage.SPECULATIVE:
                        await self._report_llm_text(content.text_content)
                    # Session continuation: ASSISTANT FINAL text — add to history
                    # and check for completion signal (speculative/final counts match)
                    if content.text_stage == TextStage.FINAL:
                        if self._sc.on_content_end_assistant_final_text(content.text_content):
                            self.create_task(self._run_sc_handoff(), name="sc_handoff")
                else:
                    if self._assistant_is_responding:
                        # TEXT INTERRUPTED before audio started means no AUDIO
                        # contentEnd will arrive — end the response here.
                        self._assistant_is_responding = False
                        await self._report_assistant_response_ended()
                    # Session continuation: TEXT INTERRUPTED is a completion
                    # signal regardless of audio state (reference lines 650-654)
                    if self._sc.on_content_end_text_interrupted():
                        self.create_task(self._run_sc_handoff(), name="sc_handoff")
            elif content.type == ContentType.AUDIO:
                # Emit deferred TTSTextFrame after all audio chunks have been sent
                await self._report_tts_text()
                if stop_reason in ("END_TURN", "INTERRUPTED"):
                    # END_TURN: normal completion. INTERRUPTED: user interrupted
                    # mid-audio. Both mean no more audio for this turn.
                    self._assistant_is_responding = False
                    await self._report_assistant_response_ended()
        elif content.role == Role.USER:
            if content.type == ContentType.TEXT:
                if content.text_stage == TextStage.FINAL:
                    # User transcription text added
                    await self._report_user_transcription_text_added(content.text_content)
                    # Session continuation: add to real-time history
                    self._sc.on_content_end_user_final_text(content.text_content)

    async def _handle_completion_end_event(self, _):
        # Session continuation: completionEnd is a fallback completion signal
        if self._sc.on_completion_end():
            self.create_task(self._run_sc_handoff(), name="sc_handoff")

    #
    # assistant response reporting
    #
    # 1. Started
    # 2. Text added
    # 3. Ended
    #

    async def _report_assistant_response_started(self):
        logger.debug("Assistant response started")
        await self.push_frame(LLMFullResponseStartFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) started
        await self.push_frame(TTSStartedFrame())

    async def _report_llm_text(self, text):
        """Push speculative assistant text and defer TTSTextFrame.

        Speculative text arrives before each audio chunk, providing real-time
        text that is synchronized with what the bot is saying. LLMTextFrame and
        AggregatedTextFrame are pushed immediately for real-time text display.
        TTSTextFrame emission is deferred to audio contentEnd so it aligns with
        audio playout timing.
        """
        logger.debug(f"Assistant speculative text: {text}")

        llm_text_frame = LLMTextFrame(text)
        llm_text_frame.append_to_context = False
        await self.push_frame(llm_text_frame)

        aggregated_text_frame = AggregatedTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        aggregated_text_frame.append_to_context = False
        await self.push_frame(aggregated_text_frame)

        self._pending_speculative_text = text

    async def _report_tts_text(self):
        if self._pending_speculative_text:
            tts_text_frame = TTSTextFrame(
                self._pending_speculative_text, aggregated_by=AggregationType.SENTENCE
            )
            tts_text_frame.includes_inter_frame_spaces = True
            await self.push_frame(tts_text_frame)
            self._pending_speculative_text = None

    async def _report_assistant_response_ended(self):
        if not self._context:  # should never happen
            return

        logger.debug("Assistant response ended")

        # Report the end of the assistant response.
        await self.push_frame(LLMFullResponseEndFrame())

        # Report that equivalent of TTS (this is a speech-to-speech model) stopped.
        await self.push_frame(TTSStoppedFrame())

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

        # Nothing to report if no user speech was transcribed (e.g. the prompt
        # was text-only, which is the case on the first user turn when the bot
        # starts the conversation).
        if not self._user_text_buffer:
            return

        logger.debug(f"User transcription ended")

        # Report to the upstream user context aggregator that some new user
        # transcription text is available.

        # HACK: Check if this transcription was triggered by our own
        # assistant response trigger. If so, we need to wrap it with
        # UserStarted/StoppedSpeakingFrames; otherwise the user aggregator
        # would trigger an interruption, which would prevent us from
        # writing the assistant response to context.
        should_wrap_in_user_started_stopped_speaking_frames = (
            self._waiting_for_trigger_transcription
            and self._user_text_buffer.strip().lower() == "ready"
        )

        # Start wrapping the upstream transcription in UserStarted/StoppedSpeakingFrames if needed
        if should_wrap_in_user_started_stopped_speaking_frames:
            logger.debug(
                "Wrapping assistant response trigger transcription with upstream UserStarted/StoppedSpeakingFrames"
            )
            await self.broadcast_frame(UserStartedSpeakingFrame)

        # Send the transcription upstream for the user context aggregator
        frame = TranscriptionFrame(
            text=self._user_text_buffer, user_id="", timestamp=time_now_iso8601()
        )
        await self.push_frame(frame, direction=FrameDirection.UPSTREAM)

        # Finish wrapping the upstream transcription in UserStarted/StoppedSpeakingFrames if needed
        if should_wrap_in_user_started_stopped_speaking_frames:
            await self.broadcast_frame(UserStoppedSpeakingFrame)

        # Clear out the buffered user text
        self._user_text_buffer = ""

        # We're no longer waiting for a trigger transcription
        self._waiting_for_trigger_transcription = False

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
                f"Assistant response trigger not needed for model '{self._settings.model}'; skipping. "
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
                * self._audio_config.input_sample_rate
                * self._audio_config.input_channel_count
                * (self._audio_config.input_sample_size / 8)
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

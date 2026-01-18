#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ultravox Realtime API service implementation.

This module provides real-time conversational AI capabilities using Ultravox's
Realtime API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.
"""

import asyncio
import datetime
import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTextRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserAudioRawFrame,
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
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.utils.time import time_now_iso8601

try:
    from websockets.asyncio import client as websocket_client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Ultravox, you need to `pip install pipecat-ai[ultravox]`.")
    raise Exception(f"Missing module: {e}")


class AgentInputParams(BaseModel):
    """Input parameters for Ultravox Realtime generation using a pre-defined Agent.

    Parameters:
        api_key: Ultravox API key for authentication.
        agent_id: The ID of the Ultravox Realtime agent you'd like to use. Agents
            are pre-configured to handle calls consistently. You can create and edit
            agents in the Ultravox console (https://app.ultravox.ai/agents) or using
            the Ultravox API (https://docs.ultravox.ai/api-reference/agents/agents-post).
        template_context: Context variables to use when instantiating a call with the
            agent. Defaults to an empty dict.
        metadata: Metadata to attach to the call. Default to an empty dict.
        max_duration: The maximum duration of the call. Defaults to None, which will
            use the agent's default maximum duration.
        extra: Extra parameters to include in the agent call creation request. Defaults
            to an empty dict. See the Ultravox API documentation for valid arguments:
            https://docs.ultravox.ai/api-reference/agents/agents-calls-post
    """

    api_key: str
    agent_id: uuid.UUID
    template_context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)
    max_duration: Optional[datetime.timedelta] = Field(
        default=None, ge=datetime.timedelta(seconds=10), le=datetime.timedelta(hours=1)
    )
    extra: Dict[str, Any] = Field(default_factory=dict)


class OneShotInputParams(BaseModel):
    """Input parameters for Ultravox Realtime generation using a one-off call.

    Parameters:
        api_key: Ultravox API key for authentication.
        system_prompt: System prompt to guide the model's behavior. Defaults to None.
        temperature: Sampling temperature for response generation. Defaults to 0.
        model: Model identifier to use. Defaults to "fixie-ai/ultravox".
        voice: Voice identifier for speech generation. Defaults to None.
        metadata: Metadata to attach to the call. Default to an empty dict.
        max_duration: The maximum duration of the call. Defaults to one hour.
        extra: Extra parameters to include in the call creation request. Defaults
            to an empty dict. See the Ultravox API documentation for valid arguments:
            https://docs.ultravox.ai/api-reference/calls/calls-post
    """

    api_key: str
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    model: Optional[str] = None
    voice: Optional[uuid.UUID] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    max_duration: datetime.timedelta = Field(
        default=datetime.timedelta(hours=1),
        ge=datetime.timedelta(seconds=10),
        le=datetime.timedelta(hours=1),
    )
    extra: Dict[str, Any] = Field(default_factory=dict)


class JoinUrlInputParams(BaseModel):
    """Input parameters for joining an existing Ultravox Realtime call via join URL.

    Parameters:
        join_url: The join URL for the existing Ultravox Realtime call.
    """

    join_url: str


class UltravoxRealtimeLLMService(LLMService):
    """Provides access to the Ultravox Realtime API.

    This service enables real-time conversations with Ultravox, supporting both
    text and audio output. It handles voice transcription, streaming audio
    responses, and tool usage.

    Note: Ultravox is an audio-native model, so voice transcriptions are not used
    by the model and may not always align with its understanding of user input.
    """

    def __init__(
        self,
        *,
        params: Union[AgentInputParams, OneShotInputParams, JoinUrlInputParams],
        one_shot_selected_tools: Optional[ToolsSchema] = None,
        **kwargs,
    ):
        """Initialize the Ultravox Realtime LLM service.

        Args:
            api_key: Ultravox API key for authentication.
            params: Configuration parameters for the model.
            one_shot_selected_tools: ToolsSchema for tools to use with this call.
                May only be set with OneShotInputParams.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        super().__init__(**kwargs)
        self._params = params
        if one_shot_selected_tools:
            if not isinstance(self._params, OneShotInputParams):
                logger.warning(
                    "one_shot_selected_tools may only be set when using OneShotInputParams; ignoring."
                )
            else:
                self._selected_tools = one_shot_selected_tools

        self._socket: Optional[websocket_client.ClientConnection] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._disconnecting = False
        self._bot_responding: Literal[None, "text", "voice"] = None
        self._last_user_id: Optional[str] = None

        self._sample_rate = 48000
        self._resampler = create_stream_resampler()

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        """Start the service and establish connection.

        Args:
            frame: The start frame.
        """
        await super().start(frame)

        try:
            match self._params:
                case JoinUrlInputParams():
                    join_url = self._params.join_url
                case AgentInputParams():
                    join_url = await self._start_agent_call(self._params)
                case OneShotInputParams():
                    join_url = await self._start_one_shot_call(self._params)

            logger.info(f"Joining Ultravox Realtime call via URL: {join_url}")
            self._socket = await websocket_client.connect(join_url)
            self._receive_task = self.create_task(self._receive_messages())
        except Exception as e:
            await self.push_error("Failed to connect to Ultravox", e, fatal=True)

    async def _start_agent_call(self, params: AgentInputParams) -> str:
        request_body = {
            "templateContext": params.template_context,
            "metadata": params.metadata,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": self._sample_rate,
                }
            },
        }
        if params.max_duration:
            request_body["maxDuration"] = f"{params.max_duration.total_seconds():3f}s"
        request_body = request_body | params.extra
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.ultravox.ai/api/agents/{params.agent_id}/calls",
                headers={"X-Api-Key": params.api_key},
                json=request_body,
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Ultravox API error {response.status}: {error_text}")
                return (await response.json())["joinUrl"]

    async def _start_one_shot_call(self, params: OneShotInputParams) -> str:
        request_body = {
            "systemPrompt": params.system_prompt,
            "temperature": params.temperature,
            "model": params.model,
            "voice": str(params.voice) if params.voice else None,
            "metadata": params.metadata,
            "maxDuration": f"{params.max_duration.total_seconds():3f}s",
            "selectedTools": self._to_selected_tools(self._selected_tools)
            if self._selected_tools
            else [],
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": self._sample_rate,
                }
            },
        } | params.extra
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.ultravox.ai/api/calls",
                headers={"X-Api-Key": params.api_key},
                json=request_body,
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Ultravox API error {response.status}: {error_text}")
                return (await response.json())["joinUrl"]

    def _to_selected_tools(self, tool: ToolsSchema) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for standard_tool in tool.standard_tools:
            result.append(
                {
                    "temporaryTool": {
                        "modelToolName": standard_tool.name,
                        "description": standard_tool.description,
                        "dynamicParameters": [
                            {
                                "name": k,
                                "location": "PARAMETER_LOCATION_BODY",
                                "schema": v,
                                "required": k in standard_tool.required,
                            }
                            for k, v in standard_tool.properties.items()
                        ],
                        "client": {},
                    }
                }
            )
        return result

    async def stop(self, frame: EndFrame):
        """Stop the service and close connections.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close connections.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _disconnect(self):
        self._disconnecting = True
        if self._socket:
            await self._socket.close()
            self._socket = None
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None

    #
    # frame processing
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames for the Ultravox Realtime service.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            context = (
                frame.context
                if isinstance(frame, LLMContextFrame)
                else LLMContext.from_openai_context(frame.context)
            )
            await self._handle_context(context)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            if "output_medium" in frame.settings:
                await self._update_output_medium(frame.settings.get("output_medium"))
        elif isinstance(frame, InputTextRawFrame):
            await self._send_user_text(frame.text)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # This may or may not align with Ultravox's end of user speech detection,
            # which relies on a more complex endpointing model. In particular it will
            # yield a seemingly very slow TTFB in the case of endpointing false
            # negatives. It will be close in the majority of cases though.
            await self.start_ttfb_metrics()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        # Ultravox handles all context server-side, so the only context we may
        # need to handle here is new function call results.
        for message in reversed(context.messages):
            if message.get("role") != "tool":
                break
            content = message.get("content")
            socket_message = {
                "type": "client_tool_result",
                "invocationId": message.get("tool_call_id"),
                "result": content
                if isinstance(content, str)
                else "".join(t.get("text") for t in content),
            }
            await self._send(socket_message)

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        """Send user audio frame to Ultravox Realtime."""
        if not self._socket:
            return
        self._last_user_id = frame.user_id if isinstance(frame, UserAudioRawFrame) else None
        audio = frame.audio
        if frame.sample_rate != self._sample_rate:
            audio = await self._resampler.resample(audio, frame.sample_rate, self._sample_rate)
        await self._send(audio)

    async def _send_user_text(self, text: str):
        """Send user text via Ultravox Realtime.

        Args:
            text: The text to send as user input.
        """
        if not self._socket:
            return
        await self._send({"type": "user_text_message", "text": text})

    async def _update_output_medium(self, output_medium: str):
        output_medium = output_medium.lower()
        if output_medium == "audio":
            output_medium = "voice"
        if output_medium.lower() not in {"voice", "text"}:
            logger.warning(f"Unsupported Ultravox output medium: {output_medium}")
            return
        await self._send({"type": "set_output_medium", "medium": output_medium})

    async def _send(self, content: Union[bytes, Dict[str, Any]]):
        """Send content via the WebSocket connection.

        Args:
            content: The content to send, either as bytes or a JSON-serializable dict.
        """
        if self._disconnecting or not self._socket:
            return

        try:
            if isinstance(content, bytes):
                await self._socket.send(content)
            else:
                await self._socket.send(json.dumps(content))
        except Exception as e:
            if self._disconnecting or not self._socket:
                return
            await self.push_error("Ultravox websocket send error", e, fatal=True)

    #
    # response handling
    #

    async def _receive_messages(self):
        """Receive messages from the Ultravox Realtime WebSocket."""
        if not self._socket:
            return
        async for message in self._socket:
            try:
                if isinstance(message, bytes):
                    await self._handle_audio(message)
                    continue

                data = json.loads(message)
                match data.get("type"):
                    case "state":
                        if self._bot_responding and data.get("state") != "speaking":
                            await self._handle_response_end()
                    case "client_tool_invocation":
                        await self._handle_tool_invocation(
                            data.get("toolName"), data.get("invocationId"), data.get("parameters")
                        )
                    case "transcript":
                        match data.get("role"):
                            case "user":
                                if not data.get("final"):
                                    logger.warning(
                                        "Unexpected non-final user transcript from Ultravox Realtime; ignoring."
                                    )
                                else:
                                    await self._handle_user_transcript(data.get("text"))
                            case "agent":
                                await self._handle_agent_transcript(
                                    data.get("medium"),
                                    data.get("text"),
                                    data.get("delta"),
                                    data.get("final", False),
                                )
                            case _:
                                logger.debug(
                                    f"Received transcript with unknown role from Ultravox Realtime: {data}"
                                )
                    case _:
                        logger.debug(f"Received unhandled Ultravox message: {data}")
            except Exception as e:
                if self._disconnecting or not self._socket:
                    return
                await self.push_error("Ultravox websocket receive error", e, fatal=True)

    async def _handle_audio(self, audio: bytes):
        """Handle incoming audio bytes from Ultravox Realtime."""
        if not audio:
            return
        if not self._bot_responding:
            await self.stop_ttfb_metrics()
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TTSStartedFrame())
            self._bot_responding = "voice"
        await self.push_frame(TTSAudioRawFrame(audio, self._sample_rate, 1))

    async def _handle_response_end(self):
        if self._bot_responding == "voice":
            await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        self._bot_responding = None

    async def _handle_tool_invocation(
        self, tool_name: str, invocation_id: str, parameters: Dict[str, Any]
    ):
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=tool_name,
                    tool_call_id=invocation_id,
                    arguments=parameters,
                    context=None,
                )
            ]
        )

    async def _handle_user_transcript(self, text: str):
        await self.push_frame(
            TranscriptionFrame(
                user_id=self._last_user_id or "",
                timestamp=time_now_iso8601(),
                result=text,
                text=text,
            ),
            FrameDirection.UPSTREAM,
        )

    async def _handle_agent_transcript(
        self, medium: str, text: Optional[str], delta: Optional[str], final: bool
    ):
        if text or delta:
            frame = LLMTextFrame(text=text or delta)
            frame.skip_tts = medium == "voice"
            await self.push_frame(frame)
        if medium == "text":
            if text:
                await self.stop_ttfb_metrics()
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TTSStartedFrame())
                await self.push_frame(TTSTextFrame(text=text, aggregated_by=AggregationType.WORD))
                self._bot_responding = "text"
            elif final:
                await self.push_frame(LLMFullResponseEndFrame())
                self._bot_responding = None
            elif delta:
                await self.push_frame(TTSTextFrame(text=delta, aggregated_by=AggregationType.WORD))

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> LLMContextAggregatorPair:
        """Create an instance of LLMContextAggregatorPair from an OpenAILLMContext.

        Constructor keyword arguments for both the user and assistant aggregators can be provided.

        NOTE: this method exists only for backward compatibility. New code
        should instead do::

            context = LLMContext(...)
            context_aggregator = LLMContextAggregatorPair(context)

        Args:
            context: The LLM context to use.
            user_params: User aggregator parameters. Defaults to LLMUserAggregatorParams().
            assistant_params: Assistant aggregator parameters. Defaults to LLMAssistantAggregatorParams().

        Returns:
            A pair of user and assistant context aggregators.

        .. deprecated:: 0.0.99
            `create_context_aggregator()` is deprecated and will be removed in a future version.
            Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
            See `OpenAILLMContext` docstring for migration guide.
        """
        # from_openai_context handles deprecation warning
        context = LLMContext.from_openai_context(context)
        assistant_params.expect_stripped_words = False
        return LLMContextAggregatorPair(
            context, user_params=user_params, assistant_params=assistant_params
        )

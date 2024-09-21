#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import base64
import io
import json
import httpx
from dataclasses import dataclass

from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from loguru import logger
from PIL import Image

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMModelUpdateFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    URLImageRawFrame,
    VisionImageRawFrame,
    FunctionCallResultFrame,
    FunctionCallInProgressFrame,
    StartInterruptionFrame
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator

from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import (
    ImageGenService,
    LLMService,
    TTSService
)

try:
    from openai import AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient, BadRequestError, NOT_GIVEN
    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")

ValidVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "shimmer": "shimmer",
}


class OpenAIUnhandledFunctionException(Exception):
    pass


class BaseOpenAILLMService(LLMService):
    """This is the base for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """
    class InputParams(BaseModel):
        frequency_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0)
        presence_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0)
        seed: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        top_p: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
            self,
            *,
            model: str,
            api_key=None,
            base_url=None,
            params: InputParams = InputParams(),
            **kwargs):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)
        self._frequency_penalty = params.frequency_penalty
        self._presence_penalty = params.presence_penalty
        self._seed = params.seed
        self._temperature = params.temperature
        self._top_p = params.top_p
        self._extra = params.extra if isinstance(params.extra, dict) else {}

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000,
                    keepalive_expiry=None)))

    def can_generate_metrics(self) -> bool:
        return True

    async def set_frequency_penalty(self, frequency_penalty: float):
        logger.debug(f"Switching LLM frequency_penalty to: [{frequency_penalty}]")
        self._frequency_penalty = frequency_penalty

    async def set_presence_penalty(self, presence_penalty: float):
        logger.debug(f"Switching LLM presence_penalty to: [{presence_penalty}]")
        self._presence_penalty = presence_penalty

    async def set_seed(self, seed: int):
        logger.debug(f"Switching LLM seed to: [{seed}]")
        self._seed = seed

    async def set_temperature(self, temperature: float):
        logger.debug(f"Switching LLM temperature to: [{temperature}]")
        self._temperature = temperature

    async def set_top_p(self, top_p: float):
        logger.debug(f"Switching LLM top_p to: [{top_p}]")
        self._top_p = top_p

    async def set_extra(self, extra: Dict[str, Any]):
        logger.debug(f"Switching LLM extra to: [{extra}]")
        self._extra = extra

    async def get_chat_completions(
            self,
            context: OpenAILLMContext,
            messages: List[ChatCompletionMessageParam]) -> AsyncStream[ChatCompletionChunk]:

        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "seed": self._seed,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        params.update(self._extra)

        chunks = await self._client.chat.completions.create(**params)
        return chunks

    async def _stream_chat_completions(
            self, context: OpenAILLMContext) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(f"Generating chat: {context.get_messages_json()}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)

        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = (
            await self._stream_chat_completions(context)
        )

        async for chunk in chunk_stream:
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )
                await self.start_llm_usage_metrics(tokens)

            if len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if chunk.choices[0].delta.tool_calls:
                # We're streaming the LLM response to enable the fastest response times.
                # For text, we just yield each chunk as we receive it and count on consumers
                # to do whatever coalescing they need (eg. to pass full sentences to TTS)
                #
                # If the LLM is a function call, we'll do some coalescing here.
                # If the response contains a function name, we'll yield a frame to tell consumers
                # that they can start preparing to call the function with that name.
                # We accumulate all the arguments for the rest of the streamed response, then when
                # the response is done, we package up all the arguments and the function name and
                # yield a frame containing the function name and the arguments.

                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                    await self.call_start_function(context, function_name)
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                await self.push_frame(TextFrame(chunk.choices[0].delta.content))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            if self.has_function(function_name):
                await self._handle_function_call(context, tool_call_id, function_name, arguments)
            else:
                raise OpenAIUnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function.")

    async def _handle_function_call(
            self,
            context,
            tool_call_id,
            function_name,
            arguments
    ):
        arguments = json.loads(arguments)
        await self.call_function(
            context=context,
            tool_call_id=tool_call_id,
            function_name=function_name,
            arguments=arguments
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMModelUpdateFrame):
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self.set_model_name(frame.model)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())


@dataclass
class OpenAIContextAggregatorPair:
    _user: 'OpenAIUserContextAggregator'
    _assistant: 'OpenAIAssistantContextAggregator'

    def user(self) -> 'OpenAIUserContextAggregator':
        return self._user

    def assistant(self) -> 'OpenAIAssistantContextAggregator':
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):

    def __init__(
            self,
            *,
            model: str = "gpt-4o",
            params: BaseOpenAILLMService.InputParams = BaseOpenAILLMService.InputParams(),
            **kwargs):
        super().__init__(model=model, params=params, **kwargs)

    @staticmethod
    def create_context_aggregator(context: OpenAILLMContext) -> OpenAIContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = OpenAIAssistantContextAggregator(user)
        return OpenAIContextAggregatorPair(
            _user=user,
            _assistant=assistant
        )


class OpenAIImageGenService(ImageGenService):

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        model: str = "dall-e-3",
    ):
        super().__init__()
        self.set_model_name(model)
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt,
            model=self.model_name,
            n=1,
            size=self._image_size
        )

        image_url = image.data[0].url

        if not image_url:
            logger.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(image_url, image.tobytes(), image.size, image.format)
            yield frame


class OpenAITTSService(TTSService):
    """This service uses the OpenAI TTS API to generate audio from text.
    The returned audio is PCM encoded at 24kHz. When using the DailyTransport, set the sample rate in the DailyParams accordingly:
    ```
    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24_000,
    )
    ```
    """

    def __init__(
            self,
            *,
            api_key: str | None = None,
            voice: str = "alloy",
            model: Literal["tts-1", "tts-1-hd"] = "tts-1",
            sample_rate: int = 24000,
            **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._voice: ValidVoice = VALID_VOICES.get(voice, "alloy")
        self.set_model_name(model)
        self._sample_rate = sample_rate

        self._client = AsyncOpenAI(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice = VALID_VOICES.get(voice, self._voice)

    async def set_model(self, model: str):
        logger.debug(f"Switching TTS model to: [{model}]")
        self._model = model

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")
        try:
            await self.start_ttfb_metrics()

            async with self._client.audio.speech.with_streaming_response.create(
                    input=text,
                    model=self.model_name,
                    voice=self._voice,
                    response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})")
                    yield ErrorFrame(f"Error getting audio (status: {r.status_code}, error: {error})")
                    return

                await self.start_tts_usage_metrics(text)

                await self.push_frame(TTSStartedFrame())
                async for chunk in r.iter_bytes(8192):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        yield frame
                await self.push_frame(TTSStoppedFrame())
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(context=context)


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: OpenAIUserContextAggregator):
        super().__init__(context=user_context_aggregator._context)
        self._user_context_aggregator = user_context_aggregator
        self._function_call_in_progress = None
        self._function_call_result = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, StartInterruptionFrame):
            self._function_call_in_progress = None
            self._function_call_finished = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = frame
        elif isinstance(frame, FunctionCallResultFrame):
            if self._function_call_in_progress and self._function_call_in_progress.tool_call_id == frame.tool_call_id:
                self._function_call_in_progress = None
                self._function_call_result = frame
                # TODO-CB: Kwin wants us to refactor this out of here but I REFUSE
                await self._push_aggregation()
            else:
                logger.warning(
                    f"FunctionCallResultFrame tool_call_id does not match FunctionCallInProgressFrame tool_call_id")
                self._function_call_in_progress = None
                self._function_call_result = None

    async def _push_aggregation(self):
        if not (self._aggregation or self._function_call_result):
            return

        run_llm = False

        aggregation = self._aggregation
        self._aggregation = ""

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": frame.tool_call_id,
                                "function": {
                                    "name": frame.function_name,
                                    "arguments": json.dumps(frame.arguments)
                                },
                                "type": "function"
                            }
                        ]
                    })
                    self._context.add_message({
                        "role": "tool",
                        "content": json.dumps(frame.result),
                        "tool_call_id": frame.tool_call_id
                    })
                    run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

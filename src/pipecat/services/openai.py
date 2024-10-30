#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import aiohttp
import httpx
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    URLImageRawFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import ImageGenService, LLMService, TTSService

try:
    from openai import (
        NOT_GIVEN,
        AsyncOpenAI,
        AsyncStream,
        BadRequestError,
        DefaultAsyncHttpxClient,
    )
    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
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
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0
        )
        presence_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0
        )
        seed: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignore right now.
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        max_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        max_completion_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str,
        api_key=None,
        base_url=None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "max_completion_tokens": params.max_completion_tokens,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self.set_model_name(model)
        self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                )
            ),
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(f"Generating chat: {context.get_messages_for_logging()}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)

        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        async for chunk in chunk_stream:
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

            if len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if not chunk.choices[0].delta:
                continue

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
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
                    tool_id_list.append(tool_call_id)
                    function_name = ""
                    arguments = ""
                    tool_call_id = ""
                    func_idx += 1
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
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            for index, (function_name, arguments, tool_id) in enumerate(
                zip(functions_list, arguments_list, tool_id_list), start=1
            ):
                if self.has_function(function_name):
                    run_llm = False
                    arguments = json.loads(arguments)
                    await self.call_function(
                        context=context,
                        function_name=function_name,
                        arguments=arguments,
                        tool_call_id=tool_id,
                        run_llm=run_llm,
                    )
                else:
                    raise OpenAIUnhandledFunctionException(
                        f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
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
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
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
    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        params: BaseOpenAILLMService.InputParams = BaseOpenAILLMService.InputParams(),
        **kwargs,
    ):
        super().__init__(model=model, params=params, **kwargs)

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> OpenAIContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = OpenAIAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


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
            prompt=prompt, model=self.model_name, n=1, size=self._image_size
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
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "sample_rate": sample_rate,
        }
        self.set_model_name(model)
        self.set_voice(voice)

        self._client = AsyncOpenAI(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")
        try:
            await self.start_ttfb_metrics()

            async with self._client.audio.speech.with_streaming_response.create(
                input=text or " ",  # Text must contain at least one character
                model=self.model_name,
                voice=VALID_VOICES[self._voice_id],
                response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()
                async for chunk in r.iter_bytes(8192):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                        yield frame
                yield TTSStoppedFrame()
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")


# internal use only -- todo: refactor
@dataclass
class OpenAIImageMessageFrame(Frame):
    user_image_raw_frame: UserImageRawFrame
    text: Optional[str] = None


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(context=context)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # Our parent method has already called push_frame(). So we can't interrupt the
        # flow here and we don't need to call push_frame() ourselves.
        try:
            if isinstance(frame, UserImageRequestFrame):
                # The LLM sends a UserImageRequestFrame upstream. Cache any context provided with
                # that frame so we can use it when we assemble the image message in the assistant
                # context aggregator.
                if frame.context:
                    if isinstance(frame.context, str):
                        self._context._user_image_request_context[frame.user_id] = frame.context
                    else:
                        logger.error(
                            f"Unexpected UserImageRequestFrame context type: {type(frame.context)}"
                        )
                        del self._context._user_image_request_context[frame.user_id]
                else:
                    if frame.user_id in self._context._user_image_request_context:
                        del self._context._user_image_request_context[frame.user_id]
            elif isinstance(frame, UserImageRawFrame):
                # Push a new OpenAIImageMessageFrame with the text context we cached
                # downstream to be handled by our assistant context aggregator. This is
                # necessary so that we add the message to the context in the right order.
                text = self._context._user_image_request_context.get(frame.user_id) or ""
                if text:
                    del self._context._user_image_request_context[frame.user_id]
                frame = OpenAIImageMessageFrame(user_image_raw_frame=frame, text=text)
                await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: OpenAIUserContextAggregator, **kwargs):
        super().__init__(context=user_context_aggregator._context, **kwargs)
        self._user_context_aggregator = user_context_aggregator
        self._function_calls_in_progress = {}
        self._function_call_result = None
        self._pending_image_frame_message = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, StartInterruptionFrame):
            self._function_calls_in_progress.clear()
            self._function_call_finished = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            logger.debug(f"FunctionCallInProgressFrame: {frame}")
            self._function_calls_in_progress[frame.tool_call_id] = frame
        elif isinstance(frame, FunctionCallResultFrame):
            logger.debug(f"FunctionCallResultFrame: {frame}")
            if frame.tool_call_id in self._function_calls_in_progress:
                del self._function_calls_in_progress[frame.tool_call_id]
                self._function_call_result = frame
                # TODO-CB: Kwin wants us to refactor this out of here but I REFUSE
                await self._push_aggregation()
            else:
                logger.warning(
                    "FunctionCallResultFrame tool_call_id does not match any function call in progress"
                )
                self._function_call_result = None
        elif isinstance(frame, OpenAIImageMessageFrame):
            self._pending_image_frame_message = frame
            await self._push_aggregation()

    async def _push_aggregation(self):
        if not (
            self._aggregation or self._function_call_result or self._pending_image_frame_message
        ):
            return

        run_llm = False

        aggregation = self._aggregation
        self._reset()

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message(
                        {
                            "role": "assistant",
                            "content": "",  # content field required for Grok function calling
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
                    # Only run the LLM if there are no more function calls in progress.
                    run_llm = not bool(self._function_calls_in_progress)
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if self._pending_image_frame_message:
                frame = self._pending_image_frame_message
                self._pending_image_frame_message = None
                self._context.add_image_frame_message(
                    format=frame.user_image_raw_frame.format,
                    size=frame.user_image_raw_frame.size,
                    image=frame.user_image_raw_frame.image,
                    text=frame.text,
                )
                run_llm = True

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

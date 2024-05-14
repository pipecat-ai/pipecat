#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import json
import time
import aiohttp

from PIL import Image

from typing import AsyncGenerator, List, Literal

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMMessagesFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    URLImageRawFrame
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService, ImageGenService

from loguru import logger

try:
    from openai import AsyncOpenAI, AsyncStream

    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessageParam,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class BaseOpenAILLMService(LLMService):
    """This is the base for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """

    def __init__(self, model: str, api_key=None, base_url=None):
        super().__init__()
        self._model: str = model
        self.create_client(api_key=api_key, base_url=base_url)

    def create_client(self, api_key=None, base_url=None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        messages: List[ChatCompletionMessageParam] = context.get_messages()
        messages_for_log = json.dumps(messages)
        logger.debug(f"Generating chat: {messages_for_log}")

        start_time = time.time()
        chunks: AsyncStream[ChatCompletionChunk] = (
            await self._client.chat.completions.create(
                model=self._model,
                stream=True,
                messages=messages,
                tools=context.tools,
                tool_choice=context.tool_choice,
            )
        )

        logger.debug(f"OpenAI LLM TTFB: {time.time() - start_time}")

        return chunks

    async def _chat_completions(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)

        logger.debug(f"Generating chat: {messages_for_log}")

        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model, stream=False, messages=messages
        )
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None

    async def _process_context(self, context: OpenAILLMContext):
        function_name = ""
        arguments = ""

        await self.push_frame(LLMResponseStartFrame())

        chunk_stream: AsyncStream[ChatCompletionChunk] = (
            await self._stream_chat_completions(context)
        )

        async for chunk in chunk_stream:
            if len(chunk.choices) == 0:
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
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    # yield LLMFunctionStartFrame(function_name=tool_call.function.name)
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments and
                    # yield a complete LLMFunctionCallFrame after run_llm_async
                    # completes
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                await self.push_frame(TextFrame(chunk.choices[0].delta.content))

        # if we got a function name and arguments, yield the frame with all the info so
        # frame consumers can take action based on the function call.
        # if function_name and arguments:
        #     yield LLMFunctionCallFrame(function_name=function_name, arguments=arguments)

        await self.push_frame(LLMResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)


class OpenAILLMService(BaseOpenAILLMService):

    def __init__(self, model="gpt-4", **kwargs):
        super().__init__(model, **kwargs)


class OpenAIImageGenService(ImageGenService):

    def __init__(
        self,
        *,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        model: str = "dall-e-3",
    ):
        super().__init__()
        self._model = model
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt,
            model=self._model,
            n=1,
            size=self._image_size
        )

        image_url = image.data[0].url

        if not image_url:
            logger.error(f"No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(image_url, image.tobytes(), image.size, image.format)
            yield frame

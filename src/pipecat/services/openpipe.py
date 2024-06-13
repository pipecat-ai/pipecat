from pipecat.services.ai_services import LLMService
from openpipe import AsyncOpenAI as OpenPipeAI
from openpipe import AsyncStream
import os
from loguru import logger
import secrets
import time
import base64
from openai.types.chat import (ChatCompletionMessageParam, ChatCompletionChunk)
from typing import List
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    URLImageRawFrame,
    VisionImageRawFrame
)


class BaseOpenPipeLLMService(LLMService):

    def __init__(
            self,
            model: str,
            c_id=None,
            api_key=None,
            openpipe_api_key=None,
            openpipe_base_url=None,
            prompt=None):
        super().__init__()
        self._model = model
        self._client = self.create_client(
            api_key=api_key,
            openpipe_api_key=openpipe_api_key,
            openpipe_base_url=openpipe_base_url)
        self.c_id = c_id if c_id else secrets.token_urlsafe(16)
        self.prompt = prompt
        logger.debug(f"Client Created: {self._client}")

    def create_client(self, api_key=None, openpipe_api_key=None, openpipe_base_url=None):
        # Set up the OpenPipe client with the provided API keys and base URL
        client = OpenPipeAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            openpipe={
                "api_key": openpipe_api_key or os.environ.get("OPENPIPE_API_KEY"),
                "base_url": openpipe_base_url or "https://app.openpipe.ai/api/v1"
            }
        )
        return client

    async def _stream_chat_completions(self, context):
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

        start_time = time.time()
        # Stream chat completions using the OpenPipe client
        chunks = (
            await self._client.chat.completions.create(
                model=self._model,
                stream=True,
                messages=messages,
                openpipe={
                    "tags": {"conversation_id": self.c_id,
                             "prompt": self.prompt},
                    "log_request": True
                }
            )
        )

        logger.debug(f"OpenPipe LLM TTFB: {time.time() - start_time}")

        return chunks

    async def _process_context(self, context):
        function_name = ""
        arguments = ""

        chunk_stream: AsyncStream[ChatCompletionChunk] = (
            await self._stream_chat_completions(context)
        )

        await self.push_frame(LLMFullResponseStartFrame())

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
                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(chunk.choices[0].delta.content))
                await self.push_frame(LLMResponseEndFrame())

        await self.push_frame(LLMFullResponseEndFrame())

        # if we got a function name and arguments, yield the frame with all the info so
        # frame consumers can take action based on the function call.
        # if function_name and arguments:
        #     yield LLMFunctionCallFrame(function_name=function_name, arguments=arguments)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)


class OpenPipeLLMService(BaseOpenPipeLLMService):

    def __init__(self, model="gpt-4o", cli_id=None, **kwargs):
        super().__init__(model, cli_id, **kwargs)

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SambaNova LLM service implementation using OpenAI-compatible interface."""

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from pipecat.frames.frames import (
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.tracing.service_decorators import traced_llm


class SambaNovaLLMService(OpenAILLMService):  # type: ignore
    """A service for interacting with SambaNova using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to SambaNova's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key: The API key for accessing SambaNova API.
        model: The model identifier to use. Defaults to "Llama-4-Maverick-17B-128E-Instruct".
        base_url: The base URL for SambaNova API. Defaults to "https://api.sambanova.ai/v1".
        **kwargs: Additional keyword arguments passed to OpenAILLMService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "Llama-4-Maverick-17B-128E-Instruct",
        base_url: str = "https://api.sambanova.ai/v1",
        **kwargs: Dict[Any, Any],
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Dict[Any, Any],
    ) -> Any:
        """Create OpenAI-compatible client for SambaNova API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API endpoint. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            Configured OpenAI-compatible client instance.
        """
        logger.debug(f"Creating SambaNova client with API {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> Any:
        """Get chat completions from SambaNova API endpoint.

        Args:
            context: OpenAI LLM context containing tools and configuration.
            messages: List of chat completion message parameters.

        Returns:
            Chat completion response stream from SambaNova API.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "stream_options": {"include_usage": True},
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks

    @traced_llm  # type: ignore
    async def _process_context(self, context: OpenAILLMContext) -> AsyncStream[ChatCompletionChunk]:
        """Process OpenAI LLM context and stream chat completion chunks.

        This method handles the streaming response from SambaNova API, including
        function call processing and text frame generation. It includes special
        handling for SambaNova's API limitations with tool call indexing.

        Args:
            context: OpenAI LLM context containing conversation state and tools.

        Returns:
            Async stream of chat completion chunks.
        """
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

        async for chunk in WatchdogAsyncIterator(chunk_stream, manager=self.task_manager):
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

            if chunk.choices is None or len(chunk.choices) == 0:
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
                    tool_call_id = tool_call.id  # type: ignore
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.content))

            # When gpt-4o-audio / gpt-4o-mini-audio is used for llm or stt+llm
            # we need to get LLMTextFrame for the transcript
            elif hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio.get(
                "transcript"
            ):
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.audio["transcript"]))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                # This allows compatibility until SambaNova API introduces indexing in tool calls.
                if len(arguments) < 1:
                    continue

                arguments = json.loads(arguments)
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)

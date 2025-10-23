#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google LLM service using OpenAI-compatible API format.

This module provides integration with Google's AI LLM models using the OpenAI
API format through Google's Gemini API OpenAI compatibility layer.
"""

import json
import os

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.llm_service import FunctionCallFromLLM

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger

from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class GoogleLLMOpenAIBetaService(OpenAILLMService):
    """Google LLM service using OpenAI-compatible API format.

    This service provides access to Google's AI LLM models (like Gemini) through
    the OpenAI API format. It handles streaming responses, function calls, and
    tool usage while maintaining compatibility with OpenAI's interface.

    Note: This service includes a workaround for a Google API bug where function
    call indices may be incorrectly set to None, resulting in empty function names.

    .. deprecated:: 0.0.82
        GoogleLLMOpenAIBetaService is deprecated and will be removed in a future version.
        Use GoogleLLMService instead for better integration with Google's native API.

    Reference:
        https://ai.google.dev/gemini-api/docs/openai
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.0-flash",
        **kwargs,
    ):
        """Initialize the Google LLM service.

        Args:
            api_key: Google API key for authentication.
            base_url: Base URL for Google's OpenAI-compatible API.
            model: Google model name to use (e.g., "gemini-2.0-flash").
            **kwargs: Additional arguments passed to the parent OpenAILLMService.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "GoogleLLMOpenAIBetaService is deprecated and will be removed in a future version. "
                "Use GoogleLLMService instead for better integration with Google's native API.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[
            ChatCompletionChunk
        ] = await self._stream_chat_completions_specific_context(context)

        async for chunk in chunk_stream:
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
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
                logger.debug(f"Tool call: {chunk.choices[0].delta.tool_calls}")
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
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.content))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            logger.debug(
                f"Function list: {functions_list}, Arguments list: {arguments_list}, Tool ID list: {tool_id_list}"
            )

            function_calls = []
            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                if function_name == "":
                    # TODO: Remove the _process_context method once Google resolves the bug
                    # where the index is incorrectly set to None instead of returning the actual index,
                    # which currently results in an empty function name('').
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

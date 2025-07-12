from typing import List

from loguru import logger
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

class BasetenLLMService(OpenAILLMService):
    """A service for interacting with Baseten's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Baseten's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key: The API key for accessing Baseten's API.
        base_url: The base URL for Baseten API: should look like https://model-{MODEL_ID}.api.baseten.co/environments/production/sync/v1
        model: The model identifier to use.
        **kwargs: Additional keyword arguments passed to OpenAILLMService.
    """

    def __init__(
            self,
            *,
            api_key: str,
            model: str,
            base_url: str,
            **kwargs
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Baseten API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            Configured OpenAI client instance for Baseten API.
        """
        logger.debug(f"Creating Baseten client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
    
    async def get_chat_competions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ):
        """Get chat completions from Baseten API.

        Removes OpenAI-specific parameters not supported by Baseten and
        configures the request with Baseten-compatible settings.

        Args:
            context: The OpenAI LLM context containing tools and settings.
            messages: List of chat completion message parameters.

        Returns:
            Async generator yielding chat completion chunks from Baseten API.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OLLama LLM service implementation for Pipecat AI framework."""

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class OLLamaLLMService(OpenAILLMService):
    """OLLama LLM service that provides local language model capabilities.

    This service extends OpenAILLMService to work with locally hosted OLLama models,
    providing a compatible interface for running large language models locally.
    """

    def __init__(
        self, *, model: str = "llama2", base_url: str = "http://localhost:11434/v1", **kwargs
    ):
        """Initialize OLLama LLM service.

        Args:
            model: The OLLama model to use. Defaults to "llama2".
            base_url: The base URL for the OLLama API endpoint.
                    Defaults to "http://localhost:11434/v1".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(model=model, base_url=base_url, api_key="ollama", **kwargs)

    def create_client(self, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Ollama.

        Args:
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Ollama.
        """
        logger.debug(f"Creating Ollama client with api {base_url}")
        return super().create_client(base_url=base_url, **kwargs)

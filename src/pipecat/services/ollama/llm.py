#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OLLama LLM service implementation for Pipecat AI framework."""

from pipecat.services.openai.llm import OpenAILLMService


class OLLamaLLMService(OpenAILLMService):
    """OLLama LLM service that provides local language model capabilities.

    This service extends OpenAILLMService to work with locally hosted OLLama models,
    providing a compatible interface for running large language models locally.

    Args:
        model: The OLLama model to use. Defaults to "llama2".
        base_url: The base URL for the OLLama API endpoint.
                 Defaults to "http://localhost:11434/v1".
    """

    def __init__(self, *, model: str = "llama2", base_url: str = "http://localhost:11434/v1"):
        super().__init__(model=model, base_url=base_url, api_key="ollama")

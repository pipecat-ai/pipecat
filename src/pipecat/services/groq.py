#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from loguru import logger

from pipecat.services.openai import OpenAILLMService

try:
    # Groq.ai is recommending OpenAI-compatible function calling, so we've switched over
    # to using the OpenAI client library here.
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Groq.ai, you need to `pip install pipecat-ai[openai]`. Also, set `GROQ_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class GroqLLMService(OpenAILLMService):
    """This class implements inference with Groq's Llama 3.1 models"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama3-groq-70b-8192-tool-use-preview",
        **kwargs,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs) -> None:
        logger.debug(f"Creating Groq client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

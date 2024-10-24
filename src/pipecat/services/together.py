#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from loguru import logger

from pipecat.services.openai import OpenAILLMService

try:
    # Together.ai is recommending OpenAI-compatible function calling, so we've switched over
    # to using the OpenAI client library here rather than the Together Python client library.
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Together.ai, you need to `pip install pipecat-ai[together]`. Also, set `TOGETHER_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class TogetherLLMService(OpenAILLMService):
    """This class implements inference with Together's Llama 3.1 models"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        logger.debug(f"Creating Together.ai client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

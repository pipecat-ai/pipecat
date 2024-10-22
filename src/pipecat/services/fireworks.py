#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.services.openai import BaseOpenAILLMService

from loguru import logger

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Fireworks, you need to `pip install pipecat-ai[fireworks]`. Also, set the `FIREWORKS_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class FireworksLLMService(BaseOpenAILLMService):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "accounts/fireworks/models/firefunction-v1",
        base_url: str = "https://api.fireworks.ai/inference/v1",
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

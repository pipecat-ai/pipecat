#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

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

    class InputParams(BaseModel):
        frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
        max_tokens: Optional[int] = Field(default=4096, ge=1)
        presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignore right now.
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)
        seed: Optional[int] = Field(default=None)

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, params=params, **kwargs)
        self.set_model_name(model)
        self._settings = {
            "max_tokens": params.max_tokens,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

    def can_generate_metrics(self) -> bool:
        return True

    def create_client(self, api_key=None, base_url=None, **kwargs):
        logger.debug(f"Creating Together.ai client with api {base_url}")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                )
            ),
        )

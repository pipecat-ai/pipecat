#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, Optional
import httpx
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    LLMUpdateSettingsFrame,
)
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
        self._max_tokens = params.max_tokens
        self._frequency_penalty = params.frequency_penalty
        self._presence_penalty = params.presence_penalty
        self._temperature = params.temperature
        self._top_k = params.top_k
        self._top_p = params.top_p
        self._extra = params.extra if isinstance(params.extra, dict) else {}

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

    async def set_frequency_penalty(self, frequency_penalty: float):
        logger.debug(f"Switching LLM frequency_penalty to: [{frequency_penalty}]")
        self._frequency_penalty = frequency_penalty

    async def set_max_tokens(self, max_tokens: int):
        logger.debug(f"Switching LLM max_tokens to: [{max_tokens}]")
        self._max_tokens = max_tokens

    async def set_presence_penalty(self, presence_penalty: float):
        logger.debug(f"Switching LLM presence_penalty to: [{presence_penalty}]")
        self._presence_penalty = presence_penalty

    async def set_temperature(self, temperature: float):
        logger.debug(f"Switching LLM temperature to: [{temperature}]")
        self._temperature = temperature

    async def set_top_k(self, top_k: float):
        logger.debug(f"Switching LLM top_k to: [{top_k}]")
        self._top_k = top_k

    async def set_top_p(self, top_p: float):
        logger.debug(f"Switching LLM top_p to: [{top_p}]")
        self._top_p = top_p

    async def set_extra(self, extra: Dict[str, Any]):
        logger.debug(f"Switching LLM extra to: [{extra}]")
        self._extra = extra

    async def _update_settings(self, frame: LLMUpdateSettingsFrame):
        if frame.model is not None:
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self.set_model_name(frame.model)
        if frame.frequency_penalty is not None:
            await self.set_frequency_penalty(frame.frequency_penalty)
        if frame.max_tokens is not None:
            await self.set_max_tokens(frame.max_tokens)
        if frame.presence_penalty is not None:
            await self.set_presence_penalty(frame.presence_penalty)
        if frame.temperature is not None:
            await self.set_temperature(frame.temperature)
        if frame.top_k is not None:
            await self.set_top_k(frame.top_k)
        if frame.top_p is not None:
            await self.set_top_p(frame.top_p)
        if frame.extra:
            await self.set_extra(frame.extra)

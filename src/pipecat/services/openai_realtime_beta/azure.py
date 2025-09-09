#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime Beta LLM service implementation."""

import warnings

from loguru import logger

from .openai import OpenAIRealtimeBetaLLMService

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class AzureRealtimeBetaLLMService(OpenAIRealtimeBetaLLMService):
    """Azure OpenAI Realtime Beta LLM service with Azure-specific authentication.

    .. deprecated:: 0.0.84
        `AzureRealtimeBetaLLMService` is deprecated, use `AzureRealtimeLLMService` instead.
        This class will be removed in version 1.0.0.

    Extends the OpenAI Realtime service to work with Azure OpenAI endpoints,
    using Azure's authentication headers and endpoint format. Provides the same
    real-time audio and text communication capabilities as the base OpenAI service.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        **kwargs,
    ):
        """Initialize Azure Realtime Beta LLM service.

        Args:
            api_key: The API key for the Azure OpenAI service.
            base_url: The full Azure WebSocket endpoint URL including api-version and deployment.
                Example: "wss://my-project.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=my-realtime-deployment"
            **kwargs: Additional arguments passed to parent OpenAIRealtimeBetaLLMService.
        """
        super().__init__(base_url=base_url, api_key=api_key, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "AzureRealtimeBetaLLMService is deprecated and will be removed in version 1.0.0. "
                "Use AzureRealtimeLLMService instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.api_key = api_key
        self.base_url = base_url

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return

            logger.info(f"Connecting to {self.base_url}, api key: {self.api_key}")
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={
                    "api-key": self.api_key,
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

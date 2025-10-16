#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime LLM service implementation."""

from loguru import logger

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Azure Realtime, you need to `pip install pipecat-ai[openai]`.")
    raise Exception(f"Missing module: {e}")


class AzureRealtimeLLMService(OpenAIRealtimeLLMService):
    """Azure OpenAI Realtime LLM service with Azure-specific authentication.

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
        """Initialize Azure Realtime LLM service.

        Args:
            api_key: The API key for the Azure OpenAI service.
            base_url: The full Azure WebSocket endpoint URL including api-version and deployment.
                Example: "wss://my-project.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=my-realtime-deployment"
            **kwargs: Additional arguments passed to parent OpenAIRealtimeLLMService.
        """
        super().__init__(base_url=base_url, api_key=api_key, **kwargs)
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

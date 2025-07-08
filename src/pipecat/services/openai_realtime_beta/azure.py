#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime Beta LLM service implementation."""

from loguru import logger

from .openai import OpenAIRealtimeBetaLLMService

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class AzureRealtimeBetaLLMService(OpenAIRealtimeBetaLLMService):
    """Azure OpenAI Realtime Beta LLM service with Azure-specific authentication.

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
        self.api_key = api_key
        self.base_url = base_url

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return

            logger.info(f"Connecting to {self.base_url}, api key: {self.api_key}")
            self._websocket = await websockets.connect(
                uri=self.base_url,
                extra_headers={
                    "api-key": self.api_key,
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

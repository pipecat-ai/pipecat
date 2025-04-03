#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    """Subclass of OpenAI Realtime API Service with adjustments for Azure's wss connection."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        **kwargs,
    ):
        """Constructor takes the same arguments as the parent class, OpenAIRealtimeBetaLLMService.

        Note that the following are required arguments:
            api_key: The API key for the Azure OpenAI service.
            base_url: The base URL for the Azure OpenAI service.

        base_url should be set to the full Azure endpoint URL including the api-version and the deployment name. For example,

        wss://my-project.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=my-realtime-deployment
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

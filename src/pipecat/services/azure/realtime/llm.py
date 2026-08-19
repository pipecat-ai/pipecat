#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime LLM service implementation."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

AzureTokenProvider = Callable[[], Awaitable[str]]
"""Async callable returning a Microsoft Entra ID bearer token.

Matches :func:`azure.identity.aio.get_bearer_token_provider` used with the
``https://ai.azure.com/.default`` scope.
"""


@dataclass
class AzureRealtimeLLMSettings(OpenAIRealtimeLLMService.Settings):
    """Settings for AzureRealtimeLLMService."""

    pass


class AzureRealtimeLLMService(OpenAIRealtimeLLMService):
    """Azure OpenAI Realtime LLM service with Azure-specific authentication.

    Extends the OpenAI Realtime service to work with Azure OpenAI endpoints,
    using Azure's authentication headers and endpoint format. Provides the same
    real-time audio and text communication capabilities as the base OpenAI service.

    Supports both key-based and Microsoft Entra ID authentication.

    Example::

        service = AzureRealtimeLLMService(
            api_key=os.getenv("AZURE_REALTIME_API_KEY"),
            base_url="wss://my-resource.openai.azure.com/openai/v1/realtime",
            settings=AzureRealtimeLLMService.Settings(model="my-deployment"),
        )
    """

    Settings = AzureRealtimeLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        token_provider: AzureTokenProvider | None = None,
        **kwargs,
    ):
        """Initialize Azure Realtime LLM service.

        Args:
            base_url: The Azure Realtime WebSocket endpoint URL. Without a query
                string, the deployment is appended as ``?model=<Settings.model>``,
                so ``Settings.model`` must name the deployment::

                    wss://my-resource.openai.azure.com/openai/v1/realtime

                A URL that already carries a query string is used verbatim, which
                covers the dated API surface::

                    wss://my-resource.openai.azure.com/openai/realtime?api-version=2025-04-01-preview&deployment=my-deployment

            api_key: The API key for the Azure OpenAI service. Required unless
                ``token_provider`` is given.
            token_provider: Async callable supplying a Microsoft Entra ID bearer token,
                used instead of ``api_key`` when given. Build one with
                :func:`azure.identity.aio.get_bearer_token_provider` and the
                ``https://ai.azure.com/.default`` scope.
            **kwargs: Additional arguments passed to parent OpenAIRealtimeLLMService.

        Raises:
            ValueError: If neither ``api_key`` nor ``token_provider`` is given.
        """
        if api_key is None and token_provider is None:
            raise ValueError("Either `api_key` or `token_provider` is required.")

        super().__init__(base_url=base_url, api_key=api_key or "", **kwargs)
        self._token_provider = token_provider

        # A caller-supplied query string already routes to a deployment, so keep the
        # URL as given rather than letting the base class append `?model=`.
        if "?" in base_url:
            self.base_url = base_url

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return

            if self._token_provider:
                headers = {"Authorization": f"Bearer {await self._token_provider()}"}
            else:
                headers = {"api-key": self.api_key}

            logger.info(f"Connecting to {self.base_url}")
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers=headers,
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"initialization error: {e}", exception=e)
            self._websocket = None

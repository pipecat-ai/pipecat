#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime LLM service implementation."""

from dataclasses import dataclass

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


@dataclass
class AzureRealtimeLLMSettings(OpenAIRealtimeLLMService.Settings):
    """Settings for AzureRealtimeLLMService."""

    pass


class AzureRealtimeLLMService(OpenAIRealtimeLLMService):
    """Azure OpenAI Realtime LLM service with Azure-specific authentication.

    Extends the OpenAI Realtime service to work with Azure OpenAI endpoints,
    using Azure's authentication headers and endpoint format. Provides the same
    real-time audio and text communication capabilities as the base OpenAI service.

    Note: Azure's Realtime API does not support the ``output_modalities`` parameter
    in either ``session.update`` or ``response.create`` requests. This class strips
    that field from all outgoing payloads.
    """

    Settings = AzureRealtimeLLMSettings
    _settings: Settings

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
                Example: "wss://my-project.openai.azure.com/openai/realtime?api-version=2025-04-01-preview&deployment=my-realtime-deployment"
            **kwargs: Additional arguments passed to parent OpenAIRealtimeLLMService.
        """
        super().__init__(base_url=base_url, api_key=api_key, **kwargs)
        self.api_key = api_key
        self.base_url = base_url

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event, stripping output_modalities before transmission.

        Azure's Realtime API rejects ``output_modalities`` in both
        ``session.update`` and ``response.create`` requests. This override
        removes the field from those events before forwarding them.
        """
        if isinstance(event, events.SessionUpdateEvent) and event.session.output_modalities:
            event = event.model_copy(
                update={"session": event.session.model_copy(update={"output_modalities": None})}
            )
        elif isinstance(event, events.ResponseCreateEvent) and event.response is not None:
            if event.response.output_modalities is not None:
                event = event.model_copy(
                    update={
                        "response": event.response.model_copy(update={"output_modalities": None})
                    }
                )
        await super().send_client_event(event)

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return

            logger.info(f"Connecting to {self.base_url}")
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={
                    "api-key": self.api_key,
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"initialization error: {e}", exception=e)
            self._websocket = None

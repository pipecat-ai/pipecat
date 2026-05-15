#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime LLM service implementation."""

from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import LLMFullResponseStartFrame
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Azure Realtime, you need to `pip install pipecat-ai[openai]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AzureRealtimeLLMSettings(OpenAIRealtimeLLMService.Settings):
    """Settings for AzureRealtimeLLMService."""

    pass


class AzureRealtimeLLMService(OpenAIRealtimeLLMService):
    """Azure OpenAI Realtime LLM service with Azure-specific authentication.

    Extends the OpenAI Realtime service to work with Azure OpenAI endpoints,
    using Azure's authentication headers and endpoint format. Provides the same
    real-time audio and text communication capabilities as the base OpenAI service.
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

    async def _create_response(self):
        """Override to omit output_modalities from response.create events.

        Azure's Realtime API does not support the ``response.output_modalities``
        parameter in ``response.create`` events and will reject requests that
        include it.  This override sends a plain ``ResponseCreateEvent`` without
        specifying output modalities while keeping all other behaviour from the
        parent class.
        """
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        adapter = self.get_llm_adapter()

        # Configure the LLM for this session if needed
        if self._llm_needs_conversation_setup:
            logger.debug(
                f"Setting up conversation on Azure Realtime LLM service with initial messages: {adapter.get_messages_for_logging(self._context)}"
            )

            # Send initial messages
            llm_invocation_params = adapter.get_llm_invocation_params(self._context)
            messages = llm_invocation_params["messages"]
            for item in messages:
                evt = events.ConversationItemCreateEvent(item=item)
                self._messages_added_manually[evt.item.id] = True
                await self.send_client_event(evt)

            # Send new settings if needed
            await self._send_session_update()

            # We're done configuring the LLM for this session
            self._llm_needs_conversation_setup = False

        logger.debug("Creating response")

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        # Azure does not support response.output_modalities, so send
        # ResponseCreateEvent without specifying it.
        await self.send_client_event(events.ResponseCreateEvent())

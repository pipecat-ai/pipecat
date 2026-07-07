#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh LLM Service implementation using OpenAI-compatible interface."""

from loguru import logger
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.dograh.mps_billing import (
    MPS_BILLING_VERSION_KEY,
    MPS_BILLING_VERSION_V2,
    get_correlation_id,
)
from pipecat.services.openai.base_llm import OpenAILLMInvocationParams, OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService


class DograhLLMService(OpenAILLMService):
    """A unified LLM service using Dograh's API with OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Dograh's unified API endpoint
    while maintaining full compatibility with OpenAI's interface. The actual LLM provider
    (OpenAI, Groq, Google, etc.) is determined by the Dograh backend configuration.
    """

    # The Dograh unified endpoint routes to multiple providers, not all of which
    # accept the "developer" message role. Disable it so the adapter converts
    # "developer" messages to "user" messages before sending.
    supports_developer_role = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://services.dograh.com/api/v1/llm",
        correlation_id: str | None = None,
        settings: OpenAILLMSettings | None = None,
        **kwargs,
    ):
        """Initialize Dograh LLM service.

        Args:
            api_key: The Dograh API key for authentication.
            base_url: The base URL for Dograh API. Defaults to "https://services.dograh.com/api/v1/llm".
            correlation_id: Optional server-generated correlation ID for MPS billing v2.
            settings: LLM settings including model, temperature, etc.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = OpenAILLMSettings(model="default")
        if settings is not None:
            default_settings.apply_update(settings)
        self._base_url = base_url
        self._correlation_id = correlation_id
        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)
        self._start_metadata = None

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Dograh API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance api_key.
            base_url: Base URL for the API. If None, uses instance base_url.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Dograh's API.
        """
        logger.debug(f"Creating Dograh LLM client with base URL: {base_url or self._base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame from the LLM user message."""
        if isinstance(frame, StartFrame):
            self._start_metadata = frame.metadata

        await super().process_frame(frame, direction)

    def _get_correlation_id(self) -> str | None:
        return get_correlation_id(
            explicit_correlation_id=self._correlation_id,
            start_metadata=self._start_metadata,
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for chat completion request with workflow_run_id.

        Overrides the base method to include workflow_run_id from StartFrame metadata
        for tracking usage per correlation ID. Uses OpenAI's metadata field to pass
        additional context.

        Args:
            params_from_context: Parameters from the LLM context

        Returns:
            Dictionary of parameters for the chat completion request
        """
        # Get base parameters from parent class
        params = super().build_chat_completion_params(params_from_context)

        correlation_id = self._get_correlation_id()
        if correlation_id:
            # Initialize metadata dict if not present
            if "metadata" not in params:
                params["metadata"] = {}

            params["metadata"]["correlation_id"] = correlation_id
            params["metadata"][MPS_BILLING_VERSION_KEY] = MPS_BILLING_VERSION_V2

        return params

    async def get_chat_completions(
        self, context: LLMContext
    ) -> AsyncStream[ChatCompletionChunk] | None:
        """Override to handle Dograh-specific quota errors.

        Args:
            context: Context to use for the chat completion.

        Returns:
            The chat completion response, or None if a quota error was handled.

        Raises:
            Pushes a fatal ErrorFrame for quota errors
        """
        try:
            return await super().get_chat_completions(context)
        except Exception as e:
            # Check if this is a quota error (PermissionDeniedError with quota_exceeded)
            error_str = str(e)
            if "quota_exceeded" in error_str and "403" in error_str:
                # Extract the meaningful error message
                error_msg = "Dograh Service quota exceeded"

                # Push a fatal error frame to trigger pipeline shutdown
                await self.push_frame(
                    ErrorFrame(error=error_msg, fatal=True), direction=FrameDirection.UPSTREAM
                )

                # Return from here and do not reraise the error. Let
                # ErrorFrae terminate the Pipeline
                return

            # Re-raise the exception for normal error handling
            raise

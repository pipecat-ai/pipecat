#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI service implementation for the Pipecat AI framework."""

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

AzureTokenProvider = Callable[[], Awaitable[str]]
"""Async callable returning a Microsoft Entra ID bearer token.

Matches :func:`azure.identity.aio.get_bearer_token_provider` used with the
``https://ai.azure.com/.default`` scope.
"""

V1_ENDPOINT_PATH = "/openai/v1"
"""Endpoint path suffix identifying Azure's v1 API surface."""

DATED_API_VERSION = "2025-04-01-preview"
"""Dated API version used for endpoints outside the v1 API surface.

Azure stopped issuing dated versions after this one; the v1 surface tracks new
features in its place.
"""


@dataclass
class AzureLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for AzureLLMService."""

    pass


class AzureLLMService(OpenAILLMService):
    """A service for interacting with Azure OpenAI using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Azure's OpenAI endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    The shape of ``endpoint`` selects the API surface. An endpoint ending in
    ``/openai/v1`` uses Azure's v1 API, which tracks new features without a dated
    ``api_version``; any other endpoint routes through a dated version of the API.
    Both key-based and Microsoft Entra ID authentication work on either surface.

    Example::

        service = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint="https://my-resource.openai.azure.com/openai/v1",
            settings=AzureLLMService.Settings(model="my-deployment"),
        )
    """

    Settings = AzureLLMSettings

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str | None = None,
        token_provider: AzureTokenProvider | None = None,
        model: str | None = None,
        api_version: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Azure LLM service.

        Args:
            endpoint: The Azure endpoint URL. Ending it in ``/openai/v1`` selects the
                v1 API surface, where ``api_version`` does not apply.
            api_key: The API key for accessing Azure OpenAI. Required unless
                ``token_provider`` is given.
            token_provider: Async callable supplying a Microsoft Entra ID bearer token,
                used instead of ``api_key`` when given. Build one with
                :func:`azure.identity.aio.get_bearer_token_provider` and the
                ``https://ai.azure.com/.default`` scope.
            model: The model identifier to use. Defaults to "gpt-4.1".

                .. deprecated:: 0.0.105
                    Use ``settings=AzureLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            api_version: Azure API version applied to endpoints outside the v1 API
                surface. Defaults to :data:`DATED_API_VERSION`.

                .. deprecated:: 1.8.0
                    Use an ``endpoint`` ending in ``/openai/v1`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.

        Raises:
            ValueError: If neither ``api_key`` nor ``token_provider`` is given.
        """
        if api_key is None and token_provider is None:
            raise ValueError("Either `api_key` or `token_provider` is required.")

        if api_version is not None:
            warnings.warn(
                "`api_version` is deprecated since 1.8.0 and will be removed in 2.0.0. "
                "Use an `endpoint` ending in `/openai/v1` instead. Azure issued no dated "
                "version after 2025-04-01-preview, and new features reach only the v1 "
                "API surface.",
                DeprecationWarning,
                stacklevel=2,
            )

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="gpt-4.1")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version or DATED_API_VERSION
        self._token_provider = token_provider
        self._use_v1_api = endpoint.rstrip("/").endswith(V1_ENDPOINT_PATH)
        super().__init__(api_key=api_key, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Azure OpenAI endpoint.

        Args:
            api_key: API key for authentication. Uses instance key if None.
            base_url: Base URL for the client. Ignored for Azure implementation.
            **kwargs: Additional keyword arguments. Ignored for Azure implementation.

        Returns:
            Configured client for the API surface the endpoint selects.
        """
        if self._use_v1_api:
            logger.debug(f"Creating Azure OpenAI v1 client with endpoint {self._endpoint}")
            # The v1 surface is reached with the plain OpenAI client: AsyncAzureOpenAI
            # appends its own `/openai` path segment, which the endpoint already carries.
            return AsyncOpenAI(
                api_key=self._token_provider or api_key,
                base_url=self._endpoint.rstrip("/") + "/",
            )

        logger.debug(f"Creating Azure OpenAI client with endpoint {self._endpoint}")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
            azure_ad_token_provider=self._token_provider,
        )

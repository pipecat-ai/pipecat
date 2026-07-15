#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI service implementation for the Pipecat AI framework."""

from dataclasses import dataclass

from loguru import logger
from openai import AsyncAzureOpenAI

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import is_given


@dataclass
class AzureLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for AzureLLMService.

    Note:
        On Azure OpenAI, the ``model`` field is used as the **deployment name**
        (the URL segment ``/openai/deployments/<name>/...``), not the model
        family. Deployment names are chosen by the user when they deploy a
        model on their Azure OpenAI resource, so there is no universally
        correct default — AzureLLMService requires this to be set explicitly.
    """

    pass


class AzureLLMService(OpenAILLMService):
    """A service for interacting with Azure OpenAI using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Azure's OpenAI endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = AzureLLMSettings

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model: str | None = None,
        api_version: str = "2024-09-01-preview",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Azure LLM service.

        Args:
            api_key: The API key for accessing Azure OpenAI.
            endpoint: The Azure endpoint URL.
            model: The Azure OpenAI **deployment name** to use — i.e. the name
                you gave the deployment on your Azure resource, which becomes
                the ``/deployments/<name>/`` segment in request URLs. Required;
                no default because deployment names are user-specific.

                .. deprecated:: 0.0.105
                    Use ``settings=AzureLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            api_version: Azure API version. Defaults to "2024-09-01-preview".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.

        Raises:
            ValueError: If no deployment name is provided via either the
                deprecated ``model=`` init arg or ``settings.model``.
        """
        # 1. Initialize default_settings — no hardcoded model default because
        # on Azure `model` is the deployment name, which is user-specific.
        default_settings = self.Settings()

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Fail loudly if no deployment name was supplied. Otherwise the
        # AsyncAzureOpenAI client would build a request URL with no (or a
        # placeholder) deployment segment and Azure would 404 with
        # `DeploymentNotFound` at first inference — much harder to diagnose.
        if not is_given(default_settings.model) or default_settings.model is None:
            raise ValueError(
                "AzureLLMService requires an explicit deployment name via "
                "`settings=AzureLLMService.Settings(model='<your-deployment-name>')`. "
                "On Azure OpenAI the `model` field is the *deployment name* you "
                "chose when deploying a model on your resource, not the model "
                "family — a resource with a gpt-4.1 deployment could have named "
                "it anything (e.g. 'chatgpt', 'my-deploy', 'prod-4-1')."
            )

        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Azure OpenAI endpoint.

        Args:
            api_key: API key for authentication. Uses instance key if None.
            base_url: Base URL for the client. Ignored for Azure implementation.
            **kwargs: Additional keyword arguments. Ignored for Azure implementation.

        Returns:
            AsyncAzureOpenAI: Configured Azure OpenAI client instance.
        """
        logger.debug(f"Creating Azure OpenAI client with endpoint {self._endpoint}")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )

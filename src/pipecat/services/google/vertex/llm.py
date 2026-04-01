#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Vertex AI LLM service implementation.

This module provides integration with Google's AI models via Vertex AI,
extending the GoogleLLMService with Vertex AI authentication.
"""

import json
import os
from dataclasses import dataclass

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import Optional

from loguru import logger

from pipecat.services.google.llm import GoogleLLMService

try:
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request
    from google.genai import Client
    from google.genai.types import HttpOptions
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class GoogleVertexLLMSettings(GoogleLLMService.Settings):
    """Settings for GoogleVertexLLMService."""

    pass


class GoogleVertexLLMService(GoogleLLMService):
    """Google Vertex AI LLM service extending GoogleLLMService.

    Provides access to Google's AI models via Vertex AI while using the same
    Google AI client and message format as GoogleLLMService. Handles authentication
    using Google service account credentials and configures the client for
    Vertex AI endpoints.

    Reference:
        https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
    """

    Settings = GoogleVertexLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        model: Optional[str] = None,
        location: str = "us-east4",
        project_id: str,
        params: Optional[GoogleLLMService.InputParams] = None,
        settings: Optional[Settings] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[list] = None,
        tool_config: Optional[dict] = None,
        http_options: Optional[HttpOptions] = None,
        **kwargs,
    ):
        """Initializes the VertexLLMService.

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.
            model: Model identifier (e.g., "gemini-2.5-flash").

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleVertexLLMService.Settings(model=...)`` instead.

            location: GCP region for Vertex AI endpoint. Defaults to "us-east4".
            project_id: Google Cloud project ID.
            params: Input parameters for the model.

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleVertexLLMService.Settings(...)`` instead.

            settings: Runtime-updatable settings for this service.  When both
                deprecated parameters and *settings* are provided, *settings*
                values take precedence.
            system_instruction: System instruction/prompt for the model.

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleVertexLLMService.Settings(system_instruction=...)`` instead.
            tools: List of available tools/functions.
            tool_config: Configuration for tool usage.
            http_options: HTTP options for the client.
            **kwargs: Additional arguments passed to GoogleLLMService.
        """
        # Check if user incorrectly passed api_key, which is used by parent
        # class but not here.
        if "api_key" in kwargs:
            logger.error(
                "GoogleVertexLLMService does not accept 'api_key' parameter. "
                "Use 'credentials' or 'credentials_path' instead for Vertex AI authentication."
            )
            raise ValueError(
                "Invalid parameter 'api_key'. Use 'credentials' or 'credentials_path' for Vertex AI authentication."
            )

        # These need to be set before calling super().__init__() because
        # super().__init__() invokes _create_client(), which needs these.
        self._credentials = self._get_credentials(credentials, credentials_path)
        self._project_id = project_id
        self._location = location

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gemini-2.5-flash",
            system_instruction=None,
            max_tokens=4096,
            temperature=None,
            top_k=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            thinking=None,
            extra={},
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if system_instruction is not None:
            self._warn_init_param_moved_to_settings("system_instruction", "system_instruction")
            default_settings.system_instruction = system_instruction

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.max_tokens = params.max_tokens
                default_settings.temperature = params.temperature
                default_settings.top_k = params.top_k
                default_settings.top_p = params.top_p
                default_settings.thinking = params.thinking
                if isinstance(params.extra, dict):
                    default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Call parent constructor with dummy api_key
        # (api_key is required by parent class, but not actually used with Vertex)
        super().__init__(
            api_key="dummy",
            settings=default_settings,
            tools=tools,
            tool_config=tool_config,
            http_options=http_options,
            **kwargs,
        )

    def create_client(self):
        """Create the Gemini client instance configured for Vertex AI."""
        self._client = Client(
            vertexai=True,
            credentials=self._credentials,
            project=self._project_id,
            location=self._location,
            http_options=self._http_options,
        )

    @staticmethod
    def _get_credentials(credentials: Optional[str], credentials_path: Optional[str]):
        """Retrieve Credentials using Google service account credentials.

        Supports multiple authentication methods:
        1. Direct JSON credentials string
        2. Path to service account JSON file
        3. Default application credentials (ADC)

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.

        Returns:
            Google credentials object for API authentication.

        Raises:
            ValueError: If no valid credentials are provided or found.
        """
        creds: Optional[service_account.Credentials] = None

        if credentials:
            # Parse and load credentials from JSON string
            creds = service_account.Credentials.from_service_account_info(
                json.loads(credentials),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif credentials_path:
            # Load credentials from JSON file
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            try:
                creds, project_id = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except GoogleAuthError:
                pass

        if not creds:
            raise ValueError("No valid credentials provided.")

        creds.refresh(Request())  # Ensure token is up-to-date, lifetime is 1 hour.

        return creds

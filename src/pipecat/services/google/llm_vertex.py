#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Vertex AI LLM service implementation.

This module provides integration with Google's AI models via Vertex AI while
maintaining OpenAI API compatibility through Google's OpenAI-compatible endpoint.
"""

import json
import os

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import Optional

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService

try:
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class GoogleVertexLLMService(OpenAILLMService):
    """Google Vertex AI LLM service with OpenAI API compatibility.

    Provides access to Google's AI models via Vertex AI while maintaining
    OpenAI API compatibility. Handles authentication using Google service
    account credentials and constructs appropriate endpoint URLs for
    different GCP regions and projects.

    Reference:
        https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library
    """

    class InputParams(OpenAILLMService.InputParams):
        """Input parameters specific to Vertex AI.

        Parameters:
            location: GCP region for Vertex AI endpoint (e.g., "us-east4").
            project_id: Google Cloud project ID.
        """

        # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
        location: str = "us-east4"
        project_id: str

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        model: str = "google/gemini-2.0-flash-001",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initializes the VertexLLMService.

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.
            model: Model identifier (e.g., "google/gemini-2.0-flash-001").
            params: Vertex AI input parameters including location and project.
            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        params = params or OpenAILLMService.InputParams()
        base_url = self._get_base_url(params)
        self._api_key = self._get_api_token(credentials, credentials_path)

        super().__init__(
            api_key=self._api_key, base_url=base_url, model=model, params=params, **kwargs
        )

    @staticmethod
    def _get_base_url(params: InputParams) -> str:
        """Construct the base URL for Vertex AI API."""
        return (
            f"https://{params.location}-aiplatform.googleapis.com/v1/"
            f"projects/{params.project_id}/locations/{params.location}/endpoints/openapi"
        )

    @staticmethod
    def _get_api_token(credentials: Optional[str], credentials_path: Optional[str]) -> str:
        """Retrieve an authentication token using Google service account credentials.

        Supports multiple authentication methods:
        1. Direct JSON credentials string
        2. Path to service account JSON file
        3. Default application credentials (ADC)

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.

        Returns:
            OAuth token for API authentication.

        Raises:
            ValueError: If no valid credentials are provided or found.
        """
        creds: Optional[service_account.Credentials] = None

        if credentials:
            # Parse and load credentials from JSON string
            creds = service_account.Credentials.from_service_account_info(
                json.loads(credentials), scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        elif credentials_path:
            # Load credentials from JSON file
            creds = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
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

        return creds.token

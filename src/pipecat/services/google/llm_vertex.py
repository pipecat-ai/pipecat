#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import os

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import Optional

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService

try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class GoogleVertexLLMService(OpenAILLMService):
    """Implements inference with Google's AI models via Vertex AI while
    maintaining OpenAI API compatibility.

    Reference:
    https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library

    """

    class InputParams(OpenAILLMService.InputParams):
        """Input parameters specific to Vertex AI."""

        # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
        location: str = "us-east4"
        project_id: str

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        model: str = "google/gemini-2.0-flash-001",
        params: InputParams = OpenAILLMService.InputParams(),
        **kwargs,
    ):
        """Initializes the VertexLLMService.

        Args:
            credentials (Optional[str]): JSON string of service account credentials.
            credentials_path (Optional[str]): Path to the service account JSON file.
            model (str): Model identifier. Defaults to "google/gemini-2.0-flash-001".
            params (InputParams): Vertex AI input parameters.
            **kwargs: Additional arguments for OpenAILLMService.
        """
        base_url = self._get_base_url(params)
        self._api_key = self._get_api_token(credentials, credentials_path)

        super().__init__(api_key=self._api_key, base_url=base_url, model=model, **kwargs)

    @staticmethod
    def _get_base_url(params: InputParams) -> str:
        """Constructs the base URL for Vertex AI API."""
        return (
            f"https://{params.location}-aiplatform.googleapis.com/v1/"
            f"projects/{params.project_id}/locations/{params.location}/endpoints/openapi"
        )

    @staticmethod
    def _get_api_token(credentials: Optional[str], credentials_path: Optional[str]) -> str:
        """Retrieves an authentication token using Google service account credentials.

        Args:
            credentials (Optional[str]): JSON string of service account credentials.
            credentials_path (Optional[str]): Path to the service account JSON file.

        Returns:
            str: OAuth token for API authentication.
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

        if not creds:
            raise ValueError("No valid credentials provided.")

        creds.refresh(Request())  # Ensure token is up-to-date, lifetime is 1 hour.

        return creds.token

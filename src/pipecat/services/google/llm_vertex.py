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

        .. deprecated:: 0.0.90
            Use `OpenAILLMService.InputParams` instead and provide `location` and
            `project_id` as direct arguments to `GoogleVertexLLMService.__init__()`.

        Parameters:
            location: GCP region for Vertex AI endpoint (e.g., "us-east4").
            project_id: Google Cloud project ID.
        """

        # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
        location: str = "us-east4"
        project_id: str

        def __init__(self, **kwargs):
            """Initializes the InputParams."""
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.warn(
                    "GoogleVertexLLMService.InputParams is deprecated. "
                    "Please use OpenAILLMService.InputParams instead and provide "
                    "'location' and 'project_id' as direct arguments to __init__.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            super().__init__(**kwargs)

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        model: str = "google/gemini-2.0-flash-001",
        location: Optional[str] = None,
        project_id: Optional[str] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initializes the VertexLLMService.

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.
            model: Model identifier (e.g., "google/gemini-2.0-flash-001").
            location: GCP region for Vertex AI endpoint (e.g., "us-east4").
            project_id: Google Cloud project ID.
            params: Vertex AI input parameters including location and project.

                .. deprecated:: 0.0.90
                    Use `OpenAILLMService.InputParams` instead and provide `location`
                    and `project_id` as direct arguments to `__init__()`.

            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        # Handle deprecated InputParams
        if params is not None and isinstance(params, GoogleVertexLLMService.InputParams):
            # Extract location and project_id from params if not provided directly
            if project_id is None:
                project_id = params.project_id
            if location is None:
                location = params.location
            # Convert to base InputParams
            params = OpenAILLMService.InputParams(
                **params.model_dump(exclude={"location", "project_id"}, exclude_unset=True)
            )
        else:
            params = params or OpenAILLMService.InputParams()

        # Validate parameters
        # NOTE: once we remove deprecated InputParams, we can update __init__()
        #       signature with the following:
        #       - location: str = "us-east4",
        #       - project_id: str,
        #       For now, we need them as-is to maintain backward compatibility.
        if project_id is None:
            raise ValueError("project_id is required")
        if location is None:
            location = "us-east4"  # Default location if not provided

        base_url = self._get_base_url(location, project_id)
        self._api_key = self._get_api_token(credentials, credentials_path)

        super().__init__(
            api_key=self._api_key,
            base_url=base_url,
            model=model,
            params=params,
            **kwargs,
        )

    @staticmethod
    def _get_base_url(location: str, project_id: str) -> str:
        """Construct the base URL for Vertex AI API."""
        # Determine the correct API host based on location
        if location == "global":
            api_host = "aiplatform.googleapis.com"
        else:
            api_host = f"{location}-aiplatform.googleapis.com"
        return f"https://{api_host}/v1/projects/{project_id}/locations/{location}/endpoints/openapi"

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

        return creds.token

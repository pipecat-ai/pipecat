#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service for accessing Gemini Live via Google Vertex AI.

This module provides integration with Google's Gemini Live model via
Vertex AI, supporting both text and audio modalities with voice transcription,
streaming responses, and tool usage.
"""

import json
from typing import List, Optional, Union

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    HttpOptions,
    InputParams,
)

try:
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request
    from google.genai import Client
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google Vertex AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


class GeminiLiveVertexLLMService(GeminiLiveLLMService):
    """Provides access to Google's Gemini Live model via Vertex AI.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage.
    """

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: str,
        project_id: str,
        model="google/gemini-2.0-flash-live-preview-04-09",
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: Optional[str] = None,
        tools: Optional[Union[List[dict], ToolsSchema]] = None,
        params: Optional[InputParams] = None,
        inference_on_context_initialization: bool = True,
        file_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/files",
        http_options: Optional[HttpOptions] = None,
        **kwargs,
    ):
        """Initialize the service for accessing Gemini Live via Google Vertex AI.

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.
            location: GCP region for Vertex AI endpoint (e.g., "us-east4").
            project_id: Google Cloud project ID.
            model: Model identifier to use. Defaults to "models/gemini-2.0-flash-live-preview-04-09".
            voice_id: TTS voice identifier. Defaults to "Charon".
            start_audio_paused: Whether to start with audio input paused. Defaults to False.
            start_video_paused: Whether to start with video input paused. Defaults to False.
            system_instruction: System prompt for the model. Defaults to None.
            tools: Tools/functions available to the model. Defaults to None.
            params: Configuration parameters for the model along with Vertex AI
                location and project ID.
            inference_on_context_initialization: Whether to generate a response when context
                is first set. Defaults to True.
            file_api_base_url: Base URL for the Gemini File API. Defaults to the official endpoint.
            http_options: HTTP options for the client.
            **kwargs: Additional arguments passed to parent GeminiLiveLLMService.
        """
        # Check if user incorrectly passed api_key, which is used by parent
        # class but not here.
        if "api_key" in kwargs:
            logger.error(
                "GeminiLiveVertexLLMService does not accept 'api_key' parameter. "
                "Use 'credentials' or 'credentials_path' instead for Vertex AI authentication."
            )
            raise ValueError(
                "Invalid parameter 'api_key'. Use 'credentials' or 'credentials_path' for Vertex AI authentication."
            )

        # These need to be set before calling super().__init__() because
        # super().__init__() invokes create_client(), which needs these.
        self._credentials = self._get_credentials(credentials, credentials_path)
        self._project_id = project_id
        self._location = location

        # Call parent constructor with the obtained API key
        super().__init__(
            # api_key is required by parent class, but actually not used with
            # Vertex
            api_key="dummy",
            model=model,
            voice_id=voice_id,
            start_audio_paused=start_audio_paused,
            start_video_paused=start_video_paused,
            system_instruction=system_instruction,
            tools=tools,
            params=params,
            inference_on_context_initialization=inference_on_context_initialization,
            file_api_base_url=file_api_base_url,
            http_options=http_options,
            **kwargs,
        )

    def create_client(self):
        """Create the Gemini client instance."""
        self._client = Client(
            vertexai=True,
            credentials=self._credentials,
            project=self._project_id,
            location=self._location,
        )

    @property
    def file_api(self):
        """Gemini File API is not supported with Vertex AI."""
        raise NotImplementedError(
            "When using Vertex AI, the recommended approach is to use Google Cloud Storage for file handling. The Gemini File API is not directly supported in this context."
        )

    @staticmethod
    def _get_credentials(credentials: Optional[str], credentials_path: Optional[str]) -> str:
        """Retrieve Credentials using Google service account credentials JSON.

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

        return creds

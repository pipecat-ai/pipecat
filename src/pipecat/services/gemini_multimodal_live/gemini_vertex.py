#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini Multimodal Live API service on Vertex AI implementation.

This module provides real-time conversational AI capabilities using Google's
Gemini Multimodal Live API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.
"""

import inspect
import json
import os
import re
from typing import Optional

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    GeminiMultimodalModalities,
)
from pipecat.services.gemini_multimodal_live.gemini import (
    InputParams as GeminiInputParams,
)
from pipecat.services.openai.llm import OpenAILLMService

try:
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from openai._types import NOT_GIVEN, NotGiven

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class GeminiMultimodalLiveVertexLLMService(GeminiMultimodalLiveLLMService):
    """Provides access to Google's Gemini Multimodal Live API on Vertex AI.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage. It does this for Vertex AI compatible models.
    """

    class VertexInputParams(OpenAILLMService.InputParams, GeminiInputParams):
        """Input parameters specific to Vertex AI.

        Parameters:
            location: GCP region for Vertex AI endpoint (e.g., "us-east4" or "us-central1").
            project_id: Google Cloud project ID.
        """

        # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
        location: str = "us-central1"
        project_id: str

    def __init__(
        self,
        *,
        api_key: str = "ignore",
        credentials: str,
        credentials_path: Optional[str] = None,
        model="gemini-2.0-flash-exp",  # note: no `models/` prefix
        params: Optional[VertexInputParams] = None,
        **kwargs,
    ):
        """Initialize the Gemini Multimodal Live LLM service.

        See `GeminiMultimodalLiveLLMService` for more arguments that can be passed via kwargs.

        Args:
            api_key: Ignored; use credentials to authenticate.
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.
            model: Model identifier to use. Defaults to "models/gemini-2.0-flash-live-001".
            params: Configuration parameters for the model. Defaults to VertexInputParams().
            **kwargs: Additional arguments passed to parent LLMService.
        """
        super().__init__(api_key=api_key, params=params, **kwargs)

        params = params or self.VertexInputParams()
        self._api_key = self._get_api_token(credentials, credentials_path)
        self._base_url = self._get_base_url(params)
        self._model_path = self._get_model_path(model, params)

    def _get_model_path(self, model: str, params: VertexInputParams) -> str:
        """Construct base path for Vertex AI model."""
        return f"projects/{params.project_id}/locations/{params.location}/publishers/google/models/{model}"

    @staticmethod
    def _get_base_url(params: VertexInputParams) -> str:
        """Construct base URL for Vertex AI API."""
        return f"{params.location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

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

    def get_model_name_or_path(self):
        """Return model path for use on Vertex AI."""
        return self._model_path

    async def websocket_connect_with_auth(self):
        """Connect to websocket using access token."""
        logger.info(f"Connecting to wss://{self._base_url} using access token")
        uri = f"wss://{self._base_url}?access_token={self._api_key}"
        self._websocket = await websocket_connect(uri=uri)

    async def _ws_send(self, message):
        """Send a message to the WebSocket connection."""
        # logger.debug(f"Sending message to websocket: {message}")
        try:
            if self._websocket:
                try:
                    payload = json.dumps(message)
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize message to JSON: {e} — message: {repr(message)}. \n\n trying again"
                    )

                    # Vertex rejects `NotGiven` & NOT_GIVEN types
                    def clean_not_given(obj):
                        if isinstance(obj, dict):
                            return {
                                k: clean_not_given(v)
                                for k, v in obj.items()
                                if "NotGiven" not in str(type(v))
                            }
                        elif isinstance(obj, list):
                            return [
                                clean_not_given(item)
                                for item in obj
                                if "NotGiven" not in str(type(item))
                            ]
                        elif "NotGiven" in str(type(obj)):
                            return None
                        elif "NOT_GIVEN" in str(type(obj)):
                            return None
                        return obj

                    filtered_message = clean_not_given(message)
                    payload = json.dumps(filtered_message)

                    # raise

                await self._websocket.send(payload)
        except Exception as e:
            if self._disconnecting:
                return
            logger.error(f"Error sending message to websocket: {e}")
            # In server-to-server contexts, a WebSocket error should be quite rare. Given how hard
            # it is to recover from a send-side error with proper state management, and that exponential
            # backoff for retries can have cost/stability implications for a service cluster, let's just
            # treat a send-side error as fatal.
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

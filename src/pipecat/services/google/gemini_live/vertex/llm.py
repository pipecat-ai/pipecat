#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service for accessing Gemini Live via Google Vertex AI.

This module provides integration with Google's Gemini Live model via
Vertex AI, supporting both text and audio modalities with voice transcription,
streaming responses, and tool usage.
"""

import json
from dataclasses import dataclass

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiMediaResolution,
    GeminiModalities,
    HttpOptions,
    InputParams,
    language_to_gemini_language,
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


@dataclass
class GeminiLiveVertexLLMSettings(GeminiLiveLLMService.Settings):
    """Settings for GeminiLiveVertexLLMService."""

    pass


class GeminiLiveVertexLLMService(GeminiLiveLLMService):
    """Provides access to Google's Gemini Live model via Vertex AI.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage.
    """

    Settings = GeminiLiveVertexLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        credentials: str | None = None,
        credentials_path: str | None = None,
        location: str,
        project_id: str,
        model: str | None = None,
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: str | None = None,
        tools: list[dict] | ToolsSchema | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        inference_on_context_initialization: bool = True,
        file_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/files",
        http_options: HttpOptions | None = None,
        **kwargs,
    ):
        """Initialize the service for accessing Gemini Live via Google Vertex AI.

        Args:
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file.
            location: GCP region for Vertex AI endpoint (e.g., "us-east4").
            project_id: Google Cloud project ID.
            model: Model identifier to use.

                .. deprecated:: 0.0.105
                    Use ``settings=GeminiLiveVertexLLMService.Settings(model=...)`` instead.

            voice_id: TTS voice identifier. Defaults to "Charon".

                .. deprecated:: 0.0.105
                    Use ``settings=GeminiLiveVertexLLMService.Settings(voice=...)`` instead.
            start_audio_paused: Whether to start with audio input paused. Defaults to False.
            start_video_paused: Whether to start with video input paused. Defaults to False.
            system_instruction: System prompt for the model. Defaults to None.
            tools: Tools/functions available to the model. Defaults to None.
            params: Configuration parameters for the model along with Vertex AI
                location and project ID.

                .. deprecated:: 0.0.105
                    Use ``settings=GeminiLiveVertexLLMService.Settings(...)`` instead.

            settings: Gemini Live LLM settings. If provided together with deprecated
                top-level parameters, the ``settings`` values take precedence.
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

        # Build default_settings from deprecated args, then apply settings delta.
        # We pass settings= to super() instead of model=/params= to avoid
        # double deprecation warnings from the parent.

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="google/gemini-live-2.5-flash-native-audio",
            voice="Charon",
            frequency_penalty=None,
            max_tokens=4096,
            presence_penalty=None,
            temperature=None,
            top_k=None,
            top_p=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            modalities=GeminiModalities.AUDIO,
            language="en-US",
            media_resolution=GeminiMediaResolution.UNSPECIFIED,
            vad=None,
            context_window_compression={},
            thinking={},
            enable_affective_dialog=False,
            proactivity={},
            extra={},
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id != "Charon":
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.frequency_penalty = params.frequency_penalty
                default_settings.max_tokens = params.max_tokens
                default_settings.presence_penalty = params.presence_penalty
                default_settings.temperature = params.temperature
                default_settings.top_k = params.top_k
                default_settings.top_p = params.top_p
                default_settings.modalities = params.modalities
                default_settings.language = (
                    language_to_gemini_language(params.language) if params.language else "en-US"
                )
                default_settings.media_resolution = params.media_resolution
                default_settings.vad = params.vad
                default_settings.context_window_compression = (
                    params.context_window_compression.model_dump()
                    if params.context_window_compression
                    else {}
                )
                default_settings.thinking = params.thinking or {}
                default_settings.enable_affective_dialog = params.enable_affective_dialog or False
                default_settings.proactivity = params.proactivity or {}
                if isinstance(params.extra, dict):
                    default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Call parent constructor with the obtained settings
        super().__init__(
            # api_key is required by parent class, but actually not used with
            # Vertex
            api_key="dummy",
            start_audio_paused=start_audio_paused,
            start_video_paused=start_video_paused,
            system_instruction=system_instruction,
            tools=tools,
            settings=default_settings,
            inference_on_context_initialization=inference_on_context_initialization,
            file_api_base_url=file_api_base_url,
            http_options=http_options,
            **kwargs,
        )

    def _get_history_config(self):
        """Vertex AI does not support history_config."""
        return None

    def create_client(self):
        """Create the Gemini client instance."""
        self._client = Client(
            vertexai=True,
            credentials=self._credentials,
            project=self._project_id,
            location=self._location,
            http_options=self._http_options,
        )

    @property
    def file_api(self):
        """Gemini File API is not supported with Vertex AI."""
        raise NotImplementedError(
            "When using Vertex AI, the recommended approach is to use Google Cloud Storage for file handling. The Gemini File API is not directly supported in this context."
        )

    @staticmethod
    def _get_credentials(credentials: str | None, credentials_path: str | None) -> str:
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
        creds: service_account.Credentials | None = None

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

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini LLM service using the Interactions API on Vertex AI."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.llm.services.gemini_interactions_adapter import (
    GeminiInteractionsVertexLLMAdapter,
)
from pipecat.services.google.interactions.llm import (
    GoogleInteractionsLLMService,
    GoogleInteractionsLLMSettings,
)
from pipecat.services.google.utils import _vertex_credentials

try:
    import google.genai as genai
    from google.genai.types import HttpOptions
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Google Vertex AI, you need to `uv add "pipecat-ai[google]"`.')
    raise ImportError(f"Missing module: {e}") from e


@dataclass
class GoogleInteractionsVertexLLMSettings(GoogleInteractionsLLMSettings):
    """Settings for GoogleInteractionsVertexLLMService."""

    pass


class GoogleInteractionsVertexLLMService(GoogleInteractionsLLMService):
    """Google Gemini LLM service using the Interactions API on Vertex AI.

    The same service as :class:`GoogleInteractionsLLMService`, authenticated
    with Google Cloud credentials and sent to a Vertex AI project. Vertex
    serves the Interactions API from the ``global``, ``us`` and ``eu``
    locations, for Gemini 3 models and later.

    Example::

        llm = GoogleInteractionsVertexLLMService(
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            project_id=os.getenv("GOOGLE_PROJECT_ID"),
            settings=GoogleInteractionsVertexLLMService.Settings(
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    Settings = GoogleInteractionsVertexLLMSettings
    _settings: Settings

    adapter_class = GeminiInteractionsVertexLLMAdapter

    def __init__(
        self,
        *,
        project_id: str,
        credentials: str | None = None,
        credentials_path: str | None = None,
        location: str = "global",
        settings: Settings | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        http_options: HttpOptions | None = None,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            project_id: Google Cloud project ID.
            credentials: JSON string of service account credentials.
            credentials_path: Path to the service account JSON file. Without
                either, the application default credentials are used.
            location: Location of the Vertex AI endpoint: ``"global"`` (the
                default), ``"us"`` or ``"eu"``.
            settings: Runtime-updatable settings for this service.
            tools: Tools to offer the model when the context has none, in
                Gemini's ``generateContent`` format or the Interactions API's.
            tool_config: Configuration for tool usage, in Gemini's
                ``generateContent`` format (``function_calling_config``).
            http_options: HTTP options for the client.
            **kwargs: Additional arguments passed to
                :class:`GoogleInteractionsLLMService`, such as the stream
                timeouts.

        Raises:
            ValueError: If ``api_key`` is passed, or no valid credentials are
                provided or found.
        """
        if "api_key" in kwargs:
            raise ValueError(
                "GoogleInteractionsVertexLLMService does not accept 'api_key'. "
                "Use 'credentials' or 'credentials_path' for Vertex AI authentication."
            )

        # Set before super().__init__(), which calls create_client().
        self._credentials = _vertex_credentials(credentials, credentials_path)
        self._project_id = project_id
        self._location = location

        default_settings = self.Settings(model="gemini-3.8-flash")
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key="",
            settings=default_settings,
            tools=tools,
            tool_config=tool_config,
            http_options=http_options,
            **kwargs,
        )

    def create_client(self):
        """Create the Gemini client instance configured for Vertex AI."""
        self._client = genai.Client(
            vertexai=True,
            credentials=self._credentials,
            project=self._project_id,
            location=self._location,
            http_options=self._http_options,
        )

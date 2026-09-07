"""Speaches LLM Service — uses an OpenAI-compatible /v1/chat/completions endpoint."""

from dataclasses import dataclass

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class SpeachesLLMSettings(OpenAILLMSettings):
    """Settings for Speaches LLM service."""

    pass


class SpeachesLLMService(OpenAILLMService):
    """Speaches LLM service using an OpenAI-compatible chat completions endpoint."""

    Settings = SpeachesLLMSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:11434/v1",
        settings: SpeachesLLMSettings | None = None,
        **kwargs,
    ):
        """Initialize the Speaches LLM service.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the Speaches-compatible endpoint.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=settings,
            **kwargs,
        )

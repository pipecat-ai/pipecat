#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI speech-to-text service implementation.

Transcribes audio with Azure OpenAI's ``/audio/transcriptions`` deployments
(``gpt-4o-transcribe``, ``gpt-4o-mini-transcribe``, ``whisper-1``) over the
OpenAI-compatible interface. For Azure AI Speech (Cognitive Services) instead,
see :class:`pipecat.services.azure.stt.AzureSTTService`.
"""

from dataclasses import dataclass, field
from typing import Literal, cast, get_args

from loguru import logger
from openai import AsyncAzureOpenAI

from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.settings import NOT_GIVEN, _NotGiven, assert_given, is_given
from pipecat.services.stt_latency import OPENAI_TTFS_P99

AzureOpenAITranscribeModel = Literal[
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
]
"""Transcription models served by Azure OpenAI's ``/audio/transcriptions`` API.

These are model *families*: a dated deployment such as
``gpt-4o-mini-transcribe-2025-12-15`` shares the request shape of its base
model, so name the base model here.
"""

_TRANSCRIBE_MODELS: tuple[str, ...] = get_args(AzureOpenAITranscribeModel)

_DEFAULT_MODEL_FAMILY: AzureOpenAITranscribeModel = "gpt-4o-transcribe"

_DEFAULT_API_VERSION = "2025-04-01-preview"


@dataclass
class AzureOpenAISTTSettings(OpenAISTTService.Settings):
    """Settings for the Azure OpenAI STT service.

    Parameters:
        model: Name of the Azure OpenAI deployment to transcribe with. Azure
            routes on the deployment name, which is chosen when the model is
            deployed and need not match the underlying model.
        model_family: Model behind the deployment. Because a deployment name is
            arbitrary, this is what determines the response format the service
            asks for. Defaults to the deployment name when that names a known
            model, and to ``"gpt-4o-transcribe"`` otherwise.
    """

    model_family: AzureOpenAITranscribeModel | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AzureOpenAISTTService(OpenAISTTService):
    """Azure OpenAI speech-to-text service.

    Transcribes audio with an Azure OpenAI ``/audio/transcriptions`` deployment,
    reusing OpenAI's request shape over an ``AsyncAzureOpenAI`` client.

    Note that ``include_prob_metrics`` asks gpt-4o deployments for
    ``include=["logprobs"]``, which Azure's audio API doesn't document; prefer a
    ``whisper-1`` deployment when probability metrics matter.

    Example::

        stt = AzureOpenAISTTService(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint="https://my-resource.openai.azure.com",
            settings=AzureOpenAISTTService.Settings(model="my-transcribe-deployment"),
        )
    """

    Settings = AzureOpenAISTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        api_version: str = _DEFAULT_API_VERSION,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = OPENAI_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Azure OpenAI STT service.

        Args:
            api_key: API key for the Azure OpenAI resource.
            endpoint: Azure OpenAI resource endpoint, e.g.
                ``"https://my-resource.openai.azure.com"``.
            api_version: Azure OpenAI API version. Defaults to
                "2025-04-01-preview", the preview covering the gpt-4o
                transcription models.
            settings: Runtime-updatable settings. Set ``model`` to the name of
                your deployment; it defaults to ``"gpt-4o-transcribe"``, which
                is the name Azure gives a deployment of that model by default.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to OpenAISTTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(model_family=_DEFAULT_MODEL_FAMILY)

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

            # A deployment name is arbitrary, so the model behind it can't be
            # derived in general. Deployments are conventionally named after the
            # model they serve, though, so honor that when the caller named a
            # deployment but no family.
            if not is_given(settings.model_family) and default_settings.model in _TRANSCRIBE_MODELS:
                default_settings.model_family = cast(
                    AzureOpenAITranscribeModel, default_settings.model
                )

        # Assigned before super().__init__() because that constructs the client
        # via _create_client().
        self._endpoint = endpoint
        self._api_version = api_version

        super().__init__(
            api_key=api_key,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

    def _create_client(self, api_key: str | None, base_url: str | None):
        """Create an OpenAI-compatible client for the Azure OpenAI endpoint.

        Args:
            api_key: API key for authentication.
            base_url: Ignored. Azure routes on the endpoint, API version, and
                deployment name instead.

        Returns:
            AsyncAzureOpenAI: Configured Azure OpenAI client instance.
        """
        logger.debug(f"Creating Azure OpenAI client with endpoint {self._endpoint}")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )

    def _model_family(self) -> str | None:
        """Model whose capabilities shape the transcription request.

        Returns:
            The configured ``model_family``, since the deployment name in
            ``model`` doesn't identify the underlying model.
        """
        return assert_given(self._settings.model_family)

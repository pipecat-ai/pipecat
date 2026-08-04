#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Azure OpenAI STT service client wiring and model family resolution."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import AsyncAzureOpenAI

from pipecat.services.azure.openai_stt import AzureOpenAISTTService


def _service(**kwargs):
    return AzureOpenAISTTService(
        api_key="fake",
        endpoint="https://my-resource.openai.azure.com",
        **kwargs,
    )


class TestAzureOpenAISTTClient(unittest.TestCase):
    """Azure routes transcription on the resource endpoint, API version, and
    deployment name rather than a base URL, so the service must build an
    ``AsyncAzureOpenAI`` client instead of inheriting the plain ``AsyncOpenAI``
    one that ``BaseWhisperSTTService`` creates."""

    def test_client_is_async_azure_openai(self):
        self.assertIsInstance(_service()._client, AsyncAzureOpenAI)

    def test_endpoint_and_api_version_reach_the_client(self):
        with patch("pipecat.services.azure.openai_stt.AsyncAzureOpenAI") as mock_client:
            _service(api_version="2025-04-01-preview")

        kwargs = mock_client.call_args.kwargs
        self.assertEqual(kwargs["azure_endpoint"], "https://my-resource.openai.azure.com")
        self.assertEqual(kwargs["api_version"], "2025-04-01-preview")
        self.assertEqual(kwargs["api_key"], "fake")

    def test_api_version_defaults_to_documented_preview(self):
        with patch("pipecat.services.azure.openai_stt.AsyncAzureOpenAI") as mock_client:
            _service()

        self.assertEqual(mock_client.call_args.kwargs["api_version"], "2025-04-01-preview")


class TestAzureOpenAISTTModelFamily(unittest.TestCase):
    """On Azure, ``model`` is the deployment name — chosen at deployment time and
    unrelated to the underlying model — so it can't be used to decide which
    response formats the deployment accepts. ``model_family`` carries that
    instead."""

    def test_defaults_to_gpt_4o_transcribe(self):
        self.assertEqual(_service()._model_family(), "gpt-4o-transcribe")

    def test_inferred_from_a_deployment_named_after_a_known_model(self):
        # Azure defaults a deployment's name to the model's name, so the common
        # case needs no extra configuration.
        service = _service(settings=AzureOpenAISTTService.Settings(model="whisper-1"))
        self.assertEqual(service._model_family(), "whisper-1")

    def test_arbitrary_deployment_name_keeps_the_default_family(self):
        service = _service(settings=AzureOpenAISTTService.Settings(model="my-deployment"))
        self.assertEqual(service._model_family(), "gpt-4o-transcribe")
        # The deployment name is still what goes on the wire.
        self.assertEqual(service._settings.model, "my-deployment")

    def test_explicit_family_wins_over_the_deployment_name(self):
        service = _service(
            settings=AzureOpenAISTTService.Settings(
                model="whisper-1",
                model_family="gpt-4o-mini-transcribe",
            )
        )
        self.assertEqual(service._model_family(), "gpt-4o-mini-transcribe")


class TestAzureOpenAISTTRequestShape(unittest.IsolatedAsyncioTestCase):
    """The probability-metrics request differs by model: gpt-4o transcribe models
    take ``json`` plus ``include=["logprobs"]``, while Whisper takes
    ``verbose_json``. Azure deployments must pick that branch from
    ``model_family``, not from the deployment name."""

    async def _captured_transcribe_kwargs(self, **service_kwargs):
        service = _service(include_prob_metrics=True, **service_kwargs)
        create = AsyncMock(return_value=MagicMock(text="hello"))
        service._client = MagicMock(audio=MagicMock(transcriptions=MagicMock(create=create)))

        await service._transcribe(b"audio-bytes")

        return create.await_args.kwargs

    async def test_deployment_name_is_sent_as_model(self):
        kwargs = await self._captured_transcribe_kwargs(
            settings=AzureOpenAISTTService.Settings(model="my-deployment")
        )
        self.assertEqual(kwargs["model"], "my-deployment")

    async def test_gpt_4o_deployment_requests_logprobs(self):
        # An arbitrarily-named gpt-4o deployment would fall through to
        # ``verbose_json`` if the branch keyed off the deployment name.
        kwargs = await self._captured_transcribe_kwargs(
            settings=AzureOpenAISTTService.Settings(
                model="my-deployment",
                model_family="gpt-4o-transcribe",
            )
        )
        self.assertEqual(kwargs["response_format"], "json")
        self.assertEqual(kwargs["include"], ["logprobs"])

    async def test_whisper_deployment_requests_verbose_json(self):
        kwargs = await self._captured_transcribe_kwargs(
            settings=AzureOpenAISTTService.Settings(
                model="my-deployment",
                model_family="whisper-1",
            )
        )
        self.assertEqual(kwargs["response_format"], "verbose_json")
        self.assertNotIn("include", kwargs)


if __name__ == "__main__":
    unittest.main()

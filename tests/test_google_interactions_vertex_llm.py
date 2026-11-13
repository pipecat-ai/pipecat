#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GoogleInteractionsVertexLLMService."""

import unittest
from unittest.mock import MagicMock, patch

import pipecat.services.google.interactions.vertex.llm as vertex_llm
from pipecat.adapters.llm.services.gemini_interactions_adapter import (
    GeminiInteractionsLLMAdapter,
    GeminiInteractionsVertexLLMAdapter,
)
from pipecat.services.google.interactions.vertex.llm import GoogleInteractionsVertexLLMService

CREDENTIALS = object()


def make_service(**kwargs) -> tuple[GoogleInteractionsVertexLLMService, MagicMock]:
    """A service built without real credentials; returns it and the Client mock."""
    with (
        patch.object(vertex_llm, "_vertex_credentials", return_value=CREDENTIALS),
        patch.object(vertex_llm.genai, "Client") as client,
    ):
        return GoogleInteractionsVertexLLMService(project_id="test-project", **kwargs), client


class TestService(unittest.TestCase):
    def test_vertex_client(self):
        """Test the client being a Vertex client for the project, at the global location."""
        _, client = make_service()
        kwargs = client.call_args.kwargs
        self.assertTrue(kwargs["vertexai"])
        self.assertIs(kwargs["credentials"], CREDENTIALS)
        self.assertEqual(kwargs["project"], "test-project")
        self.assertEqual(kwargs["location"], "global")
        self.assertNotIn("api_key", kwargs)

    def test_location(self):
        """Test a location other than global reaching the client."""
        _, client = make_service(location="eu")
        self.assertEqual(client.call_args.kwargs["location"], "eu")

    def test_rejects_api_key(self):
        """Test an API key being refused, since Vertex authenticates with credentials."""
        with self.assertRaisesRegex(ValueError, "does not accept 'api_key'"):
            make_service(api_key="key")

    def test_default_model(self):
        """Test the default model, and a model from the settings replacing it."""
        service, _ = make_service()
        self.assertEqual(service._settings.model, "gemini-3.8-flash")
        self.assertEqual(service._settings.max_tokens, 4096)

        service, _ = make_service(
            settings=GoogleInteractionsVertexLLMService.Settings(model="gemini-3.6-flash")
        )
        self.assertEqual(service._settings.model, "gemini-3.6-flash")

    def test_gs_urls_read_by_vertex(self):
        """Test gs:// file URLs passing through on Vertex but not on the Gemini API."""
        url = "gs://bucket/cat.jpg"
        self.assertTrue(GeminiInteractionsVertexLLMAdapter().supports_file_url(url, "image/jpeg"))
        self.assertFalse(GeminiInteractionsLLMAdapter().supports_file_url(url, "image/jpeg"))
        service, _ = make_service()
        self.assertIsInstance(service.get_llm_adapter(), GeminiInteractionsVertexLLMAdapter)


if __name__ == "__main__":
    unittest.main()

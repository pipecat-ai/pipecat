#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.adapters.base_llm_adapter import LLMContextConversionError
from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter, GeminiVertexLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.file_resolver import FileResolver, FileResolverError
from pipecat.utils.file_storage import LocalFileStorage


def _mock_http_session(response):
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _mock_http_response(*, status=200, body=b"", raise_for_status_error=None):
    response = MagicMock()
    response.status = status
    if raise_for_status_error is not None:
        response.raise_for_status = MagicMock(side_effect=raise_for_status_error)
    else:
        response.raise_for_status = MagicMock()
    response.content.read = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


class TestFileResolverFetch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Default http(s) reachability to "public" so tests exercising the
        # fetch itself don't depend on real DNS. Policy tests override this.
        classify_patcher = patch(
            "pipecat.utils.file_resolver.classify_url_reachability",
            new=AsyncMock(return_value="public"),
        )
        self.classify_mock = classify_patcher.start()
        self.addCleanup(classify_patcher.stop)

    # -- data: URLs -------------------------------------------------------------

    async def test_data_url_is_decoded(self):
        raw = b"%PDF-1.4 fake content"
        b64 = base64.b64encode(raw).decode()
        resolver = FileResolver()
        result = await resolver.fetch(f"data:application/pdf;base64,{b64}")
        self.assertEqual(result, raw)

    async def test_malformed_data_url_raises(self):
        resolver = FileResolver()
        with self.assertRaises(FileResolverError):
            await resolver.fetch("data:application/pdf;base64")

    # -- http(s) URLs -----------------------------------------------------------

    async def test_http_fetch_returns_bytes(self):
        raw = b"%PDF-1.4 fetched content"
        session = _mock_http_session(_mock_http_response(body=raw))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver()
            result = await resolver.fetch("https://example.com/doc.pdf")
        self.assertEqual(result, raw)

    async def test_http_blocked_raises_without_fetching(self):
        self.classify_mock.return_value = "blocked"
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession") as session_cls:
            resolver = FileResolver()
            with self.assertRaises(FileResolverError):
                await resolver.fetch("http://169.254.169.254/latest/meta-data/")
        session_cls.assert_not_called()

    async def test_http_allowed_network_is_fetched(self):
        self.classify_mock.return_value = "allowed"
        raw = b"internal file"
        session = _mock_http_session(_mock_http_response(body=raw))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver(allowed_url_networks=["10.0.0.0/8"])
            result = await resolver.fetch("https://internal.example.com/private.pdf")
        self.assertEqual(result, raw)

    async def test_allowed_url_networks_passed_to_classification(self):
        import ipaddress

        resolver = FileResolver(allowed_url_networks=["10.0.0.0/8"])
        await resolver.classify("https://internal.example.com/private.pdf")
        self.classify_mock.assert_called_once_with(
            "https://internal.example.com/private.pdf",
            [ipaddress.ip_network("10.0.0.0/8")],
        )

    async def test_http_redirect_is_refused(self):
        session = _mock_http_session(_mock_http_response(status=302))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver()
            with self.assertRaises(FileResolverError) as ctx:
                await resolver.fetch("https://example.com/redirects.pdf")
        self.assertIn("redirect", str(ctx.exception))
        session.get.assert_called_once()
        self.assertFalse(session.get.call_args.kwargs.get("allow_redirects", True))

    async def test_http_error_raises(self):
        import aiohttp

        error = aiohttp.ClientResponseError(MagicMock(), (), status=404)
        session = _mock_http_session(_mock_http_response(raise_for_status_error=error))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver()
            with self.assertRaises(FileResolverError):
                await resolver.fetch("https://example.com/missing.pdf")

    async def test_http_timeout_raises(self):
        session = _mock_http_session(_mock_http_response(raise_for_status_error=TimeoutError()))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver()
            with self.assertRaises(FileResolverError) as ctx:
                await resolver.fetch("https://slow.example.com/file.pdf")
        self.assertIn("Timed out", str(ctx.exception))

    async def test_http_oversized_raises(self):
        session = _mock_http_session(_mock_http_response(body=b"x" * 11))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver(max_fetch_bytes=10)
            with self.assertRaises(FileResolverError) as ctx:
                await resolver.fetch("https://example.com/huge.pdf")
        self.assertIn("byte limit", str(ctx.exception))

    # -- storage-minted URLs ------------------------------------------------------

    async def test_storage_url_loads_from_storage(self):
        raw = b"%PDF-1.4 uploaded content"
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)
            file_url = await storage.save("doc.pdf", raw)
            resolver = FileResolver(file_storage=storage)
            result = await resolver.fetch(file_url)
        self.assertEqual(result, raw)

    async def test_local_storage_upload_deleted_after_load(self):
        """LocalFileStorage owns its uploads, so a consumed file is removed from disk."""
        raw = b"%PDF-1.4 uploaded content"
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)
            file_url = await storage.save("doc.pdf", raw)
            resolver = FileResolver(file_storage=storage)
            await resolver.fetch(file_url)

            with self.assertRaises(FileResolverError):
                await resolver.fetch(file_url)

    async def test_storage_without_delete_after_load_keeps_the_file(self):
        """A backend that doesn't own its URLs (delete_after_load False) is never deleted from."""
        storage = MagicMock()
        storage.delete_after_load = False
        storage.load = AsyncMock(return_value=b"shared object")
        storage.delete = AsyncMock()

        resolver = FileResolver(file_storage=storage)
        result = await resolver.fetch("s3://bucket/shared/report.pdf")

        self.assertEqual(result, b"shared object")
        storage.delete.assert_not_called()

    async def test_storage_url_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = FileResolver(file_storage=LocalFileStorage(tmpdir))
            with self.assertRaises(FileResolverError) as ctx:
                await resolver.fetch("pipecat:" + "0" * 32)
        self.assertIn("not found", str(ctx.exception))

    async def test_storage_url_path_traversal_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = FileResolver(file_storage=LocalFileStorage(tmpdir))
            with self.assertRaises(FileResolverError):
                await resolver.fetch("pipecat:../../../etc/passwd")

    async def test_unresolvable_scheme_without_storage_raises(self):
        resolver = FileResolver()
        with self.assertRaises(FileResolverError):
            await resolver.fetch("pipecat:" + "0" * 32)

    async def test_unresolvable_scheme_gs_without_storage_raises(self):
        resolver = FileResolver()
        with self.assertRaises(FileResolverError):
            await resolver.fetch("gs://bucket/key.pdf")


class TestAdapterSupportsFileUrl(unittest.TestCase):
    def test_openai_supports_http_images_only(self):
        adapter = OpenAILLMAdapter()
        self.assertTrue(adapter.supports_file_url("https://x.com/a.png", "image/png"))
        self.assertFalse(adapter.supports_file_url("https://x.com/a.pdf", "application/pdf"))
        self.assertFalse(adapter.supports_file_url("s3://bucket/a.png", "image/png"))

    def test_openai_responses_supports_http(self):
        adapter = OpenAIResponsesLLMAdapter()
        self.assertTrue(adapter.supports_file_url("https://x.com/a.pdf", "application/pdf"))
        self.assertTrue(adapter.supports_file_url("https://x.com/a.png", "image/png"))
        self.assertFalse(adapter.supports_file_url("gs://bucket/a.pdf", "application/pdf"))

    def test_anthropic_supports_http_images_and_pdfs(self):
        adapter = AnthropicLLMAdapter()
        self.assertTrue(adapter.supports_file_url("https://x.com/a.png", "image/png"))
        self.assertTrue(adapter.supports_file_url("https://x.com/a.pdf", "application/pdf"))
        self.assertFalse(adapter.supports_file_url("https://x.com/a.mp4", "video/mp4"))
        self.assertFalse(adapter.supports_file_url("s3://bucket/a.pdf", "application/pdf"))

    def test_bedrock_supports_s3_only(self):
        adapter = AWSBedrockLLMAdapter()
        self.assertTrue(adapter.supports_file_url("s3://bucket/a.pdf", "application/pdf"))
        self.assertFalse(adapter.supports_file_url("https://x.com/a.pdf", "application/pdf"))

    def test_gemini_supports_own_file_api_but_not_gs(self):
        """The developer API needs a gs:// URI registered with the Files API first,
        so only registered Files API URIs pass through; gs:// gets fetched and inlined."""
        adapter = GeminiLLMAdapter()
        self.assertFalse(adapter.supports_file_url("gs://bucket/a.pdf", "application/pdf"))
        self.assertTrue(
            adapter.supports_file_url(
                "https://generativelanguage.googleapis.com/v1beta/files/abc", "application/pdf"
            )
        )
        self.assertFalse(adapter.supports_file_url("https://x.com/a.pdf", "application/pdf"))
        self.assertFalse(adapter.supports_file_url("s3://bucket/a.pdf", "application/pdf"))

    def test_gemini_vertex_additionally_supports_gs(self):
        adapter = GeminiVertexLLMAdapter()
        self.assertTrue(adapter.supports_file_url("gs://bucket/a.pdf", "application/pdf"))
        self.assertTrue(
            adapter.supports_file_url(
                "https://generativelanguage.googleapis.com/v1beta/files/abc", "application/pdf"
            )
        )
        self.assertFalse(adapter.supports_file_url("https://x.com/a.pdf", "application/pdf"))


class TestResolveFileItems(unittest.IsolatedAsyncioTestCase):
    """The resolution pass: pass through what the provider can consume, fetch and inline the rest."""

    RAW = b"%PDF-1.4 fetched"

    def _resolver(self, *, classify="public"):
        resolver = FileResolver()
        resolver.classify = AsyncMock(return_value=classify)
        resolver.fetch = AsyncMock(return_value=self.RAW)
        return resolver

    @staticmethod
    def _file_url_context(url, mime="application/pdf", filename="doc.pdf"):
        message = LLMContext.create_file_url_message(format=mime, url=url, filename=filename)
        return LLMContext(messages=[message])

    @staticmethod
    def _file_item(context):
        return context.get_messages()[0]["content"][-1]

    async def test_supported_public_url_passes_through(self):
        context = self._file_url_context("https://example.com/a.png", mime="image/png")
        resolver = self._resolver(classify="public")
        await OpenAILLMAdapter().resolve_file_items(context, resolver)

        item = self._file_item(context)
        self.assertEqual(item["type"], "file_url")
        self.assertEqual(item["file"]["url"], "https://example.com/a.png")
        resolver.fetch.assert_not_called()

    async def test_unsupported_url_is_fetched_and_inlined(self):
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()
        await OpenAILLMAdapter().resolve_file_items(context, resolver)

        resolver.fetch.assert_called_once_with("https://example.com/a.pdf")
        item = self._file_item(context)
        self.assertEqual(item["type"], "file_base64")
        expected_b64 = base64.b64encode(self.RAW).decode()
        self.assertEqual(item["file"]["file_data"], f"data:application/pdf;base64,{expected_b64}")
        self.assertEqual(item["file"]["filename"], "doc.pdf")
        self.assertEqual(item["file"]["mime_type"], "application/pdf")

    async def test_supported_url_on_private_allowed_network_is_fetched(self):
        context = self._file_url_context("https://internal.example.com/a.pdf")
        resolver = self._resolver(classify="allowed")
        await AnthropicLLMAdapter().resolve_file_items(context, resolver)

        resolver.fetch.assert_called_once()
        self.assertEqual(self._file_item(context)["type"], "file_base64")

    async def test_cloud_uri_passes_through_without_reachability_check(self):
        context = self._file_url_context("s3://bucket/a.pdf")
        resolver = self._resolver()
        await AWSBedrockLLMAdapter().resolve_file_items(context, resolver)

        resolver.classify.assert_not_called()
        resolver.fetch.assert_not_called()
        self.assertEqual(self._file_item(context)["type"], "file_url")

    async def test_pass_through_is_memoized_per_adapter(self):
        context = self._file_url_context("https://example.com/a.png", mime="image/png")
        resolver = self._resolver(classify="public")
        adapter = OpenAILLMAdapter()
        await adapter.resolve_file_items(context, resolver)
        await adapter.resolve_file_items(context, resolver)

        resolver.classify.assert_called_once()

    async def test_switching_adapters_reevaluates_the_url(self):
        """A URL one provider passes through gets fetched when another can't consume it."""
        context = self._file_url_context("gs://bucket/a.pdf")
        resolver = self._resolver()

        await GeminiVertexLLMAdapter().resolve_file_items(context, resolver)
        self.assertEqual(self._file_item(context)["type"], "file_url")
        resolver.fetch.assert_not_called()

        await AnthropicLLMAdapter().resolve_file_items(context, resolver)
        resolver.fetch.assert_called_once_with("gs://bucket/a.pdf")
        self.assertEqual(self._file_item(context)["type"], "file_base64")

    async def test_provider_variants_do_not_share_pass_through_decisions(self):
        """Adapters sharing an LLM-specific-message id (Vertex vs developer-API Gemini)
        still re-evaluate each other's pass-through decisions."""
        context = self._file_url_context("gs://bucket/a.pdf")
        resolver = self._resolver()

        await GeminiVertexLLMAdapter().resolve_file_items(context, resolver)
        resolver.fetch.assert_not_called()

        await GeminiLLMAdapter().resolve_file_items(context, resolver)
        resolver.fetch.assert_called_once_with("gs://bucket/a.pdf")
        self.assertEqual(self._file_item(context)["type"], "file_base64")

    async def test_raw_bytes_cached_for_adapters_that_prefer_them(self):
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()
        await AWSBedrockLLMAdapter().resolve_file_items(context, resolver)

        self.assertEqual(self._file_item(context)["file"]["_raw_bytes"], self.RAW)

    async def test_inline_base64_decoded_once_for_adapters_that_prefer_raw_bytes(self):
        raw = b"inline content"
        b64 = base64.b64encode(raw).decode()
        message = await LLMContext.create_file_message(
            type="bytes", format="application/pdf", file=f"data:application/pdf;base64,{b64}"
        )
        context = LLMContext(messages=[message])
        resolver = self._resolver()
        await AWSBedrockLLMAdapter().resolve_file_items(context, resolver)

        item = self._file_item(context)
        self.assertEqual(item["file"]["_raw_bytes"], raw)

    async def test_fetch_failure_raises_conversion_error(self):
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()
        resolver.fetch = AsyncMock(side_effect=FileResolverError("nope"))

        with self.assertRaises(LLMContextConversionError):
            await OpenAILLMAdapter().resolve_file_items(context, resolver)

    async def test_text_only_context_is_untouched(self):
        context = LLMContext(messages=[{"role": "user", "content": "hello"}])
        resolver = self._resolver()
        await OpenAILLMAdapter().resolve_file_items(context, resolver)
        resolver.fetch.assert_not_called()
        resolver.classify.assert_not_called()


if __name__ == "__main__":
    unittest.main()

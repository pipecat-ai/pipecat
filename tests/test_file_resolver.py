#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.adapters.base_llm_adapter import LLMContextConversionError
from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter, GeminiVertexLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter
from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.file_resolver import FileResolver, FileResolverError
from pipecat.utils.file_storage import LocalFileStorage
from pipecat.utils.security.ssrf import UrlReachability


def _mock_http_session(response):
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _mock_http_response(
    *, status=200, body=b"", content_length=None, raise_for_status_error=None, chunk_size=8
):
    """Mock aiohttp response whose body streams in several chunks."""
    response = MagicMock()
    response.status = status
    response.content_length = content_length
    if raise_for_status_error is not None:
        response.raise_for_status = MagicMock(side_effect=raise_for_status_error)
    else:
        response.raise_for_status = MagicMock()

    def iter_chunked(size):
        async def chunks():
            for i in range(0, len(body), chunk_size):
                yield body[i : i + chunk_size]

        return chunks()

    response.content.iter_chunked = iter_chunked
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

    async def test_allowed_url_networks_defaults_from_env_var(self):
        import ipaddress
        import os

        with patch.dict(os.environ, {"PIPECAT_ALLOWED_FILE_URL_NETWORKS": "10.0.0.0/8"}):
            resolver = FileResolver()
        self.assertEqual(resolver._allowed_url_networks, [ipaddress.ip_network("10.0.0.0/8")])

    async def test_allowed_url_networks_argument_overrides_env_var(self):
        import os

        with patch.dict(os.environ, {"PIPECAT_ALLOWED_FILE_URL_NETWORKS": "10.0.0.0/8"}):
            resolver = FileResolver(allowed_url_networks=[])
        self.assertEqual(resolver._allowed_url_networks, [])

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

    async def test_http_oversized_stream_raises(self):
        """A body that only reveals its size while streaming is cut off at the cap."""
        session = _mock_http_session(_mock_http_response(body=b"x" * 11, chunk_size=4))
        with patch("pipecat.utils.file_resolver.aiohttp.ClientSession", return_value=session):
            resolver = FileResolver(max_fetch_bytes=10)
            with self.assertRaises(FileResolverError) as ctx:
                await resolver.fetch("https://example.com/huge.pdf")
        self.assertIn("byte limit", str(ctx.exception))

    async def test_http_oversized_content_length_rejected_before_reading(self):
        """A Content-Length over the cap is rejected without reading the body."""
        response = _mock_http_response(body=b"irrelevant", content_length=11)
        response.content.iter_chunked = MagicMock(
            side_effect=AssertionError("body should not be read")
        )
        session = _mock_http_session(response)
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

    async def test_forget_forces_a_refetch(self):
        """An application that knows a URL's content changed can drop the cache."""
        raw_v1 = b"version 1"
        raw_v2 = b"version 2"
        resolver = FileResolver()
        resolver._fetch_uncached = AsyncMock(side_effect=[raw_v1, raw_v2])

        self.assertEqual(await resolver.fetch("https://example.com/live.pdf"), raw_v1)
        self.assertEqual(await resolver.fetch("https://example.com/live.pdf"), raw_v1)

        resolver.forget("https://example.com/live.pdf")

        self.assertEqual(await resolver.fetch("https://example.com/live.pdf"), raw_v2)
        self.assertIsNone(resolver.cached_data_url("https://example.com/live.pdf"))

    async def test_local_storage_upload_deleted_after_load(self):
        """LocalFileStorage owns its uploads: a consumed file is removed from disk,
        and the resolver's cache becomes the surviving copy."""
        raw = b"%PDF-1.4 uploaded content"
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)
            file_url = await storage.save("doc.pdf", raw)
            resolver = FileResolver(file_storage=storage)
            self.assertEqual(await resolver.fetch(file_url), raw)

            self.assertEqual(list(Path(tmpdir).iterdir()), [])
            # Deleted from disk, but still resolvable from the cache.
            self.assertEqual(await resolver.fetch(file_url), raw)

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
    """The resolution pass: pass through what the provider can consume, fetch the
    rest into the resolver's caches — never touching the context itself."""

    RAW = b"%PDF-1.4 fetched"

    def _resolver(self):
        """A real resolver with the network edge stubbed out: _fetch_uncached is
        mocked, so fetch() still populates the caches."""
        resolver = FileResolver()
        resolver._fetch_uncached = AsyncMock(return_value=self.RAW)
        return resolver

    @staticmethod
    def _seed_reachability(resolver, url, verdict):
        resolver._reachability_cache[url] = verdict

    @staticmethod
    def _file_url_context(url, mime="application/pdf", filename="doc.pdf"):
        message = LLMContext.create_file_url_message(format=mime, url=url, filename=filename)
        return LLMContext(messages=[message])

    @staticmethod
    def _snapshot(context):
        import copy

        return copy.deepcopy(context.get_messages())

    async def test_supported_public_url_passes_through_untouched(self):
        context = self._file_url_context("https://example.com/a.png", mime="image/png")
        before = self._snapshot(context)
        resolver = self._resolver()
        self._seed_reachability(resolver, "https://example.com/a.png", UrlReachability.PUBLIC)

        adapter = OpenAILLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)

        resolver._fetch_uncached.assert_not_called()
        self.assertEqual(context.get_messages(), before)

    async def test_unsupported_url_is_fetched_and_inlined_at_conversion(self):
        context = self._file_url_context("https://example.com/a.pdf")
        before = self._snapshot(context)
        resolver = self._resolver()

        adapter = OpenAILLMAdapter()
        adapter.file_resolver = resolver
        params = await adapter.get_llm_invocation_params(context, convert_developer_to_user=False)

        resolver._fetch_uncached.assert_called_once_with("https://example.com/a.pdf")
        # The provider request carries the inlined bytes...
        item = params["messages"][0]["content"][-1]
        expected_b64 = base64.b64encode(self.RAW).decode()
        self.assertEqual(item["type"], "file")
        self.assertEqual(item["file"]["file_data"], f"data:application/pdf;base64,{expected_b64}")
        # ...while the canonical context still holds the URL, untouched.
        self.assertEqual(context.get_messages(), before)

    async def test_supported_url_on_private_allowed_network_is_fetched(self):
        context = self._file_url_context("https://internal.example.com/a.pdf")
        resolver = self._resolver()
        self._seed_reachability(
            resolver, "https://internal.example.com/a.pdf", UrlReachability.ALLOWED
        )

        adapter = AnthropicLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)

        resolver._fetch_uncached.assert_called_once()
        self.assertIsNotNone(resolver.cached_data_url("https://internal.example.com/a.pdf"))

    async def test_cloud_uri_passes_through_without_reachability_check(self):
        context = self._file_url_context("s3://bucket/a.pdf")
        resolver = self._resolver()

        with patch(
            "pipecat.utils.file_resolver.classify_url_reachability", new=AsyncMock()
        ) as classify_mock:
            adapter = AWSBedrockLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)

        classify_mock.assert_not_called()
        resolver._fetch_uncached.assert_not_called()

    async def test_reachability_classified_once_per_url(self):
        """DNS classification is cached on the resolver, so repeated resolution
        passes — and other adapters sharing the resolver — don't re-resolve."""
        context = self._file_url_context("https://example.com/a.png", mime="image/png")
        resolver = self._resolver()

        with patch(
            "pipecat.utils.file_resolver.classify_url_reachability",
            new=AsyncMock(return_value=UrlReachability.PUBLIC),
        ) as classify_mock:
            adapter = OpenAILLMAdapter()
            adapter.file_resolver = resolver
            await adapter.prepare_file_content(context)
            await adapter.prepare_file_content(context)
            other_adapter = OpenAIResponsesLLMAdapter()
            other_adapter.file_resolver = resolver
            await other_adapter.prepare_file_content(context)

        classify_mock.assert_called_once()

    async def test_switching_adapters_shares_the_download(self):
        """A URL one provider passes through gets fetched when another can't
        consume it — and adapters sharing the resolver download it only once."""
        context = self._file_url_context("gs://bucket/a.pdf")
        before = self._snapshot(context)
        resolver = self._resolver()

        adapter = GeminiVertexLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)
        resolver._fetch_uncached.assert_not_called()

        adapter = AnthropicLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)
        resolver._fetch_uncached.assert_called_once_with("gs://bucket/a.pdf")

        # A third provider (raw-bytes consumer) reuses the cached download.
        adapter = AWSBedrockLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)
        resolver._fetch_uncached.assert_called_once()
        self.assertEqual(resolver.cached_bytes("gs://bucket/a.pdf"), self.RAW)

        self.assertEqual(context.get_messages(), before)

    async def test_data_url_form_prepared_for_base64_consumers(self):
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()
        self._seed_reachability(resolver, "https://example.com/a.pdf", UrlReachability.ALLOWED)

        adapter = AnthropicLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)

        expected_b64 = base64.b64encode(self.RAW).decode()
        self.assertEqual(
            resolver.cached_data_url("https://example.com/a.pdf"),
            f"data:application/pdf;base64,{expected_b64}",
        )

    async def test_inline_base64_decoded_once_for_adapters_that_prefer_raw_bytes(self):
        raw = b"inline content"
        b64 = base64.b64encode(raw).decode()
        data_url = f"data:application/pdf;base64,{b64}"
        message = await LLMContext.create_file_message(
            type="bytes", format="application/pdf", file=data_url
        )
        context = LLMContext(messages=[message])
        before = self._snapshot(context)
        resolver = FileResolver()

        adapter = AWSBedrockLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)

        self.assertEqual(resolver.cached_bytes(data_url), raw)
        self.assertEqual(context.get_messages(), before)

    async def test_logging_conversion_reads_the_same_caches(self):
        """get_messages_for_logging converts the context too; a resolved file
        must not blow it up just because that path doesn't run resolution."""
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()
        self._seed_reachability(resolver, "https://example.com/a.pdf", UrlReachability.ALLOWED)

        adapter = AWSBedrockLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.get_llm_invocation_params(context)

        messages = adapter.get_messages_for_logging(context)
        self.assertEqual(len(messages), 1)

    async def test_fetch_failure_raises_conversion_error(self):
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()
        resolver._fetch_uncached = AsyncMock(side_effect=FileResolverError("nope"))

        adapter = OpenAILLMAdapter()
        adapter.file_resolver = resolver
        with self.assertRaises(LLMContextConversionError):
            await adapter.prepare_file_content(context)

    async def test_adapter_for_provider_without_files_fetches_nothing(self):
        context = self._file_url_context("https://example.com/a.pdf")
        resolver = self._resolver()

        adapter = OpenAIRealtimeLLMAdapter()
        adapter.file_resolver = resolver
        await adapter.get_llm_invocation_params(context)

        resolver._fetch_uncached.assert_not_called()

    async def test_text_only_context_is_untouched(self):
        context = LLMContext(messages=[{"role": "user", "content": "hello"}])
        resolver = self._resolver()
        adapter = OpenAILLMAdapter()
        adapter.file_resolver = resolver
        await adapter.prepare_file_content(context)
        resolver._fetch_uncached.assert_not_called()


if __name__ == "__main__":
    unittest.main()

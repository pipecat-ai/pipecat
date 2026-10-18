#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resolution of file URLs in an LLM context into bytes the provider can consume.

A file reaches the context as a URL — ``pipecat:<id>`` from the development
runner's upload endpoint, ``gs://``/``s3://`` from cloud storage, or plain
``http(s)``. Whether that URL can be passed to the LLM provider as-is, or must
first be fetched by this server and inlined as bytes, depends on the provider
(each adapter declares what it can consume) and on who can reach the URL.
:class:`FileResolver` is the fetching half of that decision: it downloads the
bytes when the provider can't, using the configured
:class:`~pipecat.utils.file_storage.FileStorage` for URLs the storage backend
minted and plain HTTP for the rest, guarded by an SSRF reachability policy.

The context itself is never modified: everything the resolver produces —
downloaded bytes, base64 data URLs, reachability verdicts — is held in
in-memory caches on the resolver, keyed by URL, where adapters read it
synchronously during context conversion. The caches make resolved files
provider-neutral: LLM services that share one resolver instance (e.g. services
switched between mid-session) reuse each other's downloads, re-deciding only
the per-provider question of whether a URL can be passed through.

The resolver is configured on the LLM service (``file_resolver=...``), which
resolves and converts in one step right before each completion — see
:meth:`~pipecat.adapters.base_llm_adapter.BaseLLMAdapter.get_llm_invocation_params`.
"""

import asyncio
import base64
import ipaddress

import aiohttp

from pipecat.utils.file_storage import FileStorage
from pipecat.utils.security.ssrf import (
    IPNetwork,
    UrlReachability,
    classify_url_reachability,
    default_allowed_file_url_networks,
)

_DEFAULT_MAX_FETCH_BYTES = 50 * 1024 * 1024
_DEFAULT_FETCH_TIMEOUT_SECS = 30


class FileResolverError(Exception):
    """Raised when a file URL in the LLM context cannot be resolved to bytes."""

    pass


class FileResolver:
    """Fetches and caches the bytes behind file URLs the LLM provider can't fetch itself.

    Handles three kinds of URL:

    - ``data:`` URLs are decoded directly.
    - ``http(s)`` URLs are fetched by this server, but only when the resolved
      address is publicly routable or falls within ``allowed_url_networks`` —
      a client-supplied URL must not be able to point this server at arbitrary
      private address space (SSRF). Redirects are refused rather than followed.
    - Any other scheme is delegated to ``file_storage``, which resolves the
      URLs it minted (``pipecat:<id>`` for :class:`~pipecat.utils.file_storage.LocalFileStorage`;
      a custom backend resolves whatever ``save()`` returned, e.g. ``gs://``
      with the deployment's own credentials).

    Each URL is fetched once and served from the cache afterwards: a URL is
    assumed to identify one immutable piece of content, the same assumption
    the rest of the web's caches make. Content that changes should live at a
    new URL (uploads through the development runner mint a fresh URL per
    upload; a cache-busting query string works for external URLs), or use
    :meth:`forget` when the application knows a URL's content changed. Note
    the flip side for URLs a provider consumes directly: the provider fetches
    those itself, on its own schedule, so their freshness is out of this
    resolver's hands entirely.

    A stored file whose backend sets ``delete_after_load`` is deleted only
    after its bytes are cached, so the cache entry becomes the surviving
    copy — which is why LLM services that may consume the same files (e.g.
    services switched between mid-session) should share one resolver instance
    rather than each constructing their own. Scope a resolver to one session:
    an instance shared across sessions would hold every session's files in
    memory and stretch the one-fetch-per-URL assumption across all of them.

    Subclass and override :meth:`fetch` to support additional schemes or
    credentialed fetches beyond what the storage backend provides.
    """

    def __init__(
        self,
        *,
        file_storage: FileStorage | None = None,
        allowed_url_networks: list[str] | None = None,
        max_fetch_bytes: int = _DEFAULT_MAX_FETCH_BYTES,
        fetch_timeout_secs: float = _DEFAULT_FETCH_TIMEOUT_SECS,
    ):
        """Initialize the file resolver.

        Args:
            file_storage: Storage backend used to resolve URLs it minted (e.g.
                ``pipecat:<id>`` from the development runner's upload
                endpoints — pass ``runner_args.file_storage`` there). Without
                it, only ``data:`` and ``http(s)`` URLs can be resolved.
            allowed_url_networks: CIDR ranges (e.g. ``["10.0.0.0/8"]``) this
                server is trusted to fetch from, in addition to the public
                internet. An ``http(s)`` URL that resolves outside both is
                refused. Defaults to the ``PIPECAT_ALLOWED_FILE_URL_NETWORKS``
                env var (comma-separated).
            max_fetch_bytes: Maximum size of a fetched file.
            fetch_timeout_secs: Total timeout for an ``http(s)`` fetch.
        """
        self._file_storage = file_storage
        if allowed_url_networks is None:
            allowed_url_networks = default_allowed_file_url_networks()
        self._allowed_url_networks: list[IPNetwork] = [
            ipaddress.ip_network(net) for net in allowed_url_networks
        ]
        self._max_fetch_bytes = max_fetch_bytes
        self._fetch_timeout_secs = fetch_timeout_secs

        # Everything below is keyed by the URL exactly as it appears in the
        # context, including data: URLs (whose Python string hash is memoized,
        # so repeated lookups don't rescan the payload).
        self._bytes_cache: dict[str, bytes] = {}
        self._data_url_cache: dict[str, str] = {}
        self._reachability_cache: dict[str, UrlReachability] = {}

    @property
    def file_storage(self) -> FileStorage | None:
        """The storage backend used to resolve URLs it minted, if any."""
        return self._file_storage

    def cached_bytes(self, url: str) -> bytes | None:
        """Return the cached bytes for `url`, or None if it hasn't been fetched."""
        return self._bytes_cache.get(url)

    def cached_data_url(self, url: str) -> str | None:
        """Return the cached ``data:`` form for `url`, or None if not prepared.

        A ``data:`` URL is its own data-URL form.
        """
        if url.startswith("data:"):
            return url
        return self._data_url_cache.get(url)

    def cached_reachability(self, url: str) -> UrlReachability | None:
        """Return the cached reachability verdict for `url`, or None if unclassified."""
        return self._reachability_cache.get(url)

    def forget(self, url: str) -> None:
        """Drop everything cached for `url`, forcing a re-fetch on next use.

        For applications that know a URL's content has changed. A forgotten
        ``delete_after_load`` upload can't be re-fetched — its stored copy was
        deleted when first cached — so forgetting one makes it unresolvable.

        Args:
            url: The URL whose cached content and reachability to drop.
        """
        self._bytes_cache.pop(url, None)
        self._data_url_cache.pop(url, None)
        self._reachability_cache.pop(url, None)

    async def classify(self, url: str) -> UrlReachability:
        """Classify who can reach an ``http(s)`` URL, honoring ``allowed_url_networks``.

        Classified once per URL; later calls return the cached verdict.
        """
        cached = self._reachability_cache.get(url)
        if cached is None:
            cached = await classify_url_reachability(url, self._allowed_url_networks)
            self._reachability_cache[url] = cached
        return cached

    async def fetch(self, url: str) -> bytes:
        """Return the bytes behind `url`, fetching and caching them on first use.

        Args:
            url: A ``data:`` URL, an ``http(s)`` URL, or a URL minted by the
                configured storage backend. A storage-resolved file is deleted
                after a successful load when the backend sets
                ``delete_after_load`` — the bytes are cached first, so the
                cache entry becomes the surviving copy.

        Raises:
            FileResolverError: If the URL is refused by the reachability
                policy, exceeds the size limit, fails to download, or has a
                scheme nothing is configured to resolve.
        """
        cached = self._bytes_cache.get(url)
        if cached is not None:
            return cached
        data = await self._fetch_uncached(url)
        self._bytes_cache[url] = data
        return data

    async def ensure_data_url(self, url: str, mime_type: str) -> str:
        """Return the ``data:`` form of `url`'s bytes, encoding and caching it on first use.

        For adapters whose provider consumes base64 rather than raw bytes, so
        the encode happens once, off the event loop, rather than on every
        context conversion.

        Args:
            url: A URL whose bytes are cached or fetchable — see :meth:`fetch`.
            mime_type: The file's MIME type, used in the data URL.

        Raises:
            FileResolverError: If the bytes can't be fetched.
        """
        if url.startswith("data:"):
            return url
        cached = self._data_url_cache.get(url)
        if cached is None:
            data = await self.fetch(url)
            encoded = await asyncio.to_thread(lambda: base64.b64encode(data).decode("utf-8"))
            cached = f"data:{mime_type};base64,{encoded}"
            self._data_url_cache[url] = cached
        return cached

    async def _fetch_uncached(self, url: str) -> bytes:
        if url.startswith("data:"):
            return await asyncio.to_thread(self._decode_data_url, url)
        if url.startswith(("http://", "https://")):
            return await self._fetch_http(url)
        if self._file_storage is not None:
            try:
                data = await self._file_storage.load(url)
            except FileNotFoundError as e:
                raise FileResolverError(f"File not found in storage: {url!r}") from e
            # The bytes are about to be cached, so a backend whose URLs are
            # all uploads it owns has no further use for the stored copy.
            if self._file_storage.delete_after_load:
                await self._file_storage.delete(url)
            return data
        raise FileResolverError(
            f"No way to resolve file URL {url!r}: configure a FileResolver with a "
            "file_storage (or a custom fetch) that understands it."
        )

    @staticmethod
    def _decode_data_url(url: str) -> bytes:
        try:
            _, encoded = url.split("base64,", 1)
            return base64.b64decode(encoded)
        except (ValueError, TypeError) as e:
            raise FileResolverError(f"Malformed data URL: {e}") from e

    async def _fetch_http(self, url: str) -> bytes:
        reachability = await self.classify(url)
        if reachability == UrlReachability.BLOCKED:
            raise FileResolverError(f"Refusing to fetch unreachable URL: {url!r}")
        try:
            timeout = aiohttp.ClientTimeout(total=self._fetch_timeout_secs)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Redirects are not followed: re-validating the host on every
                # hop is more machinery than this needs, and a redirect into
                # unreachable address space is exactly what classify() above
                # is meant to rule out.
                async with session.get(url, allow_redirects=False) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        raise FileResolverError(
                            f"Refusing to follow redirect fetching file from URL: {url!r}"
                        )
                    response.raise_for_status()
                    if (
                        response.content_length is not None
                        and response.content_length > self._max_fetch_bytes
                    ):
                        raise FileResolverError(
                            f"File at URL exceeds {self._max_fetch_bytes} byte limit"
                        )
                    # StreamReader.read(n) returns whatever is buffered (up to
                    # n), not the whole body, so accumulate chunks and enforce
                    # the size cap as they arrive.
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        received += len(chunk)
                        if received > self._max_fetch_bytes:
                            raise FileResolverError(
                                f"File at URL exceeds {self._max_fetch_bytes} byte limit"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
        except TimeoutError as e:
            raise FileResolverError(f"Timed out fetching file from URL: {url!r}") from e
        except aiohttp.ClientError as e:
            raise FileResolverError(f"Failed to fetch file from URL {url!r}: {e}") from e

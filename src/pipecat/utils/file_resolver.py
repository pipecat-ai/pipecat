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

The resolver is configured on the LLM service (``file_resolver=...``), which
runs the resolution pass right before each completion — see
:meth:`~pipecat.adapters.base_llm_adapter.BaseLLMAdapter.resolve_file_items`.
"""

import asyncio
import base64
import ipaddress

import aiohttp

from pipecat.utils.file_storage import FileStorage
from pipecat.utils.security.ssrf import IPNetwork, UrlReachability, classify_url_reachability

_DEFAULT_MAX_FETCH_BYTES = 50 * 1024 * 1024
_DEFAULT_FETCH_TIMEOUT_SECS = 30


class FileResolverError(Exception):
    """Raised when a file URL in the LLM context cannot be resolved to bytes."""

    pass


class FileResolver:
    """Fetches the bytes behind a file URL that the LLM provider can't fetch itself.

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
                ``pipecat:<id>`` from the development runner's ``POST /files``
                endpoint — pass ``runner_file_storage()`` there). Without it,
                only ``data:`` and ``http(s)`` URLs can be resolved.
            allowed_url_networks: CIDR ranges (e.g. ``["10.0.0.0/8"]``) this
                server is trusted to fetch from, in addition to the public
                internet. An ``http(s)`` URL that resolves outside both is
                refused.
            max_fetch_bytes: Maximum size of a fetched file.
            fetch_timeout_secs: Total timeout for an ``http(s)`` fetch.
        """
        self._file_storage = file_storage
        self._allowed_url_networks: list[IPNetwork] = [
            ipaddress.ip_network(net) for net in (allowed_url_networks or [])
        ]
        self._max_fetch_bytes = max_fetch_bytes
        self._fetch_timeout_secs = fetch_timeout_secs

    @property
    def file_storage(self) -> FileStorage | None:
        """The storage backend used to resolve URLs it minted, if any."""
        return self._file_storage

    async def classify(self, url: str) -> UrlReachability:
        """Classify who can reach an ``http(s)`` URL, honoring ``allowed_url_networks``."""
        return await classify_url_reachability(url, self._allowed_url_networks)

    async def fetch(self, url: str) -> bytes:
        """Return the bytes behind `url`.

        Args:
            url: A ``data:`` URL, an ``http(s)`` URL, or a URL minted by the
                configured storage backend. A storage-resolved file is deleted
                after a successful load when the backend sets
                ``delete_after_load`` (the caller inlines the bytes, so the
                stored copy is no longer needed).

        Raises:
            FileResolverError: If the URL is refused by the reachability
                policy, exceeds the size limit, fails to download, or has a
                scheme nothing is configured to resolve.
        """
        if url.startswith("data:"):
            return await asyncio.to_thread(self._decode_data_url, url)
        if url.startswith(("http://", "https://")):
            return await self._fetch_http(url)
        if self._file_storage is not None:
            try:
                data = await self._file_storage.load(url)
            except FileNotFoundError as e:
                raise FileResolverError(f"File not found in storage: {url!r}") from e
            # The caller inlines the bytes into the LLM context, so a backend
            # whose URLs are all uploads it owns has no further use for the
            # stored copy.
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
        if reachability == "blocked":
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
                    data = await response.content.read(self._max_fetch_bytes + 1)
                    if len(data) > self._max_fetch_bytes:
                        raise FileResolverError(
                            f"File at URL exceeds {self._max_fetch_bytes} byte limit"
                        )
                    return data
        except TimeoutError as e:
            raise FileResolverError(f"Timed out fetching file from URL: {url!r}") from e
        except aiohttp.ClientError as e:
            raise FileResolverError(f"Failed to fetch file from URL {url!r}: {e}") from e

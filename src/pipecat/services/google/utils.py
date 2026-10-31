#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility functions for Google services."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any, TypeVar

from loguru import logger

from pipecat import version as pipecat_version

T = TypeVar("T")


def update_google_client_http_options(http_options: dict[str, Any] | Any | None) -> Any:
    """Updates http_options with the x-goog-api-client header.

    Args:
        http_options: The existing http_options, which can be None, a dictionary,
                      or an object with a 'headers' attribute.

    Returns:
        The updated http_options.
    """
    client_header = {"x-goog-api-client": f"pipecat/{pipecat_version()}"}

    if http_options is None:
        http_options = {"headers": client_header}
    elif isinstance(http_options, dict):
        # Create a copy to avoid modifying the original dictionary if it's reused elsewhere
        http_options = http_options.copy()
        if "headers" in http_options:
            http_options["headers"].update(client_header)
        else:
            http_options["headers"] = client_header
    elif hasattr(http_options, "headers"):
        # We can't easily copy an arbitrary object, so we modify it in place.
        # This assumes the object is mutable and it's safe to do so.
        if http_options.headers is None:
            http_options.headers = client_header
        else:
            http_options.headers.update(client_header)

    return http_options


async def stream_with_retry(
    open_stream: Callable[[], Awaitable[AsyncIterator[T]]],
    *,
    idle_timeout: float | None,
    retry_timeout: float | None,
    retry_on_timeout: bool,
    timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,),
    label: str = "",
) -> AsyncGenerator[T, None]:
    """Yield a streamed response's chunks, re-issuing the request once if it stalls.

    A request that is accepted and then produces nothing shows up as a first
    chunk that never arrives. Re-issuing is safe only up to that point: once a
    chunk has been yielded, its content is already on its way downstream.

    Args:
        open_stream: Sends the request and returns its streamed response.
        idle_timeout: How long to wait for each chunk, and for the first one
            when the request isn't retried. ``None`` waits indefinitely.
        retry_timeout: How long to wait for the first chunk before re-issuing
            the request, when ``retry_on_timeout`` is set.
        retry_on_timeout: Whether to re-issue the request once if its first
            chunk doesn't arrive within ``retry_timeout``.
        timeout_errors: The errors that mean the first chunk timed out.
        label: Names the stream's owner in the retry's log line.

    Yields:
        Each chunk of the streamed response.

    Raises:
        TimeoutError: If the stream stalls past the timeout it is bounded by.
    """
    if retry_on_timeout:
        async with aclosing(
            iter_with_timeouts(
                await open_stream(), first_timeout=retry_timeout, idle_timeout=idle_timeout
            )
        ) as stream:
            try:
                first_chunk = await stream.__anext__()
            except StopAsyncIteration:
                return
            except timeout_errors:
                logger.debug(f"{label}: retrying content generation due to timeout")
            else:
                yield first_chunk
                async for chunk in stream:
                    yield chunk
                return

    # Either retries are disabled, or the initial attempt timed out.
    async with aclosing(
        iter_with_timeouts(
            await open_stream(), first_timeout=idle_timeout, idle_timeout=idle_timeout
        )
    ) as stream:
        async for chunk in stream:
            yield chunk


async def iter_with_timeouts(
    response: AsyncIterator[T],
    *,
    first_timeout: float | None,
    idle_timeout: float | None,
) -> AsyncGenerator[T, None]:
    """Yield streamed chunks, giving up if the stream stalls between them.

    A stream that stops producing without closing would otherwise leave the
    response open indefinitely, since the API clients apply no timeout of their
    own. The timeout covers the gap between chunks rather than the response as
    a whole, so a slow but healthy stream is never cut short.

    Args:
        response: The streamed response to consume.
        first_timeout: How long to wait for the first chunk.
        idle_timeout: How long to wait for every later chunk.

    Yields:
        Each chunk of the streamed response.

    Raises:
        TimeoutError: If the next chunk doesn't arrive in time.
    """
    chunks = response.__aiter__()
    timeout = first_timeout
    try:
        while True:
            try:
                if timeout is None:
                    chunk = await chunks.__anext__()
                else:
                    chunk = await asyncio.wait_for(chunks.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            yield chunk
            timeout = idle_timeout
    finally:
        # Release the HTTP resources held by a stream we walk away from, whether
        # that's because it stalled, was interrupted, or is being re-issued.
        await close_stream(response)


async def close_stream(stream: Any) -> None:
    """Close a streamed response, releasing its HTTP resources.

    Closing is best-effort: ``generateContent``'s stream type only promises
    ``AsyncIterator``, which has no ``aclose()``, and the Interactions API's
    stream closes with ``close()``.

    Args:
        stream: The streamed response.
    """
    for name in ("aclose", "close"):
        close = getattr(stream, name, None)
        if close is not None:
            await close()
            return

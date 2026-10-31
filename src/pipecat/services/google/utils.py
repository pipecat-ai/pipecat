#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility functions for Google services."""

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any, TypeVar

from loguru import logger

from pipecat import version as pipecat_version

T = TypeVar("T")


# The lowest thinking setting Google documents for each model, sent when the
# caller doesn't configure ``thinking``, keyed by model name. The 2.5 models
# take a budget, where 0 turns thinking off and 2.5 Pro's minimum is 128; later
# models take a level. 3.7 Flash, 3.8 Flash, 3.1 Pro and 4 Argon reject
# "minimal", and the Flash-Lite models already default to it. Aliases such as
# ``gemini-flash-latest`` are left out, since the model they name changes.
_THINKING_DEFAULTS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash": {"thinking_budget": 0},
    "gemini-2.5-flash-lite": {"thinking_budget": 0},
    "gemini-2.5-pro": {"thinking_budget": 128},
    "gemini-3-flash-preview": {"thinking_level": "minimal"},
    "gemini-3.1-flash-lite": {"thinking_level": "minimal"},
    "gemini-3.1-pro-preview": {"thinking_level": "low"},
    "gemini-3.1-pro-preview-customtools": {"thinking_level": "low"},
    "gemini-3.5-flash": {"thinking_level": "minimal"},
    "gemini-3.5-flash-lite": {"thinking_level": "minimal"},
    "gemini-3.6-flash": {"thinking_level": "minimal"},
    "gemini-3.7-flash": {"thinking_level": "low"},
    "gemini-3.8-flash": {"thinking_level": "low"},
    "gemini-4-argon": {"thinking_level": "low"},
    "gemma-4-26b-a4b-it": {"thinking_level": "minimal"},
    "gemma-4-31b-it": {"thinking_level": "minimal"},
}


def _model_name(model: str) -> str:
    """The name a model id refers to, for looking the model up.

    ``_THINKING_DEFAULTS`` and ``_MODELS_SUPPORTING_THINKING_LEVEL`` are keyed by
    name. Ids can name a model by resource path (``models/gemini-3.6-flash``,
    ``publishers/google/models/gemini-3.6-flash``) or by version
    (``gemini-2.5-flash-001``).
    """
    return re.sub(r"-\d{3}$", "", model.lower().rsplit("/", 1)[-1])


# Models that take their thinking configuration from thinking_level, keyed by
# model name prefix. Whether one of these honors a thinking_budget set alongside
# it varies by model and by backend, so such a budget can't be relied on.
_MODELS_SUPPORTING_THINKING_LEVEL = ("gemini-3", "gemini-4", "gemma-4")


def _can_think(name: str) -> bool:
    """Whether a model name belongs to a Gemini or Gemma text model that can think.

    Covers Gemini 2.5 and later, Gemma 4, and aliases such as
    ``gemini-flash-latest``. Image, speech and embedding models are excluded.
    """
    if any(kind in name for kind in ("image", "tts", "audio", "live", "embedding")):
        return False
    return bool(re.match(r"gemini-(2\.5|[3-9])|gemma-4|gemini-[a-z-]+-latest$", name))


def _thinking_default(
    model: str, *, levels_only: bool = False
) -> tuple[dict[str, Any] | None, bool]:
    """A model's thinking default, and whether it can think.

    Args:
        model: The model id, in any form ``_model_name`` recognizes.
        levels_only: Whether the API takes thinking as a level only. Gemini 2.5
            takes a budget, so such an API has no default to send it and no
            setting to suggest.

    Returns:
        The model's thinking config fields from ``_THINKING_DEFAULTS``, or None
        when it has none, and whether the model can think, which decides
        whether a model with no default is warned about.
    """
    name = _model_name(model)
    if levels_only and name.startswith("gemini-2.5"):
        return None, False
    return _THINKING_DEFAULTS.get(name), _can_think(name)


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
    stream closes with ``close()``, which leaves the event generator it reads
    from open. That generator is closed first, so it is finalized here rather
    than by the event loop's async generator hooks, which crash under uvloop
    on Python 3.12+ (MagicStack/uvloop#699).

    Args:
        stream: The streamed response.
    """
    generator = getattr(stream, "generator", None)
    if generator is not None and hasattr(generator, "aclose"):
        await generator.aclose()
    for name in ("aclose", "close"):
        close = getattr(stream, name, None)
        if close is not None:
            await close()
            return

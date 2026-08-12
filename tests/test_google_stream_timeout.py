#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the streamed-response timeouts and retry in GoogleLLMService."""

import asyncio
from unittest.mock import patch

import pytest
from google import genai
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    Part,
)

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService

# Short enough to keep tests quick, long enough to outlast scheduling jitter.
IDLE_TIMEOUT = 0.2
RETRY_TIMEOUT = 0.05


def _chunk(text=None, finish_reason=None) -> GenerateContentResponse:
    """Build a single streamed chunk, optionally carrying text."""
    parts = [Part(text=text)] if text is not None else []
    return GenerateContentResponse(
        candidates=[
            Candidate(content=Content(role="model", parts=parts), finish_reason=finish_reason)
        ]
    )


def _attempts(*generators):
    """Hand out one canned stream per attempt, recording the attempts made.

    Returns:
        A generator factory and the list of attempts it has served. The last
        generator is reused if the service asks for more streams than supplied.
    """
    attempts = []

    def factory():
        attempts.append(None)
        return generators[min(len(attempts), len(generators)) - 1]()

    return factory, attempts


async def _stream(generator, timeout=IDLE_TIMEOUT, **service_kwargs):
    """Run a context through the service against a canned stream.

    Returns:
        The pushed frames, any errors pushed upstream, and the names of any
        service events fired.
    """
    service = GoogleLLMService(
        api_key="test-key", stream_idle_timeout_secs=timeout, **service_kwargs
    )
    frames, errors, events = [], [], []

    async def capture_frame(frame, direction=None):
        frames.append(frame)

    async def capture_error(error):
        errors.append(error.error)

    @service.event_handler("on_completion_timeout")
    async def on_completion_timeout(_):
        events.append("on_completion_timeout")

    async def fake_stream(context):
        return generator()

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_stream_content", fake_stream),
        patch.object(service, "push_error_frame", capture_error),
    ):
        await service._process_context(LLMContext())

    # Event handlers run as background tasks, so give them a turn to finish.
    await asyncio.sleep(0.05)
    return frames, errors, events


@pytest.mark.asyncio
async def test_completed_stream_does_not_time_out():
    """A stream that closes normally ends on its own terms."""

    async def generator():
        yield _chunk("Hello there.")
        yield _chunk(finish_reason=FinishReason.STOP)

    frames, errors, events = await _stream(generator)

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Hello there."]
    assert errors == []
    assert events == []


@pytest.mark.asyncio
async def test_stalled_stream_times_out_and_closes_the_response():
    """A stream that stops producing without closing is bounded, not waited on forever."""

    async def generator():
        yield _chunk("Starting to answer")
        await asyncio.Event().wait()

    frames, errors, events = await _stream(generator)

    assert errors == ["LLM completion timeout"]
    assert events == ["on_completion_timeout"]
    # The text that did arrive is kept, and the response is still closed so the
    # aggregator downstream isn't left waiting on a turn that will never end.
    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Starting to answer"]
    assert isinstance(frames[-1], LLMFullResponseEndFrame)


@pytest.mark.asyncio
async def test_slow_stream_is_not_cut_short():
    """The timeout covers the gap between chunks, not the response as a whole."""

    async def generator():
        for i in range(5):
            await asyncio.sleep(IDLE_TIMEOUT / 2)
            yield _chunk(f"chunk{i} ")
        yield _chunk(finish_reason=FinishReason.STOP)

    frames, errors, events = await _stream(generator)

    # Total run outlasts the idle timeout several times over without tripping it.
    assert len([f for f in frames if isinstance(f, LLMTextFrame)]) == 5
    assert errors == []
    assert events == []


@pytest.mark.asyncio
async def test_timeout_can_be_disabled():
    """Passing None waits on the stream indefinitely."""

    async def generator():
        await asyncio.sleep(IDLE_TIMEOUT * 3)
        yield _chunk("Worth the wait.")
        yield _chunk(finish_reason=FinishReason.STOP)

    frames, errors, events = await _stream(generator, timeout=None)

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Worth the wait."]
    assert errors == []
    assert events == []


@pytest.mark.asyncio
async def test_stalled_first_chunk_is_not_retried_by_default():
    """Re-issuing a request is opt-in, since the wait covers the model's thinking."""

    async def stalled():
        await asyncio.Event().wait()
        yield  # pragma: no cover - unreachable, keeps this an async generator

    factory, attempts = _attempts(stalled)
    _, errors, events = await _stream(factory)

    assert len(attempts) == 1
    assert errors == ["LLM completion timeout"]
    assert events == ["on_completion_timeout"]


@pytest.mark.asyncio
async def test_stalled_first_chunk_is_retried():
    """A request that is accepted and then produces nothing is issued again."""

    async def stalled():
        await asyncio.Event().wait()
        yield  # pragma: no cover - unreachable, keeps this an async generator

    async def answered():
        yield _chunk("Second time lucky.")
        yield _chunk(finish_reason=FinishReason.STOP)

    factory, attempts = _attempts(stalled, answered)
    frames, errors, events = await _stream(
        factory, retry_on_timeout=True, retry_timeout_secs=RETRY_TIMEOUT
    )

    assert len(attempts) == 2
    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Second time lucky."]
    assert errors == []
    assert events == []


@pytest.mark.asyncio
async def test_stream_that_stalls_after_its_first_chunk_is_not_retried():
    """Once a chunk is downstream, re-issuing would duplicate the response."""

    async def generator():
        yield _chunk("Starting to answer")
        await asyncio.Event().wait()

    factory, attempts = _attempts(generator)
    frames, errors, events = await _stream(
        factory, retry_on_timeout=True, retry_timeout_secs=RETRY_TIMEOUT
    )

    assert len(attempts) == 1
    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Starting to answer"]
    assert errors == ["LLM completion timeout"]
    assert events == ["on_completion_timeout"]


@pytest.mark.asyncio
async def test_request_is_re_issued_only_once():
    """A second stall ends the turn rather than issuing a third request."""

    async def stalled():
        await asyncio.Event().wait()
        yield  # pragma: no cover - unreachable, keeps this an async generator

    factory, attempts = _attempts(stalled)
    _, errors, events = await _stream(
        factory, retry_on_timeout=True, retry_timeout_secs=RETRY_TIMEOUT
    )

    assert len(attempts) == 2
    assert errors == ["LLM completion timeout"]
    assert events == ["on_completion_timeout"]


@pytest.mark.asyncio
async def test_abandoned_stream_is_closed():
    """The stream left behind by a retry doesn't hold its HTTP resources open."""
    closed = []

    async def stalled():
        try:
            await asyncio.Event().wait()
            yield  # pragma: no cover - unreachable, keeps this an async generator
        finally:
            closed.append("stalled")

    async def answered():
        try:
            yield _chunk("Second time lucky.")
            yield _chunk(finish_reason=FinishReason.STOP)
        finally:
            closed.append("answered")

    factory, _ = _attempts(stalled, answered)
    await _stream(factory, retry_on_timeout=True, retry_timeout_secs=RETRY_TIMEOUT)

    assert closed == ["stalled", "answered"]


@pytest.mark.asyncio
async def test_client_stream_can_be_closed():
    """Closing an abandoned stream reaches the API client's own stream object.

    The tests above stand in canned async generators, which support ``aclose()``
    by construction. This one holds the real client to the same contract, since
    a stream that can't be closed is one whose HTTP resources leak on a retry.
    """
    client = genai.Client(api_key="test-key")

    # The request goes out when the first chunk is pulled, so the stream can be
    # created and closed without reaching the network.
    stream = await client.aio.models.generate_content_stream(
        model="gemini-2.5-flash", contents="Hello"
    )

    assert callable(getattr(stream, "aclose", None))
    await stream.aclose()


def test_timeouts_are_enabled_by_default():
    """The stream is bounded unless a caller opts out."""
    service = GoogleLLMService(api_key="test-key")

    assert service._stream_idle_timeout_secs == 20.0
    assert service._retry_timeout_secs == 5.0
    assert service._retry_on_timeout is False

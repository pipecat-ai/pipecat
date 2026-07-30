#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for OpenAI LLM error handling."""

import io
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from loguru import logger

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService


@pytest.mark.asyncio
async def test_openai_llm_emits_error_frame_on_timeout():
    """Test that OpenAI LLM service emits ErrorFrame when a timeout occurs.

    This enables LLMSwitcher to trigger failover to backup LLMs when the
    primary LLM times out.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        # Track pushed frames and errors
        pushed_frames = []
        pushed_errors = []
        timeout_handler_called = False

        original_push_frame = service.push_frame

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pushed_frames.append(frame)
            await original_push_frame(frame, direction)

        async def mock_push_error(error_msg, exception=None):
            pushed_errors.append({"error_msg": error_msg, "exception": exception})

        async def mock_timeout_handler(event_name):
            nonlocal timeout_handler_called
            if event_name == "on_completion_timeout":
                timeout_handler_called = True

        service.push_frame = mock_push_frame
        service.push_error = mock_push_error
        service._call_event_handler = AsyncMock(side_effect=mock_timeout_handler)

        # Mock _process_context to raise TimeoutException
        service._process_context = AsyncMock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        # Mock metrics methods
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()

        # Create a context frame to process
        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )
        frame = LLMContextFrame(context=context)

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify timeout handler was called
        service._call_event_handler.assert_called_once_with("on_completion_timeout")
        assert timeout_handler_called

        # Verify push_error was called with correct message
        assert len(pushed_errors) == 1
        assert pushed_errors[0]["error_msg"] == "LLM completion timeout"
        assert isinstance(pushed_errors[0]["exception"], httpx.TimeoutException)

        # Verify LLMFullResponseStartFrame and LLMFullResponseEndFrame were pushed
        frame_types = [type(f) for f in pushed_frames]
        assert LLMFullResponseStartFrame in frame_types
        assert LLMFullResponseEndFrame in frame_types


@pytest.mark.asyncio
async def test_openai_llm_timeout_still_pushes_end_frame():
    """Test that LLMFullResponseEndFrame is pushed even when timeout occurs.

    The finally block should ensure proper cleanup regardless of timeout.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        pushed_frames = []

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pushed_frames.append(frame)

        service.push_frame = mock_push_frame
        service.push_error = AsyncMock()
        service._call_event_handler = AsyncMock()
        service._process_context = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )
        frame = LLMContextFrame(context=context)

        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify both start and end frames are pushed
        frame_types = [type(f) for f in pushed_frames]
        assert LLMFullResponseStartFrame in frame_types
        assert LLMFullResponseEndFrame in frame_types

        # Verify metrics were stopped
        service.stop_processing_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_openai_llm_stream_closed_on_cancellation():
    """Test that the stream is closed when CancelledError occurs during iteration.

    This prevents socket leaks when the pipeline is interrupted (e.g., user interruption).
    See issue #3589.
    """
    import asyncio

    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        # Track if close was called
        stream_closed = False

        class MockAsyncStream:
            """Mock AsyncStream that tracks close() calls and raises CancelledError."""

            def __init__(self):
                self.iteration_count = 0

            async def close(self):
                nonlocal stream_closed
                stream_closed = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.iteration_count += 1
                if self.iteration_count > 1:
                    # Simulate cancellation during iteration
                    raise asyncio.CancelledError()
                # Return a minimal chunk for first iteration
                mock_chunk = AsyncMock()
                mock_chunk.usage = None
                mock_chunk.model = None
                mock_chunk.choices = []
                return mock_chunk

        mock_stream = MockAsyncStream()

        # Mock the stream creation methods
        service.get_chat_completions = AsyncMock(return_value=mock_stream)
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Process context should raise CancelledError but stream should still be closed
        with pytest.raises(asyncio.CancelledError):
            await service._process_context(context)

        # Verify stream was closed despite the cancellation
        assert stream_closed, "Stream should be closed even when CancelledError occurs"


@pytest.mark.asyncio
async def test_openai_llm_emits_error_frame_on_exception():
    """Test that OpenAI LLM service emits ErrorFrame when a general exception occurs.

    This enables proper error handling for API errors, rate limits, and other failures.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        pushed_errors = []

        async def mock_push_error(error_msg, exception=None):
            pushed_errors.append({"error_msg": error_msg, "exception": exception})

        service.push_frame = AsyncMock()
        service.push_error = mock_push_error
        service._call_event_handler = AsyncMock()
        service._process_context = AsyncMock(side_effect=RuntimeError("API Error"))
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )
        frame = LLMContextFrame(context=context)

        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify push_error was called with correct message
        assert len(pushed_errors) == 1
        assert "Error during completion" in pushed_errors[0]["error_msg"]
        assert "API Error" in pushed_errors[0]["error_msg"]
        assert isinstance(pushed_errors[0]["exception"], RuntimeError)


@pytest.mark.asyncio
async def test_openai_llm_async_iterator_closed_on_stream_end():
    """Test that the async iterator is explicitly closed after stream consumption.

    This prevents uvloop's broken asyncgen finalizer from firing on Python 3.12+
    when async generators are garbage-collected without explicit cleanup.
    See MagicStack/uvloop#699.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        # Track if the iterator's aclose was called
        iterator_aclosed = False
        stream_closed = False

        class MockAsyncIterator:
            """Mock async iterator that tracks aclose() calls."""

            def __init__(self):
                self.iteration_count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.iteration_count += 1
                if self.iteration_count > 2:
                    raise StopAsyncIteration()
                # Return a minimal chunk
                mock_chunk = AsyncMock()
                mock_chunk.usage = None
                mock_chunk.model = None
                mock_chunk.choices = []
                return mock_chunk

            async def aclose(self):
                nonlocal iterator_aclosed
                iterator_aclosed = True

        class MockAsyncStream:
            """Mock stream whose __aiter__ returns a separate iterator object."""

            def __init__(self, iterator):
                self._iterator = iterator

            def __aiter__(self):
                return self._iterator

            async def close(self):
                nonlocal stream_closed
                stream_closed = True

        mock_iterator = MockAsyncIterator()
        mock_stream = MockAsyncStream(mock_iterator)

        service.get_chat_completions = AsyncMock(return_value=mock_stream)
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        await service._process_context(context)

        # Verify the iterator was explicitly closed (prevents uvloop crash)
        assert iterator_aclosed, "Async iterator should be explicitly closed"
        # Verify the stream was also closed (releases HTTP resources)
        assert stream_closed, "Stream should be closed to release HTTP resources"


class _EmptyStream:
    """A completion stream that yields nothing — i.e. an empty completion."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration()

    async def aclose(self):
        pass

    async def close(self):
        pass


class _FakeClock:
    """Manually advanced monotonic clock; reading it never moves it.

    Advanced explicitly by the fake generation rather than per read, so the
    assertions below depend on how much wall clock each *generation* costs and
    not on how many times the implementation happens to read the clock.
    """

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


def _make_empty_completion_service(clock: _FakeClock | None = None, secs_per_generation: float = 0):
    """A service whose every generation comes back empty, with attempts counted.

    The clock is injected on the service instead of monkeypatching the stdlib
    ``time`` module, which is process-global and would also replace asyncio's
    event-loop clock for the duration of the test.
    """
    from pipecat.services.openai.llm import OpenAILLMService

    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
    service._client = AsyncMock()
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()

    clock = clock or _FakeClock()
    service._generation_clock = clock.monotonic

    def generate(_ctx):
        clock.advance(secs_per_generation)
        return _EmptyStream()

    service.get_chat_completions = AsyncMock(side_effect=generate)
    return service


@pytest.mark.asyncio
async def test_empty_completion_retries_all_run_when_the_budget_is_ample():
    """A healthy-but-empty server still gets the full retry ladder."""
    from pipecat.services.openai import base_llm

    # Generations are instant: the budget can never be the limiting factor.
    service = _make_empty_completion_service(secs_per_generation=0)
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])

    await service._process_context(context)

    # Initial generation + MAX_EMPTY_COMPLETION_RETRIES.
    assert service.get_chat_completions.await_count == base_llm.MAX_EMPTY_COMPLETION_RETRIES + 1


@pytest.mark.asyncio
async def test_empty_completion_retry_ladder_is_cut_off_by_the_wall_clock():
    """A slow server stops the ladder early instead of stacking timeouts.

    The three retry layers (empty-completion recursion, the connection retry and
    the SDK's own max_retries) multiply, so without an aggregate wall-clock bound
    a degraded server can hold one live turn open for minutes of dead air.
    """
    from pipecat.services.openai import base_llm

    # Each generation burns 4s of a 6s budget: the first retry still fits, the
    # second cannot, so the ladder must stop short of the depth cap.
    secs_per_generation = base_llm.EMPTY_RETRY_TOTAL_BUDGET_SECS * 2 / 3
    service = _make_empty_completion_service(secs_per_generation=secs_per_generation)
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])

    log_output = io.StringIO()
    handler_id = logger.add(log_output, level="WARNING", format="{message}")
    try:
        await service._process_context(context)
    finally:
        logger.remove(handler_id)

    assert service.get_chat_completions.await_count == 2
    assert service.get_chat_completions.await_count < base_llm.MAX_EMPTY_COMPLETION_RETRIES + 1
    # Distinct from the depth-cap path, so the exhaustion rate is measurable
    # before anyone retunes the budget.
    assert "retry budget exhausted" in log_output.getvalue()


@pytest.mark.asyncio
async def test_client_is_bounded_by_default():
    """The shared AsyncOpenAI client must carry explicit retry/timeout bounds.

    SDK defaults are max_retries=2 and timeout=600s. On a voice pipeline that is
    a ten-minute dead-air turn, and it applies to every OpenAI-compatible
    provider that inherits create_client (which is most of them).
    """
    from pipecat.services.openai import base_llm
    from pipecat.services.openai.llm import OpenAILLMService

    service = OpenAILLMService(
        api_key="test-key", settings=OpenAILLMService.Settings(model="gpt-4")
    )
    assert service._client.max_retries == base_llm.CLIENT_MAX_RETRIES
    assert service._client.timeout.read == base_llm.CLIENT_READ_TIMEOUT_SECS
    assert service._client.timeout.connect == base_llm.CLIENT_CONNECT_TIMEOUT_SECS


@pytest.mark.asyncio
async def test_speaches_inherits_the_bounded_client():
    """The provider that carried 100% of production traffic must be bounded too.

    SpeachesLLMService does not override create_client, so before this it ran on
    stock AsyncOpenAI defaults while the fail-fast policy lived in an
    application-side subclass of the OpenAI provider, which carried no traffic.
    """
    from pipecat.services.openai import base_llm
    from pipecat.services.speaches.llm import SpeachesLLMService

    service = SpeachesLLMService(api_key="none", base_url="http://localhost:11434/v1")
    assert service._client.max_retries == base_llm.CLIENT_MAX_RETRIES
    assert service._client.timeout.read == base_llm.CLIENT_READ_TIMEOUT_SECS


@pytest.mark.asyncio
async def test_client_bounds_are_overridable():
    """Long-running non-voice consumers must be able to widen the bounds."""
    from pipecat.services.openai.llm import OpenAILLMService

    service = OpenAILLMService(
        api_key="test-key",
        settings=OpenAILLMService.Settings(model="gpt-4"),
        client_max_retries=4,
        client_timeout=httpx.Timeout(connect=1.0, read=120.0, write=1.0, pool=1.0),
    )
    assert service._client.max_retries == 4
    assert service._client.timeout.read == 120.0


@pytest.mark.asyncio
async def test_run_inference_is_not_capped_by_the_conversational_read_timeout():
    """Out-of-band inference must not inherit the live-turn read bound.

    ``run_inference`` is non-streaming, so the client's ``read`` timeout would
    bound *total generation* rather than time-to-first-byte. A transcript
    summary or a slot extraction on a self-hosted model can easily exceed the
    30s that is correct for a first token on a live turn.
    """
    from pipecat.services.openai import base_llm
    from pipecat.services.openai.llm import OpenAILLMService

    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))

    captured = {}

    async def fake_create(**params):
        captured.update(params)
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()

    service._client = AsyncMock()
    service._client.chat.completions.create = AsyncMock(side_effect=fake_create)

    context = LLMContext(messages=[{"role": "user", "content": "Summarize"}])
    result = await service.run_inference(context)

    assert result == "ok"
    assert captured["stream"] is False
    assert captured["timeout"] == base_llm.INFERENCE_TIMEOUT_SECS
    # ...and it is much looser than the conversational bound, which is the point.
    assert captured["timeout"] > base_llm.CLIENT_READ_TIMEOUT_SECS


@pytest.mark.asyncio
async def test_run_inference_timeout_is_overridable():
    """Deployments with slower out-of-band models must be able to widen it."""
    from pipecat.services.openai.llm import OpenAILLMService

    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(model="gpt-4"),
            inference_timeout_secs=42.0,
        )

    captured = {}

    async def fake_create(**params):
        captured.update(params)
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()

    service._client = AsyncMock()
    service._client.chat.completions.create = AsyncMock(side_effect=fake_create)

    await service.run_inference(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

    assert captured["timeout"] == 42.0

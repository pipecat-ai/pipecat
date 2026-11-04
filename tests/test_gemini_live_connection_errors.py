#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how GeminiLiveLLMService reports Live connection failures.

The service must report a session that fails to open, a connection that keeps
failing and a failed send as errors the pipeline can act on, so
``ProcessorUnusablePolicy`` applies instead of the bot staying silent.

A fake stands in for google-genai's ``client.aio.live``: each ``connect()``
attempt either raises or yields a scripted session, and sessions raise the
same exceptions google-genai and websockets raise in production (``APIError``
for a Live socket closed with an error code, ``ConnectionClosedError`` for a
send on a closed socket). No network or API key is needed.

A connection that drops while the bot is speaking must also end that reply, as
the server will never complete it.
"""

import asyncio
import contextlib
import time
import unittest
from collections.abc import Callable
from typing import Any

from google.genai import errors
from google.genai.types import (
    Blob,
    Content,
    LiveConnectConfig,
    LiveServerContent,
    LiveServerMessage,
    Part,
)
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy, WorkerParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.gemini_live.llm import MAX_CONSECUTIVE_FAILURES, GeminiLiveLLMService
from pipecat.utils.asyncio.task_manager import TaskManager

RUN_TIMEOUT_SECS = 5

# Replaces the service's 30 s EndFrame deferral in the mid-reply tests, so a
# deferred EndFrame shows up as a slow end rather than a hung test.
END_FRAME_DEFERRAL_SECS = 2.0


def _api_error(code: int, reason: str) -> errors.APIError:
    """What google-genai raises when the Live socket closes with ``code``."""
    try:
        errors.APIError.raise_error(code, reason, None)
    except errors.APIError as e:
        return e
    raise AssertionError("APIError.raise_error did not raise")


def _setup_rejected() -> errors.APIError:
    """What google-genai raises when the server rejects the session setup."""
    return _api_error(
        1008, "models/gemini-x is not found for API version v1beta, or is not supported"
    )


def _connection_closed() -> ConnectionClosedError:
    """What websockets raises when sending on a socket the server closed."""
    return ConnectionClosedError(Close(1011, "internal error"), None)


# A short chunk of bot audio, which starts a bot reply (TTSStartedFrame).
_BOT_AUDIO = LiveServerMessage(
    server_content=LiveServerContent(
        model_turn=Content(
            parts=[Part(inline_data=Blob(data=b"\0\0" * 2400, mime_type="audio/pcm;rate=24000"))]
        )
    )
)


class _FakeSession:
    """A Live session that yields ``messages`` and then drops or stays open.

    Args:
        messages: Server messages to deliver first.
        drop: Exception to raise from ``receive()`` after the messages; when
            None the session stays open.
        drop_when: If given, wait for this event before dropping.
        send_error: Exception to raise from every send; when None sends succeed.
    """

    def __init__(
        self,
        messages: list[LiveServerMessage] | None = None,
        drop: Exception | None = None,
        drop_when: asyncio.Event | None = None,
        send_error: Exception | None = None,
    ):
        self._messages = list(messages or [])
        self._drop = drop
        self._drop_when = drop_when
        self._send_error = send_error

    async def receive(self):
        while self._messages:
            yield self._messages.pop(0)
        if self._drop:
            if self._drop_when:
                await self._drop_when.wait()
            raise self._drop
        await asyncio.Event().wait()

    async def send_client_content(self, **kwargs):
        if self._send_error:
            raise self._send_error

    async def send_realtime_input(self, **kwargs):
        if self._send_error:
            raise self._send_error

    async def send_tool_response(self, **kwargs):
        if self._send_error:
            raise self._send_error

    async def close(self):
        pass


class _FakeLive:
    """Stands in for ``client.aio.live``.

    ``script(attempt)`` is called for each connect attempt (starting at 1) and
    returns the session to open, or the exception opening it raises.
    """

    def __init__(self, script: Callable[[int], _FakeSession | Exception]):
        self._script = script
        self.attempts = 0

    @contextlib.asynccontextmanager
    async def connect(self, *, model, config=None):
        self.attempts += 1
        step = self._script(self.attempts)
        if isinstance(step, Exception):
            raise step
        yield step


class _FakeClient:
    def __init__(self, live: _FakeLive):
        self.aio = type("_FakeAio", (), {"live": live})()


class _FrameCollector(FrameProcessor):
    """Records the frames the service pushes downstream."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)

    def count(self, frame_type: type[Frame]) -> int:
        return sum(isinstance(f, frame_type) for f in self.frames)


class _PipelineTestCase(unittest.IsolatedAsyncioTestCase):
    """Runs the service in a pipeline with a fake Live client.

    The worker uses ``ProcessorUnusablePolicy.END``, so a failure the pipeline
    can act on shows up as an unusable-processor error and ends the run.
    """

    def setUp(self):
        self.collector = _FrameCollector()
        self.unusable: list[ErrorFrame] = []

    def make_worker(self, script: Callable[[int], _FakeSession | Exception]):
        self.live = _FakeLive(script)
        self.llm = GeminiLiveLLMService(api_key="test-key")
        self.llm._client = _FakeClient(self.live)

        worker = PipelineWorker(
            Pipeline([self.llm, self.collector]),
            processor_unusable_policy=ProcessorUnusablePolicy.END,
        )

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            if frame.processor and not frame.processor.is_usable:
                self.unusable.append(frame)

        return worker

    async def run_until_ended(self, worker: PipelineWorker, failure: str):
        """Run the worker; fail with ``failure`` if it hasn't ended in time."""
        try:
            async with asyncio.timeout(RUN_TIMEOUT_SECS):
                await worker.run(WorkerParams(task_manager=TaskManager()))
        except TimeoutError:
            self.fail(f"{failure} (pipeline still running after {RUN_TIMEOUT_SECS}s)")

    async def wait_for(self, predicate: Callable[[], bool], what: str):
        try:
            async with asyncio.timeout(RUN_TIMEOUT_SECS):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            self.fail(f"timed out waiting for {what}")

    def assert_service_unusable(self):
        self.assertFalse(self.llm.is_usable, "service should be marked unusable")
        self.assertEqual(len(self.unusable), 1, "expected one unusable-processor error")
        self.assertIs(self.unusable[0].processor, self.llm)


class TestConnectionErrorsEndThePipeline(_PipelineTestCase):
    """A failure the service can't recover from ends the pipeline."""

    async def test_session_that_fails_to_open_is_retried_then_marks_service_unusable(self):
        """The server rejects the setup (1008), so no session ever opens."""
        worker = self.make_worker(lambda attempt: _setup_rejected())

        await self.run_until_ended(worker, "a session that fails to open should end the pipeline")

        self.assertEqual(self.live.attempts, MAX_CONSECUTIVE_FAILURES)
        self.assert_service_unusable()

    async def test_giving_up_after_max_failures_marks_service_unusable(self):
        """Every session drops right away (1011) until the service gives up."""
        worker = self.make_worker(
            lambda attempt: _FakeSession(drop=_api_error(1011, "Internal error encountered."))
        )

        await self.run_until_ended(worker, "giving up on reconnecting should end the pipeline")

        self.assertEqual(self.live.attempts, MAX_CONSECUTIVE_FAILURES)
        self.assert_service_unusable()

    async def test_send_error_marks_service_unusable(self):
        """Sending the initial context fails because the socket is gone."""
        worker = self.make_worker(lambda attempt: _FakeSession(send_error=_connection_closed()))
        context = LLMContext(messages=[{"role": "user", "content": "Hello!"}])

        async def send_context_once_connected():
            await self.wait_for(lambda: self.llm._session is not None, "the session to open")
            await worker.queue_frame(LLMContextFrame(context=context))

        sender = asyncio.create_task(send_context_once_connected())
        try:
            await self.run_until_ended(worker, "a send error should end the pipeline")
        finally:
            await sender

        self.assert_service_unusable()


class TestConnectionDroppedMidReply(_PipelineTestCase):
    """A connection that drops while the bot is speaking ends that reply."""

    def assert_reply_ended(self):
        self.assertEqual(self.collector.count(TTSStartedFrame), 1)
        self.assertEqual(
            self.collector.count(TTSStoppedFrame), 1, "the cut-off reply was never ended"
        )
        self.assertEqual(self.collector.count(LLMFullResponseEndFrame), 1)
        self.assertFalse(self.llm._bot_is_responding)

    async def test_drop_mid_reply_ends_the_reply(self):
        """The session drops while the bot is speaking, and the reconnect succeeds."""

        def script(attempt: int) -> _FakeSession:
            if attempt == 1:
                return _FakeSession(
                    messages=[_BOT_AUDIO], drop=_api_error(1011, "Internal error encountered.")
                )
            return _FakeSession()

        worker = self.make_worker(script)
        # Shorten the deferral (30 s by default) so a deferred EndFrame is
        # released before the run times out, and measure how long it took.
        self.llm._END_FRAME_DEFERRAL_TIMEOUT_SECS = END_FRAME_DEFERRAL_SECS
        end_queued_at = None

        async def end_once_reconnected():
            nonlocal end_queued_at
            await self.wait_for(lambda: self.live.attempts >= 2, "the reconnect")
            await self.wait_for(lambda: self.llm._session is not None, "the new session")
            end_queued_at = time.monotonic()
            await worker.queue_frame(EndFrame())

        ender = asyncio.create_task(end_once_reconnected())
        try:
            await self.run_until_ended(worker, "the pipeline should end after the EndFrame")
        finally:
            await ender

        self.assertLess(
            time.monotonic() - end_queued_at,
            END_FRAME_DEFERRAL_SECS / 2,
            "the EndFrame was deferred waiting for the cut-off reply to finish",
        )
        self.assert_reply_ended()
        self.assertTrue(self.llm.is_usable, "a successful reconnect should keep the service")

    async def test_end_frame_held_for_the_reply_is_released_when_it_drops(self):
        """An EndFrame waits for the bot's reply, then the session drops."""
        drop = asyncio.Event()

        def script(attempt: int) -> _FakeSession:
            if attempt == 1:
                return _FakeSession(
                    messages=[_BOT_AUDIO],
                    drop=_api_error(1011, "Internal error encountered."),
                    drop_when=drop,
                )
            return _FakeSession()

        worker = self.make_worker(script)
        # Shorten the deferral (30 s by default): if the held EndFrame is lost,
        # it is handed back to the service so the run can still end, and its
        # deferral must run out before the run times out.
        self.llm._END_FRAME_DEFERRAL_TIMEOUT_SECS = END_FRAME_DEFERRAL_SECS
        released = False

        async def end_then_drop():
            nonlocal released
            await self.wait_for(lambda: self.llm._bot_is_responding, "the bot to start replying")
            end_frame = EndFrame()
            await worker.queue_frame(end_frame)
            await self.wait_for(
                lambda: self.llm._end_frame_pending_bot_turn_finished is not None,
                "the EndFrame to be deferred",
            )
            drop.set()
            try:
                async with asyncio.timeout(END_FRAME_DEFERRAL_SECS / 2):
                    while not self.collector.count(EndFrame):
                        await asyncio.sleep(0.01)
                released = True
            except TimeoutError:
                # The worker is already closing and won't take a new EndFrame.
                await self.llm.queue_frame(end_frame)

        dropper = asyncio.create_task(end_then_drop())
        try:
            await self.run_until_ended(worker, "the pipeline should end after the EndFrame")
        finally:
            await dropper

        self.assertTrue(released, "the EndFrame held for the cut-off reply was discarded")
        self.assert_reply_ended()
        self.assertEqual(self.live.attempts, 1, "no reconnect while the pipeline is ending")


class TestErrorsWhileDisconnecting(unittest.IsolatedAsyncioTestCase):
    """Failures caused by our own disconnect are expected and not reported."""

    def setUp(self):
        self.llm = GeminiLiveLLMService(api_key="test-key")
        self.llm._disconnecting = True
        self.pushed: list[dict[str, Any]] = []
        self.reconnects = 0

        async def push_error(**kwargs):
            self.pushed.append(kwargs)

        async def reconnect():
            self.reconnects += 1

        self.llm.push_error = push_error  # type: ignore[method-assign]
        self.llm._reconnect = reconnect  # type: ignore[method-assign]

    async def test_failure_to_open_session_is_ignored(self):
        self.llm._client = _FakeClient(_FakeLive(lambda attempt: _setup_rejected()))

        await self.llm._connection_task_handler(config=LiveConnectConfig())

        self.assertEqual(self.pushed, [])
        self.assertEqual(self.reconnects, 0)
        self.assertEqual(self.llm._consecutive_failures, 0)

    async def test_send_error_is_ignored(self):
        self.llm._session = _FakeSession()

        await self.llm._handle_send_error(_connection_closed())

        self.assertEqual(self.pushed, [])


if __name__ == "__main__":
    unittest.main()

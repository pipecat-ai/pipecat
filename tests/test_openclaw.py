#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenClaw Gateway support."""

import asyncio
import unittest

from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.openclaw.client import (
    OpenClawError,
    OpenClawGatewayClient,
    collect_result,
)
from pipecat.services.openclaw.frames import (
    OpenClawAbortFrame,
    OpenClawEndFrame,
    OpenClawSendFrame,
    OpenClawStartedFrame,
    OpenClawSteerFrame,
    OpenClawTextFrame,
)
from pipecat.services.openclaw.gateway import OpenClawGatewayService
from pipecat.tests.utils import SleepFrame, run_test
from tests.openclaw_fake_gateway import FakeGateway


class TestOpenClawGatewayClient(unittest.IsolatedAsyncioTestCase):
    """The Gateway client, driven against a real websocket."""

    async def asyncSetUp(self):
        self.gateway = await FakeGateway().__aenter__()
        self.client = OpenClawGatewayClient(url=self.gateway.url, token="test-token")

    async def asyncTearDown(self):
        await self.client.disconnect()
        await self.gateway.__aexit__(None, None, None)

    async def _next(self, events, timeout=2.0):
        return await asyncio.wait_for(events.__anext__(), timeout=timeout)

    async def _eventually(self, predicate, timeout=2.0):
        async def _wait():
            while not predicate():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait(), timeout=timeout)

    #
    # Handshake
    #

    async def test_handshake_answers_the_challenge(self):
        """The client connects by answering connect.challenge, not by asking."""
        await self.client.connect()

        params = self.gateway.params("connect")
        self.assertEqual(params["minProtocol"], 4)
        self.assertEqual(params["maxProtocol"], 4)
        self.assertEqual(params["auth"], {"token": "test-token"})

    async def test_the_client_identifies_as_one_the_gateway_knows(self):
        """The Gateway validates the client id against a closed set."""
        await self.client.connect()

        client = self.gateway.params("connect")["client"]
        self.assertEqual(client["id"], "gateway-client")
        self.assertEqual(client["mode"], "backend")
        self.assertEqual(client["displayName"], "pipecat")

    async def test_a_rejected_token_fails_the_connection(self):
        self.gateway.errors["connect"] = {"code": "unauthorized", "message": "bad token"}

        with self.assertRaises(OpenClawError) as caught:
            await self.client.connect()
        self.assertEqual(caught.exception.code, "unauthorized")

    #
    # Runs
    #

    async def test_the_message_is_sent_verbatim(self):
        """The client adds no framing of its own to what the caller sends."""
        await self.client.start("What changed in the parser?")

        params = self.gateway.params("chat.send")
        self.assertEqual(params["message"], "What changed in the parser?")
        self.assertEqual(params["sessionKey"], "agent:main:main")

    async def test_a_run_streams_deltas_and_completes(self):
        run = await self.client.start("hello")
        events = self.client.events(run)

        await self.gateway.chat(run.run_id, "delta", "one ")
        await self.gateway.chat(run.run_id, "delta", "two")
        await self.gateway.chat(run.run_id, "final", "one two")

        self.assertEqual((await self._next(events)).text, "one ")
        self.assertEqual((await self._next(events)).text, "two")
        completed = await self._next(events)
        self.assertEqual(completed.kind, "completed")
        with self.assertRaises(StopAsyncIteration):
            await self._next(events)

    async def test_collect_result_folds_a_run_into_one_answer(self):
        run = await self.client.start("hello")
        collected = asyncio.create_task(collect_result(self.client.events(run)))
        await asyncio.sleep(0)

        await self.gateway.chat(run.run_id, "delta", "partial")
        await self.gateway.chat(run.run_id, "final", "the answer")

        result = await asyncio.wait_for(collected, timeout=2)
        self.assertEqual(result.summary, "the answer")
        self.assertEqual(result.status, "completed")

    async def test_an_error_state_fails_the_run(self):
        run = await self.client.start("hello")
        events = self.client.events(run)

        await self.gateway.chat(run.run_id, "error", error_message="the model is unavailable")

        event = await self._next(events)
        self.assertEqual(event.kind, "failed")
        self.assertEqual(event.text, "the model is unavailable")

    async def test_a_dropped_connection_fails_the_run(self):
        """A socket that dies mid-run ends the stream instead of hanging."""
        run = await self.client.start("hello")
        events = self.client.events(run)

        await self.gateway.drop()

        event = await self._next(events)
        self.assertEqual(event.kind, "failed")
        self.assertIn("closed", event.text)

    async def test_the_socket_comes_back_after_a_drop(self):
        """A run cannot survive the drop, but the connection under it can."""
        run = await self.client.start("hello")
        events = self.client.events(run)

        await self.gateway.drop()
        self.assertEqual((await self._next(events)).kind, "failed")
        await self._eventually(lambda: self.gateway.count("connect") == 2)

        run = await self.client.start("again")
        events = self.client.events(run)
        await self.gateway.chat(run.run_id, "final", "back")
        self.assertEqual((await self._next(events)).text, "back")

    async def test_frames_arriving_before_the_run_id_is_known_are_not_lost(self):
        """The Gateway can name a run something other than our idempotency key.

        Frames carrying that id can arrive before the response that names it,
        so they are held and replayed rather than dropped.
        """
        self.gateway.run_id = "gateway-chose-this"

        async def stream_early(frame):
            if frame.get("method") == "chat.send":
                await self.gateway.chat("gateway-chose-this", "delta", "early")

        self.gateway.on_request = stream_early

        run = await self.client.start("hello")
        self.assertEqual(run.run_id, "gateway-chose-this")

        events = self.client.events(run)
        self.assertEqual((await self._next(events)).text, "early")

    #
    # Steering
    #

    async def test_steering_follows_the_replacement_run(self):
        """A steer aborts the live run and starts a new one carrying the follow-up.

        The old run's aborted frame can arrive after the steer. If the stream
        still answered to it, the caller would be told their task was cancelled
        while the replacement carried on unwatched.
        """
        run = await self.client.start("research NVFP4")
        first_run_id = run.run_id
        events = self.client.events(run)

        await self.client.steer(run, "actually, compare it to FP8")
        self.assertNotEqual(run.run_id, first_run_id)
        self.assertTrue(self.gateway.params("sessions.steer")["message"].startswith("actually"))

        await self.gateway.chat(first_run_id, "aborted", "interrupted")
        await self.gateway.chat(run.run_id, "delta", "comparing")
        await self.gateway.chat(run.run_id, "final", "FP8 wins on throughput")

        self.assertEqual((await self._next(events)).text, "comparing")
        completed = await self._next(events)
        self.assertEqual(completed.kind, "completed")
        self.assertEqual(completed.text, "FP8 wins on throughput")

    async def test_a_failed_steer_leaves_the_run_reachable(self):
        """The run keeps answering to its own ids when a steer does not land.

        The steer rekeys the run onto the replacement before asking for it. If
        the request fails, the run the Gateway is still executing has to be
        routable again, or its terminal event would arrive for nobody and the
        stream would never end.
        """
        run = await self.client.start("research NVFP4")
        first_run_id = run.run_id
        events = self.client.events(run)
        self.gateway.errors["sessions.steer"] = {"code": "busy", "message": "session is busy"}

        with self.assertRaises(OpenClawError):
            await self.client.steer(run, "actually, compare it to FP8")
        self.assertEqual(run.run_id, first_run_id)

        await self.gateway.chat(first_run_id, "final", "NVFP4 halves the footprint")
        completed = await self._next(events)
        self.assertEqual(completed.kind, "completed")
        self.assertEqual(completed.text, "NVFP4 halves the footprint")

    #
    # Aborting
    #

    async def test_abort_reports_that_a_run_was_stopped(self):
        run = await self.client.start("hello")
        self.gateway.aborted = True

        self.assertTrue(await self.client.abort(run, "the user said stop"))
        self.assertEqual(self.gateway.params("chat.abort")["runId"], run.run_id)

    async def test_abort_reports_nothing_to_stop_rather_than_failing(self):
        """`aborted: false` is the routine race, not an error.

        The Gateway answers both a finished run and an unknown one this way.
        """
        run = await self.client.start("hello")
        self.gateway.aborted = False

        self.assertFalse(await self.client.abort(run))

    async def test_abort_works_after_the_stream_has_ended(self):
        """Cancellation reaches abort by way of the finished event stream.

        The connection outlives the run for exactly this reason: a request on a
        socket whose reader has stopped never gets a reply.
        """
        run = await self.client.start("hello")
        events = self.client.events(run)
        await self.gateway.chat(run.run_id, "final", "done")
        await self._next(events)
        with self.assertRaises(StopAsyncIteration):
            await self._next(events)

        self.gateway.aborted = False
        self.assertFalse(await asyncio.wait_for(self.client.abort(run), timeout=2))

    async def test_a_gateway_error_is_raised(self):
        self.gateway.errors["chat.send"] = {"code": "busy", "message": "session is busy"}

        with self.assertRaises(OpenClawError) as caught:
            await self.client.start("hello")
        self.assertEqual(caught.exception.code, "busy")
        self.assertIn("session is busy", str(caught.exception))

    async def test_one_connection_serves_every_run(self):
        """Runs reuse the connection instead of opening one each."""
        for _ in range(3):
            run = await self.client.start("hello")
            events = self.client.events(run)
            await self.gateway.chat(run.run_id, "final", "done")
            await self._next(events)

        self.assertEqual(self.gateway.count("connect"), 1)
        self.assertEqual(self.gateway.count("chat.send"), 3)

    async def test_a_run_fails_when_the_reader_gives_up(self):
        """A client that will not reconnect still ends its runs.

        The receive loop stops without closing the socket itself when
        reconnection is disabled, so the runs waiting on it are failed as the
        loop exits rather than left waiting for events that cannot arrive.
        """
        client = OpenClawGatewayClient(url=self.gateway.url, reconnect_on_error=False)
        self.addAsyncCleanup(client.disconnect)
        run = await client.start("hello")
        events = client.events(run)

        await self.gateway.drop()

        self.assertEqual((await self._next(events)).kind, "failed")

    async def test_the_connection_comes_back_after_the_reader_gives_up(self):
        """The dead reader is not left in place, so the next run reconnects."""
        client = OpenClawGatewayClient(url=self.gateway.url, reconnect_on_error=False)
        self.addAsyncCleanup(client.disconnect)
        run = await client.start("hello")
        events = client.events(run)
        await self.gateway.drop()
        await self._next(events)

        run = await asyncio.wait_for(client.start("again"), timeout=2)
        events = client.events(run)
        await self.gateway.chat(run.run_id, "final", "back")
        self.assertEqual((await self._next(events)).text, "back")

    async def test_a_request_that_never_leaves_is_not_left_waiting(self):
        """A send that fails settles the request instead of stranding it."""

        class _BrokenSocket:
            state = State.OPEN

            async def send(self, _):
                raise ConnectionError("the socket is gone")

            async def close(self):
                pass

        await self.client.connect()
        self.client._websocket = _BrokenSocket()

        with self.assertRaises(ConnectionError):
            await self.client.start("hello")
        self.assertFalse(self.client._pending)


class TestOpenClawGatewayService(unittest.IsolatedAsyncioTestCase):
    """The processor, driven through a pipeline."""

    async def asyncSetUp(self):
        self.gateway = await FakeGateway().__aenter__()
        self.processor = OpenClawGatewayService(url=self.gateway.url, token="test-token")

    async def asyncTearDown(self):
        await self.gateway.__aexit__(None, None, None)

    def _stream(self, handler):
        """Answer the gateway's requests by pushing chat frames."""

        async def on_request(request):
            await handler(request.get("method"), request.get("params") or {})

        self.gateway.on_request = on_request

    async def test_a_run_becomes_frames(self):
        async def answer(method, params):
            if method == "chat.send":
                run_id = params["idempotencyKey"]
                await self.gateway.chat(run_id, "delta", "one ")
                await self.gateway.chat(run_id, "delta", "two")
                await self.gateway.chat(run_id, "final", "one two")

        self._stream(answer)

        received, _ = await run_test(
            self.processor,
            frames_to_send=[OpenClawSendFrame(message="hello"), SleepFrame()],
            expected_down_frames=[
                OpenClawStartedFrame,
                OpenClawTextFrame,
                OpenClawTextFrame,
                OpenClawEndFrame,
            ],
        )
        self.assertEqual(self.gateway.params("chat.send")["message"], "hello")
        self.assertEqual(received[-1].status, "completed")
        self.assertEqual(received[-1].text, "one two")

    async def test_an_abort_ends_the_run_as_cancelled(self):
        async def answer(method, params):
            if method == "chat.send":
                await self.gateway.chat(params["idempotencyKey"], "delta", "thinking")
            elif method == "chat.abort":
                await self.gateway.chat(params["runId"], "aborted", "stopped")

        self._stream(answer)

        received, _ = await run_test(
            self.processor,
            frames_to_send=[
                OpenClawSendFrame(message="hello"),
                SleepFrame(),
                OpenClawAbortFrame(reason="the user said stop"),
                SleepFrame(),
            ],
            expected_down_frames=[
                OpenClawStartedFrame,
                OpenClawTextFrame,
                OpenClawEndFrame,
            ],
        )
        self.assertEqual(received[-1].status, "cancelled")

    async def test_a_steer_keeps_streaming_the_same_run(self):
        """Steering starts a new run on the Gateway, and one stream downstream."""

        async def answer(method, params):
            if method == "chat.send":
                await self.gateway.chat(params["idempotencyKey"], "delta", "researching")
            elif method == "sessions.steer":
                run_id = params["idempotencyKey"]
                await self.gateway.chat(run_id, "delta", "comparing")
                await self.gateway.chat(run_id, "final", "FP8 wins on throughput")

        self._stream(answer)

        received, _ = await run_test(
            self.processor,
            frames_to_send=[
                OpenClawSendFrame(message="research NVFP4"),
                SleepFrame(),
                OpenClawSteerFrame(message="compare it to FP8"),
                SleepFrame(),
            ],
            expected_down_frames=[
                OpenClawStartedFrame,
                OpenClawTextFrame,
                OpenClawTextFrame,
                OpenClawEndFrame,
            ],
        )
        self.assertEqual(received[-1].text, "FP8 wins on throughput")

    async def test_a_second_run_ends_the_first_one(self):
        """A session runs one turn at a time, so every start frame gets an end."""

        async def answer(method, params):
            if method == "chat.send" and params["message"] == "second":
                await self.gateway.chat(params["idempotencyKey"], "final", "done")

        self._stream(answer)

        await run_test(
            self.processor,
            frames_to_send=[
                OpenClawSendFrame(message="first"),
                SleepFrame(),
                OpenClawSendFrame(message="second"),
                SleepFrame(),
            ],
            expected_down_frames=[
                OpenClawStartedFrame,
                OpenClawEndFrame,
                OpenClawStartedFrame,
                OpenClawEndFrame,
            ],
        )

    async def test_a_steer_after_the_run_finished_is_ignored(self):
        """A finished run is not steered.

        Steering a run the Gateway has already completed would start a
        replacement with nothing streaming it, so its output would never
        reach the pipeline.
        """

        async def answer(method, params):
            if method == "chat.send":
                await self.gateway.chat(params["idempotencyKey"], "final", "done")

        self._stream(answer)

        await run_test(
            self.processor,
            frames_to_send=[
                OpenClawSendFrame(message="hello"),
                SleepFrame(),
                OpenClawSteerFrame(message="actually, do this instead"),
                SleepFrame(),
            ],
            expected_down_frames=[
                OpenClawStartedFrame,
                OpenClawEndFrame,
            ],
        )
        self.assertEqual(self.gateway.count("sessions.steer"), 0)

    async def test_a_failed_run_is_reported_upstream(self):
        """A run the Gateway gave up on ends the stream and raises an error."""

        async def answer(method, params):
            if method == "chat.send":
                await self.gateway.chat(
                    params["idempotencyKey"], "error", error_message="the model is unavailable"
                )

        self._stream(answer)

        received, errors = await run_test(
            self.processor,
            frames_to_send=[OpenClawSendFrame(message="hello"), SleepFrame()],
            expected_down_frames=[OpenClawStartedFrame, OpenClawEndFrame],
            expected_up_frames=[ErrorFrame],
        )
        self.assertEqual(received[-1].status, "failed")
        self.assertIn("the model is unavailable", received[-1].text)

    async def test_an_unreachable_gateway_is_reported_upstream(self):
        await self.gateway.__aexit__(None, None, None)

        await run_test(
            OpenClawGatewayService(url=self.gateway.url, connect_timeout=1.0),
            frames_to_send=[SleepFrame()],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )


if __name__ == "__main__":
    unittest.main()

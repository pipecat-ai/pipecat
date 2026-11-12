#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scripted runs stop on a service failure even outside an expectation wait."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalScriptTurnProgress, EvalTrace
from pipecat.evals.script import EvalExpectation, EvalScriptScenario, EvalScriptTurn, EvalSendAfter
from pipecat.evals.script_driver import EvalScriptDriver
from pipecat.frames.frames import BotStartedSpeakingFrame


class TestScriptErrorLifecycle(unittest.IsolatedAsyncioTestCase):
    def _driver(self, turns, *, stop_on_failure=True, progress=None):
        trace = EvalTrace()
        stream = EvalEventStream(bot_audio=True, trace=trace)
        client = AsyncMock()
        client.has_user_tts = False
        driver = EvalScriptDriver(
            scenario=EvalScriptScenario(
                name="service failure", turns=turns, stop_on_failure=stop_on_failure
            ),
            default_timeout_ms=30_000,
            client=client,
            stream=stream,
            judge=None,
            trace=trace,
            progress=progress or AsyncMock(),
        )
        return driver, client, stream

    async def _fail_while_waiting(self, driver, stream, entered):
        before = asyncio.all_tasks()
        run = asyncio.create_task(driver.run())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            # The pipeline publishes this independently of its queued diagnostic event.
            stream.harness_failed("the caller voice failed")
            failures = await asyncio.wait_for(run, 1)
        finally:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
        self.assertEqual(asyncio.all_tasks() - before, set())
        self.assertEqual(failures[-1].kind, "error")
        self.assertEqual(failures[-1].reason, "the caller voice failed")
        self.assertEqual(driver.turns[failures[-1].turn_index].status, "failed")
        return failures

    async def test_an_error_during_a_send_cannot_pass_a_turn_without_expectations(self):
        driver, client, stream = self._driver([EvalScriptTurn(user="hello")])

        async def send(_):
            stream.harness_failed("the caller voice failed")

        client.send_text.side_effect = send
        failures = await driver.run()

        self.assertEqual(
            [(f.kind, f.reason) for f in failures], [("error", "the caller voice failed")]
        )
        self.assertEqual(driver.turns[0].status, "failed")

    async def test_an_error_cancels_an_in_progress_send_and_joins_its_cleanup(self):
        driver, client, stream = self._driver([EvalScriptTurn(user="hello")])
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def send(_):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        client.send_text.side_effect = send
        await self._fail_while_waiting(driver, stream, entered)
        self.assertTrue(cleaned.is_set())

    async def test_an_error_interrupts_a_pure_or_already_anchored_send_after_delay(self):
        for event in (None, "llm_started"):
            with self.subTest(event=event):
                driver, client, stream = self._driver(
                    [
                        EvalScriptTurn(
                            user="hello", send_after=EvalSendAfter(event=event, delay_ms=30_000)
                        )
                    ]
                )
                if event is not None:
                    await stream.append({"type": event})
                entered = asyncio.Event()
                wait = driver._wait_send_after

                async def waiting(after, entered=entered, wait=wait):
                    entered.set()
                    await wait(after)

                with patch.object(driver, "_wait_send_after", waiting):
                    await self._fail_while_waiting(driver, stream, entered)
                client.send_text.assert_not_awaited()

    async def test_an_error_interrupts_bot_quiet_or_pending_transcription(self):
        for transcribing in (False, True):
            with self.subTest(transcribing=transcribing):
                driver, client, stream = self._driver([EvalScriptTurn(user="hello")])
                if transcribing:
                    stream.bot_segment_started()
                    method = "wait_bot_transcribed"
                else:
                    stream.frame_to_event(BotStartedSpeakingFrame())
                    method = "wait_bot_quiet"
                entered = asyncio.Event()
                wait = getattr(stream, method)

                async def waiting(timeout, entered=entered, wait=wait):
                    entered.set()
                    return await wait(timeout)

                with patch.object(stream, method, waiting):
                    await self._fail_while_waiting(driver, stream, entered)
                client.send_text.assert_not_awaited()

    async def test_an_error_interrupts_an_observing_turn_waiting_for_the_previous_reply(self):
        driver, client, stream = self._driver(
            [EvalScriptTurn(user="hello"), EvalScriptTurn(user=None)]
        )
        entered = asyncio.Event()

        async def send(_):
            stream.frame_to_event(BotStartedSpeakingFrame())

        client.send_text.side_effect = send
        wait = stream.wait_bot_quiet

        async def waiting(timeout):
            entered.set()
            return await wait(timeout)

        with patch.object(stream, "wait_bot_quiet", waiting):
            await self._fail_while_waiting(driver, stream, entered)
        self.assertEqual([turn.status for turn in driver.turns], ["passed", "failed"])

    async def test_an_error_preserves_an_earlier_failure_in_the_unfinished_turn(self):
        entered = asyncio.Event()

        async def progress(event):
            if isinstance(event, EvalScriptTurnProgress) and event.expectation_index == 0:
                entered.set()

        driver, client, stream = self._driver(
            [
                EvalScriptTurn(
                    user="hello",
                    expect=[
                        EvalExpectation(event="llm_marker", marker="complete"),
                        EvalExpectation(event="llm_response"),
                    ],
                ),
                EvalScriptTurn(user="must not send"),
            ],
            stop_on_failure=False,
            progress=progress,
        )

        async def send(_):
            await stream.append({"type": "llm_marker", "kind": "short", "text": "short"})

        client.send_text.side_effect = send
        failures = await self._fail_while_waiting(driver, stream, entered)

        self.assertEqual([f.kind for f in failures], ["marker_mismatch", "error"])
        self.assertEqual(driver.turns[0].failures, failures)
        self.assertEqual([turn.status for turn in driver.turns], ["failed", "not_run"])
        client.send_text.assert_awaited_once_with("hello")

    async def test_an_error_keeps_a_completed_failed_turn_and_stops_even_if_failures_may_continue(
        self,
    ):
        entered = asyncio.Event()
        driver, client, stream = self._driver(
            [
                EvalScriptTurn(
                    user="first", expect=[EvalExpectation(event="llm_marker", marker="complete")]
                ),
                EvalScriptTurn(user="second"),
                EvalScriptTurn(user="must not send"),
            ],
            stop_on_failure=False,
        )

        async def send(text):
            if text == "first":
                await stream.append({"type": "llm_marker", "kind": "short", "text": "short"})
            else:
                entered.set()
                await asyncio.Event().wait()

        client.send_text.side_effect = send
        failures = await self._fail_while_waiting(driver, stream, entered)

        self.assertEqual([f.kind for f in failures], ["marker_mismatch", "error"])
        self.assertEqual([turn.status for turn in driver.turns], ["failed", "failed", "not_run"])
        self.assertEqual([f.turn_index for f in failures], [0, 1])

    async def test_a_decided_timeout_survives_an_error_during_its_progress_callback(self):
        entered = asyncio.Event()

        async def progress(event):
            if isinstance(event, EvalScriptTurnProgress) and event.status == "timeout":
                entered.set()
                await asyncio.Event().wait()

        driver, _, stream = self._driver(
            [
                EvalScriptTurn(
                    user="hello", expect=[EvalExpectation(event="llm_response", within_ms=1)]
                )
            ],
            progress=progress,
        )
        failures = await self._fail_while_waiting(driver, stream, entered)

        self.assertEqual([failure.kind for failure in failures], ["timeout", "error"])
        self.assertEqual(driver.turns[0].failures, failures)

    async def test_an_error_after_the_completed_run_keeps_its_verdict(self):
        before = asyncio.all_tasks()
        driver, _, stream = self._driver([EvalScriptTurn(user="hello")])
        failures = await driver.run()
        stream.harness_failed("too late to change the completed run")

        self.assertEqual(failures, [])
        self.assertEqual(driver.turns[0].status, "passed")
        self.assertEqual(asyncio.all_tasks() - before, set())

    async def test_a_reported_harness_error_is_not_duplicated_if_its_progress_callback_blocks(self):
        before = asyncio.all_tasks()

        async def progress(event):
            if isinstance(event, EvalScriptTurnProgress) and event.status == "failed":
                await asyncio.Event().wait()

        driver, client, stream = self._driver(
            [EvalScriptTurn(user="hello", expect=[EvalExpectation(event="llm_response")])],
            progress=progress,
        )

        async def send(_):
            stream.harness_failed("voice service failed")

        client.send_text.side_effect = send
        failures = await asyncio.wait_for(driver.run(), 1)

        self.assertEqual(
            [(f.kind, f.reason) for f in failures], [("error", "voice service failed")]
        )
        self.assertEqual(driver.turns[0].failures, failures)
        self.assertEqual(driver.turns[0].status, "failed")
        self.assertEqual(asyncio.all_tasks() - before, set())

    async def test_a_decided_send_after_timeout_survives_an_error_during_progress(self):
        entered = asyncio.Event()

        async def progress(event):
            if isinstance(event, EvalScriptTurnProgress) and event.status == "timeout":
                entered.set()
                await asyncio.Event().wait()

        driver, _, stream = self._driver(
            [
                EvalScriptTurn(
                    user="hello", send_after=EvalSendAfter(event="llm_started", delay_ms=0)
                )
            ],
            progress=progress,
        )
        with patch.object(driver, "_wait_send_after", AsyncMock(side_effect=TimeoutError)):
            failures = await self._fail_while_waiting(driver, stream, entered)

        self.assertEqual([failure.kind for failure in failures], ["send_after_timeout", "error"])
        self.assertEqual(driver.turns[0].failures, failures)

    async def test_a_completed_run_wins_if_the_error_arrives_before_its_waiter_resumes(self):
        driver, _, stream = self._driver([EvalScriptTurn(user="hello")])

        async def progress(_):
            asyncio.get_running_loop().call_soon(stream.harness_failed, "after completion")

        driver._progress = progress
        failures = await driver.run()
        await asyncio.sleep(0)

        self.assertEqual(stream.harness_error, "after completion")
        self.assertEqual(failures, [])
        self.assertEqual(driver.turns[0].status, "passed")

    async def test_a_preexisting_first_error_cannot_be_dropped_or_start_a_send(self):
        driver, client, stream = self._driver([EvalScriptTurn(user="hello")])
        stream.harness_failed("first error")
        await stream.append({"type": "harness_error", "text": "first error"})
        stream.drop_pending_bot_output("before send")
        stream.harness_failed("later error")

        failures = await driver.run()

        self.assertEqual([(f.kind, f.reason) for f in failures], [("error", "first error")])
        self.assertEqual(failures[0].turn_index, -1)
        self.assertEqual(driver.turns[0].status, "not_run")
        client.send_text.assert_not_awaited()

    async def test_an_error_when_speech_finishes_cannot_release_the_next_send(self):
        driver, client, stream = self._driver([EvalScriptTurn(user="hello")])
        stream.frame_to_event(BotStartedSpeakingFrame())

        async def quiet(_):
            stream.harness_failed("speech service failed")
            return True

        with patch.object(stream, "wait_bot_quiet", quiet):
            failures = await driver.run()

        self.assertEqual([f.kind for f in failures], ["error"])
        client.send_text.assert_not_awaited()

    async def test_caller_cancellation_joins_the_send_and_monitor(self):
        before = asyncio.all_tasks()
        driver, client, _ = self._driver([EvalScriptTurn(user="hello")])
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def send(_):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        client.send_text.side_effect = send
        run = asyncio.create_task(driver.run())
        await asyncio.wait_for(entered.wait(), 1)
        run.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run
        self.assertTrue(cleaned.is_set())
        self.assertEqual(asyncio.all_tasks() - before, set())

    async def test_a_progress_exception_is_not_replaced_by_a_service_error(self):
        before = asyncio.all_tasks()
        driver, _, stream = self._driver([EvalScriptTurn(user="hello")])

        async def progress(_):
            stream.harness_failed("service error")
            raise ValueError("progress callback failed")

        driver._progress = progress
        with self.assertRaisesRegex(ValueError, "progress callback failed"):
            await driver.run()
        self.assertEqual(asyncio.all_tasks() - before, set())

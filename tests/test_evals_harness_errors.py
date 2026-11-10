#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A scripted run cannot score evidence after its own speech pipeline fails."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.script import EvalExpectation, EvalScriptScenario, EvalScriptTurn, EvalSendAfter
from pipecat.evals.script_driver import EvalScriptDriver


def _driver(turns, *, stream=None, stop_on_failure=True, progress=None):
    trace = EvalTrace()
    stream = stream or EvalEventStream(bot_audio=False, trace=trace)
    sent = asyncio.Event()

    async def send(_text):
        sent.set()

    client = SimpleNamespace(has_user_tts=False, send_text=AsyncMock(side_effect=send))
    driver = EvalScriptDriver(
        scenario=EvalScriptScenario(
            name="speech-errors", turns=turns, stop_on_failure=stop_on_failure
        ),
        default_timeout_ms=30_000,
        client=client,
        stream=stream,
        judge=None,
        trace=trace,
        progress=progress or AsyncMock(),
    )
    return driver, stream, client, sent


async def _failure(stream):
    await stream.append({"type": "harness_error", "text": "speech service unavailable"})


async def _interrupt_wait(driver, stream, entered):
    task = asyncio.create_task(driver.run())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await _failure(stream)
        failures = await asyncio.wait_for(task, 0.5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert [failure.kind for failure in failures] == ["error"]
    assert failures[0].event_name == "<error>"
    return failures


@pytest.mark.asyncio
@pytest.mark.parametrize("absent", [False, True])
async def test_speech_error_ends_response_and_absence_waits_without_a_timeout(absent):
    turn = EvalScriptTurn(
        user="hello", expect=[EvalExpectation(event="llm_response", absent=absent)]
    )
    driver, stream, _client, sent = _driver([turn])
    await _interrupt_wait(driver, stream, sent)
    assert driver.turns[0].status == "failed"
    assert [failure.kind for failure in driver.turns[0].failures] == ["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_method", ["wait_bot_quiet", "wait_bot_transcribed"])
async def test_speech_error_interrupts_bot_quiet_and_final_transcription_waits(wait_method):
    stream = EvalEventStream(bot_audio=True, trace=EvalTrace())
    if wait_method == "wait_bot_quiet":
        stream._bot_quiet.clear()
    else:
        stream.bot_turn_started()
    entered = asyncio.Event()
    original_wait = getattr(stream, wait_method)

    async def observed_wait(timeout):
        entered.set()
        return await original_wait(timeout)

    setattr(stream, wait_method, observed_wait)
    driver, stream, client, _sent = _driver([EvalScriptTurn(user="hello")], stream=stream)
    await _interrupt_wait(driver, stream, entered)
    client.send_text.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [None, "llm_started"])
async def test_speech_error_interrupts_send_after_delay_and_event_waits(event):
    driver, stream, client, _sent = _driver(
        [EvalScriptTurn(user="hello", send_after=EvalSendAfter(event=event, delay_ms=30_000))]
    )
    entered = asyncio.Event()
    wait = driver._wait_send_after

    async def observed_wait(send_after):
        entered.set()
        return await wait(send_after)

    driver._wait_send_after = observed_wait
    await _interrupt_wait(driver, stream, entered)
    client.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_speech_error_cancels_an_in_progress_send_and_its_resources():
    driver, stream, client, sent = _driver([EvalScriptTurn(user="hello")])
    cleaned = asyncio.Event()

    async def send(_text):
        sent.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    client.send_text.side_effect = send
    await _interrupt_wait(driver, stream, sent)
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_an_error_before_the_first_turn_is_not_dropped_by_sending_input():
    driver, stream, client, _sent = _driver([EvalScriptTurn(user="hello")])
    await _failure(stream)
    stream.drop_pending_bot_output("before driving")
    failures = await asyncio.wait_for(driver.run(), 0.5)
    assert [failure.kind for failure in failures] == ["error"]
    assert failures[0].turn_index == -1
    assert driver.turns[0].status == "not_run"
    client.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_error_during_a_send_cannot_pass_a_turn_without_expectations():
    driver, stream, client, _sent = _driver([EvalScriptTurn(user="hello")])

    async def send(_text):
        await _failure(stream)

    client.send_text.side_effect = send
    failures = await driver.run()
    assert [failure.kind for failure in failures] == ["error"]
    assert driver.turns[0].status == "failed"


@pytest.mark.asyncio
async def test_error_interrupts_an_observing_turn_waiting_for_the_previous_reply():
    stream = EvalEventStream(bot_audio=True, trace=EvalTrace())
    entered = asyncio.Event()
    original_wait = stream.wait_bot_transcribed

    async def observed_wait(timeout):
        entered.set()
        return await original_wait(timeout)

    stream.wait_bot_transcribed = observed_wait

    async def progress(record):
        if record.status == "matched":
            stream.bot_turn_started()

    driver, stream, client, _sent = _driver(
        [
            EvalScriptTurn(user="hello", expect=[EvalExpectation(event="llm_response")]),
            EvalScriptTurn(user=None, expect=[EvalExpectation(event="response")]),
        ],
        stream=stream,
        progress=progress,
    )

    async def send(_text):
        await stream.append({"type": "llm_response", "text": "hello"})

    client.send_text.side_effect = send
    await _interrupt_wait(driver, stream, entered)
    assert [turn.status for turn in driver.turns] == ["passed", "failed"]
    assert client.send_text.await_count == 1


@pytest.mark.asyncio
async def test_error_stops_even_when_assertion_failures_may_continue_and_keeps_prior_failures():
    first = EvalScriptTurn(
        user="first", expect=[EvalExpectation(event="llm_response", text_excludes="bad")]
    )
    second = EvalScriptTurn(user="second", expect=[EvalExpectation(event="llm_response")])
    driver, stream, client, _sent = _driver(
        [first, second, EvalScriptTurn(user="never")], stop_on_failure=False
    )
    second_sent = asyncio.Event()

    async def send(text):
        if text == "first":
            await stream.append({"type": "llm_response", "text": "bad reply"})
        else:
            second_sent.set()

    client.send_text.side_effect = send
    task = asyncio.create_task(driver.run())
    try:
        await asyncio.wait_for(second_sent.wait(), 1)
        await _failure(stream)
        failures = await asyncio.wait_for(task, 0.5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(failures) == 2
    assert failures[0].kind != "error"
    assert failures[1].kind == "error"
    assert [turn.status for turn in driver.turns] == ["failed", "failed", "not_run"]
    assert client.send_text.await_count == 2


@pytest.mark.asyncio
async def test_error_preserves_a_failed_expectation_in_the_current_unfinished_turn():
    decided = asyncio.Event()

    async def progress(record):
        if record.status == "failed" and record.expectation_index == 0:
            decided.set()

    driver, stream, client, _sent = _driver(
        [
            EvalScriptTurn(
                user="hello",
                expect=[
                    EvalExpectation(event="llm_response", text_excludes="bad"),
                    EvalExpectation(event="user_transcription"),
                ],
            )
        ],
        progress=progress,
    )

    async def send(_text):
        await stream.append({"type": "llm_response", "text": "bad reply"})

    client.send_text.side_effect = send
    task = asyncio.create_task(driver.run())
    try:
        await asyncio.wait_for(decided.wait(), 1)
        await _failure(stream)
        failures = await asyncio.wait_for(task, 0.5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(failures) == 2
    assert failures[0].kind != "error"
    assert failures[0].expectation_index == 0
    assert failures[1].kind == "error"
    assert driver.turns[0].failures == failures


@pytest.mark.asyncio
async def test_progress_exception_propagates_without_an_error_monitor_leak():
    async def progress(_record):
        raise ValueError("progress handler failed")

    driver, _stream, _client, _sent = _driver([EvalScriptTurn(user="hello")], progress=progress)
    before = asyncio.all_tasks()
    with pytest.raises(ValueError, match="progress handler failed"):
        await driver.run()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_an_error_after_the_run_completed_does_not_change_its_result_or_leave_a_monitor():
    driver, stream, client, _sent = _driver(
        [EvalScriptTurn(user="hello", expect=[EvalExpectation(event="llm_response")])]
    )

    async def send(_text):
        await stream.append({"type": "llm_response", "text": "hello"})

    client.send_text.side_effect = send
    before = asyncio.all_tasks()
    assert await driver.run() == []
    await _failure(stream)
    assert driver.turns[0].status == "passed"
    assert driver.turns[0].failures == []
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_caller_cancellation_cancels_the_send_and_error_monitor():
    driver, _stream, client, sent = _driver([EvalScriptTurn(user="hello")])
    cleaned = asyncio.Event()

    async def send(_text):
        sent.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    client.send_text.side_effect = send
    before = asyncio.all_tasks()
    task = asyncio.create_task(driver.run())
    await asyncio.wait_for(sent.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert not (asyncio.all_tasks() - before)

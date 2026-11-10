#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulation scoring stops when the harness loses its speech evidence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from pipecat.evals.client import BOT_ENDED_EVENT, HARNESS_ERROR_EVENT
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalAssertionFailure, EvalSimulationMetricScore, EvalTrace
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.evals.simulation_driver import EvalSimulationDriver


def _driver(*, progress=None):
    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=False, trace=trace)
    client = SimpleNamespace(hang_up=AsyncMock())
    driver = EvalSimulationDriver(
        simulation=EvalSimulationScenario(
            name="speech-failures", persona="caller", goal="ask for help", success="agent helps"
        ),
        persona=SimpleNamespace(on_end_call=Mock()),
        client=client,
        stream=stream,
        judge=None,
        trace=trace,
        progress=progress or AsyncMock(),
    )

    async def judge():
        driver._succeeded = True
        driver._reason = "goal met"
        return []

    driver._judge_conversation = AsyncMock(side_effect=judge)
    return driver, stream, client


async def _speech_error(stream):
    await stream.append({"type": HARNESS_ERROR_EVENT, "text": "PRIVATE_PROVIDER_BODY"})


def _result(driver, failures):
    return driver.result(failures=failures, duration_ms=0, events_seen=[], debug_log=[])


@pytest.mark.asyncio
async def test_a_preexisting_error_survives_queue_dropping_and_prevents_judging():
    driver, stream, client = _driver()
    await _speech_error(stream)
    stream.drop_pending_bot_output("before send")
    await stream.append({"type": BOT_ENDED_EVENT})
    failures = await asyncio.wait_for(driver.run(), 0.5)
    assert [failure.kind for failure in failures] == ["error"]
    driver._judge_conversation.assert_not_called()
    client.hang_up.assert_awaited()
    result = _result(driver, failures)
    assert result.error == "the eval harness pipeline failed"
    assert result.ended_by == "error"
    assert not result.passed
    assert "PRIVATE_PROVIDER_BODY" not in result.reason
    assert any("PRIVATE_PROVIDER_BODY" in line for line in driver._trace.lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["progress", "judging"])
async def test_a_speech_failure_cancels_blocked_progress_or_judging(phase):
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocked(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    driver, stream, client = _driver(progress=blocked if phase == "progress" else None)
    if phase == "judging":
        driver._judge_conversation.side_effect = blocked
    await stream.append(
        {"type": "llm_response", "text": "hello"}
        if phase == "progress"
        else {"type": BOT_ENDED_EVENT}
    )
    before = asyncio.all_tasks()
    task = asyncio.create_task(driver.run())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await _speech_error(stream)
        failures = await asyncio.wait_for(task, 0.5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert [failure.kind for failure in failures] == ["error"]
    assert cleaned.is_set()
    assert not (asyncio.all_tasks() - before)
    client.hang_up.assert_awaited()
    if phase == "progress":
        driver._judge_conversation.assert_not_called()
    assert _result(driver, failures).error is not None


@pytest.mark.asyncio
async def test_a_speech_failure_reported_inside_judging_cannot_commit_a_pass():
    driver, stream, _client = _driver()

    async def judge():
        await _speech_error(stream)
        driver._succeeded = True
        driver._metrics = [EvalSimulationMetricScore(name="quality", score=1)]
        return []

    driver._judge_conversation.side_effect = judge
    await stream.append({"type": BOT_ENDED_EVENT})
    failures = await driver.run()
    result = _result(driver, failures)
    assert [failure.kind for failure in failures] == ["error"]
    assert not result.succeeded
    assert not result.passed
    assert result.metrics == []


@pytest.mark.asyncio
@pytest.mark.parametrize("criterion_passed", [False, True])
async def test_a_speech_failure_after_a_completed_result_keeps_its_verdict(criterion_passed):
    driver, stream, _client = _driver()

    async def judge():
        driver._succeeded = True
        driver._metrics = [
            EvalSimulationMetricScore(
                name="quality", score=float(criterion_passed), passed=criterion_passed
            )
        ]
        return []

    driver._judge_conversation.side_effect = judge
    await stream.append({"type": BOT_ENDED_EVENT})
    before = asyncio.all_tasks()
    failures = await driver.run()
    await _speech_error(stream)
    result = _result(driver, failures)
    assert failures == []
    assert result.error is None
    assert result.passed is criterion_passed
    assert result.metrics[0].passed is criterion_passed
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["progress", "judging"])
async def test_caller_cancellation_cleans_up_the_active_operation_and_monitor(phase):
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocked(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    driver, stream, _client = _driver(progress=blocked if phase == "progress" else None)
    if phase == "judging":
        driver._judge_conversation.side_effect = blocked
    await stream.append(
        {"type": "llm_response", "text": "hello"}
        if phase == "progress"
        else {"type": BOT_ENDED_EVENT}
    )
    before = asyncio.all_tasks()
    task = asyncio.create_task(driver.run())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_a_progress_exception_propagates_without_a_monitor_leak():
    async def progress(_record):
        raise ValueError("progress failed")

    driver, stream, _client = _driver(progress=progress)
    await stream.append({"type": "llm_response", "text": "hello"})
    before = asyncio.all_tasks()
    with pytest.raises(ValueError, match="progress failed"):
        await driver.run()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_simulation_does_not_begin_judging_after_error_in_ended_progress():
    async def progress(record):
        if record.status == "ended":
            await _speech_error(stream)

    driver, stream, _client = _driver(progress=progress)
    await stream.append({"type": "bot_ended"})
    failures = await driver.run()
    assert [f.kind for f in failures] == ["error"]
    driver._judge_conversation.assert_not_called()


@pytest.mark.parametrize(
    "kind", ["error", "connect_failed", "closed_before_ready", "handshake_timeout"]
)
def test_simulation_retains_machine_readable_startup_failure_kind(kind):
    driver, _stream, _client = _driver()
    result = _result(
        driver,
        [EvalAssertionFailure(-1, -1, "<bot-ready>", "Safe startup reason", kind)],
    )
    assert result.failure_kind == kind
    assert result.error == "Safe startup reason"
    assert not result.passed


def test_simulation_without_run_failure_keeps_failure_kind_optional():
    driver, _stream, _client = _driver()
    result = _result(driver, [])
    assert result.failure_kind is None

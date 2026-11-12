#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A simulation cannot judge an incomplete conversation after its harness fails."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pipecat.evals.client import BOT_ENDED_EVENT, HARNESS_ERROR_EVENT
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import JudgeVerdict, RunVerdicts
from pipecat.evals.results import EvalTrace
from pipecat.evals.simulation import EvalSimulationMetric, EvalSimulationScenario
from pipecat.evals.simulation_driver import EvalSimulationDriver


def _verdicts(*, criterion_passed=True):
    return RunVerdicts(
        goal=JudgeVerdict(
            verdict="yes",
            reason="goal met",
            raw_response="",
            confidence=0.95,
            probabilities={"yes": 0.95, "no": 0.05},
        ),
        turns={
            "helpfulness": [
                JudgeVerdict(
                    verdict="yes" if criterion_passed else "no",
                    reason="criterion verdict",
                    raw_response="",
                )
            ]
        },
    )


def _driver(*, progress=None, criterion_passed=True):
    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=False, trace=trace)
    client = SimpleNamespace(hang_up=AsyncMock())
    judge = SimpleNamespace(
        add_user_message=Mock(),
        add_assistant_message=Mock(),
        add_tool_call=Mock(),
        evaluate_run=AsyncMock(return_value=_verdicts(criterion_passed=criterion_passed)),
    )
    driver = EvalSimulationDriver(
        simulation=EvalSimulationScenario(
            name="voice failure",
            persona="caller",
            goal="ask for help",
            success="the bot helped",
            max_duration_s=30,
            max_silence_s=30,
            metrics=[
                EvalSimulationMetric(name="helpfulness", criterion="the bot helps", min_score=1)
            ],
        ),
        persona=SimpleNamespace(on_end_call=Mock()),
        client=client,
        stream=stream,
        judge=judge,
        trace=trace,
        progress=progress or AsyncMock(),
    )
    return driver, stream, client, judge


def _result(driver, failures):
    return driver.result(failures=failures, duration_ms=0, events_seen=[], debug_log=[])


async def _end_conversation(stream):
    await stream.append({"type": "llm_response", "text": "I can help."})
    await stream.append({"type": BOT_ENDED_EVENT})


def _assert_error(driver, failures, message="caller voice failed"):
    assert [(failure.kind, failure.reason) for failure in failures] == [("error", message)]
    result = _result(driver, failures)
    assert result.ended_by == "error"
    assert result.error == message
    assert not result.succeeded
    assert not result.passed
    assert result.metrics == []
    assert result.confidence is None
    assert result.probabilities == {}


@pytest.mark.asyncio
async def test_a_discarded_queued_harness_error_cannot_still_judge_and_pass():
    driver, stream, client, judge = _driver()
    stream.harness_failed("caller voice failed")
    await stream.append({"type": HARNESS_ERROR_EVENT, "text": "caller voice failed"})
    stream.drop_pending_bot_output("before send")
    await _end_conversation(stream)

    failures = await asyncio.wait_for(driver.run(), 1)

    _assert_error(driver, failures)
    judge.evaluate_run.assert_not_awaited()
    client.hang_up.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["progress", "judging"])
async def test_an_error_cancels_blocked_progress_or_final_judging_and_joins_cleanup(phase):
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def blocked(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    driver, stream, client, judge = _driver(progress=blocked if phase == "progress" else None)
    if phase == "judging":
        judge.evaluate_run.side_effect = blocked
    await _end_conversation(stream)
    before = asyncio.all_tasks()
    run = asyncio.create_task(driver.run())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        # This notification is independent of the bot's output queue.
        stream.harness_failed("caller voice failed")
        failures = await asyncio.wait_for(run, 1)
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)

    _assert_error(driver, failures)
    assert cleaned.is_set()
    assert not (asyncio.all_tasks() - before)
    client.hang_up.assert_awaited()
    if phase == "progress":
        judge.evaluate_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_error_inside_final_judging_cannot_commit_its_verdict_or_scores():
    driver, stream, _client, judge = _driver()

    async def judge_run(*_args):
        stream.harness_failed("caller voice failed")
        return _verdicts()

    judge.evaluate_run.side_effect = judge_run
    await _end_conversation(stream)
    failures = await driver.run()

    _assert_error(driver, failures)
    judge.evaluate_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_error_in_ended_progress_prevents_the_judge_from_starting():
    async def progress(record):
        if record.status == "ended":
            stream.harness_failed("caller voice failed")

    driver, stream, _client, judge = _driver(progress=progress)
    await _end_conversation(stream)
    failures = await driver.run()

    _assert_error(driver, failures)
    judge.evaluate_run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("criterion_passed", [False, True])
async def test_an_error_after_completion_keeps_the_decided_verdict_and_metrics(criterion_passed):
    driver, stream, _client, judge = _driver(criterion_passed=criterion_passed)
    await _end_conversation(stream)
    before = asyncio.all_tasks()
    failures = await driver.run()
    stream.harness_failed("after the completed result")
    result = _result(driver, failures)

    assert failures == []
    assert result.error is None
    assert result.passed is criterion_passed
    assert result.metrics[0].passed is criterion_passed
    assert result.confidence == 0.95
    assert result.probabilities == {"yes": 0.95, "no": 0.05}
    judge.evaluate_run.assert_awaited_once()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_a_finished_verdict_wins_if_the_error_arrives_before_its_waiter_resumes():
    driver, stream, _client, judge = _driver()

    async def judge_run(*_args):
        asyncio.get_running_loop().call_soon(stream.harness_failed, "after completion")
        return _verdicts()

    judge.evaluate_run.side_effect = judge_run
    await _end_conversation(stream)
    failures = await driver.run()
    await asyncio.sleep(0)
    result = _result(driver, failures)

    assert stream.harness_error == "after completion"
    assert failures == []
    assert result.passed
    assert result.confidence == 0.95


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["progress", "judging"])
async def test_caller_cancellation_joins_the_active_operation_and_error_monitor(phase):
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def blocked(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    driver, stream, _client, judge = _driver(progress=blocked if phase == "progress" else None)
    if phase == "judging":
        judge.evaluate_run.side_effect = blocked
    await _end_conversation(stream)
    before = asyncio.all_tasks()
    run = asyncio.create_task(driver.run())
    await asyncio.wait_for(entered.wait(), 1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert cleaned.is_set()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_a_progress_exception_is_not_replaced_by_a_coincident_service_error():
    async def progress(_record):
        stream.harness_failed("caller voice failed")
        raise ValueError("progress failed")

    driver, stream, _client, _judge = _driver(progress=progress)
    await _end_conversation(stream)
    before = asyncio.all_tasks()
    with pytest.raises(ValueError, match="progress failed"):
        await driver.run()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_a_legacy_queued_error_without_the_notification_still_ends_the_run():
    driver, stream, client, judge = _driver()
    await stream.append({"type": HARNESS_ERROR_EVENT, "text": "caller voice failed"})

    failures = await asyncio.wait_for(driver.run(), 1)

    _assert_error(driver, failures)
    judge.evaluate_run.assert_not_awaited()
    client.hang_up.assert_awaited()


@pytest.mark.asyncio
async def test_the_persistent_first_error_wins_over_a_later_queued_error():
    driver, stream, _client, judge = _driver()
    next_any = stream.next_any

    async def error_before_next_event(deadline):
        stream.harness_failed("caller voice failed")
        await stream.append({"type": HARNESS_ERROR_EVENT, "text": "a later error"})
        return await next_any(deadline)

    with patch.object(stream, "next_any", error_before_next_event):
        failures = await driver.run()

    _assert_error(driver, failures)
    judge.evaluate_run.assert_not_awaited()

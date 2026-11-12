#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Startup failures retain their originating service or target transport."""

import asyncio
import dataclasses
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.evals.script import EvalScriptScenario, EvalScriptTurn
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.evals.simulation_session import EvalSimulationSession
from pipecat.evals.suite import EvalRun, _append_result, _simulation_result_from_dict
from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from tests.test_evals_session import _ScriptedPersonaLLM


@asynccontextmanager
async def _started_session(kind="script"):
    params = EvalSessionParams(bot_ready_timeout_s=30)
    if kind == "simulation":
        session = EvalSimulationSession(
            EvalSimulationScenario(name="startup", persona="caller", goal="ask", success="answer"),
            "ws://localhost:0",
            params=params,
            persona_llm=_ScriptedPersonaLLM({}),
            judge=None,
        )
    else:
        session = EvalScriptSession(
            EvalScriptScenario(name="startup-speech-error", turns=[EvalScriptTurn(user="hello")]),
            "ws://localhost:0",
            params=params,
        )
    client = session._client
    transport = client._transport()
    # Register the actual client callback and build its real pipeline objects;
    # the runner is stopped at the network/lifecycle boundary for these tests.
    with (
        patch.object(client, "_transport", return_value=transport),
        patch("pipecat.evals.client.WorkerRunner.add_workers", new=AsyncMock()),
        patch("pipecat.evals.client.WorkerRunner.run", new=AsyncMock()),
        patch.object(client, "_configure_persona", new=AsyncMock()),
    ):
        await client.start()
        await client._run_task
    yield session, transport


async def _pipeline_error(session, exception, processor):
    worker = session._client._worker
    callback = worker._event_handlers["on_pipeline_error"].handlers[0]
    frame = ErrorFrame(error="speech service failed", exception=exception, processor=processor)
    await callback(worker, frame)
    assert frame.processor is processor


def _assert_service_failure(failures):
    assert [failure.kind for failure in failures] == ["error"]
    assert failures[0].reason == "speech service failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception", [OSError("provider DNS failed"), RuntimeError("provider failed")]
)
@pytest.mark.parametrize("during_wait", [False, True])
async def test_provider_startup_failure_interrupts_bot_ready_without_target_retry(
    exception, during_wait
):
    async with _started_session() as (session, _transport):
        provider = FrameProcessor(name="test-speech-provider")
        before = asyncio.all_tasks()
        task = asyncio.create_task(session._converse()) if during_wait else None
        if task is not None:
            await asyncio.sleep(0)
        await _pipeline_error(session, exception, provider)
        session._stream.drop_pending_bot_output("startup boundary")
        try:
            failures = await asyncio.wait_for(task or session._converse(), 0.5)
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        _assert_service_failure(failures)
        assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_unknown_error_origin_cannot_be_classified_as_target_connection_failure():
    async with _started_session() as (session, _transport):
        await _pipeline_error(session, OSError("unknown origin"), None)
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures)


@pytest.mark.asyncio
async def test_provider_error_before_target_closure_remains_the_first_cause():
    async with _started_session() as (session, _transport):
        client = session._client
        await _pipeline_error(session, RuntimeError("provider failed first"), FrameProcessor())
        client._refuse("closed_before_ready", "target closed afterwards")
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures)


@pytest.mark.asyncio
async def test_actual_target_closure_before_provider_error_keeps_its_first_cause():
    async with _started_session() as (session, _transport):
        client = session._client
        client._refuse("closed_before_ready", "target closed first")
        await _pipeline_error(session, RuntimeError("provider failed afterwards"), FrameProcessor())
        failures = await asyncio.wait_for(session._converse(), 0.5)
        assert [failure.kind for failure in failures] == ["closed_before_ready"]
        assert "target closed first" in failures[0].reason


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["input", "output"])
async def test_actual_target_transport_error_keeps_connect_failed(direction):
    async with _started_session() as (session, transport):
        await _pipeline_error(
            session, OSError("target connection failed"), getattr(transport, direction)()
        )
        failures = await asyncio.wait_for(session._converse(), 0.5)
        assert [failure.kind for failure in failures] == ["connect_failed"]


@pytest.mark.asyncio
async def test_simultaneous_ready_and_provider_failure_still_fail_startup():
    async with _started_session() as (session, _transport):
        session._client._handshake_over.set()
        await _pipeline_error(session, RuntimeError("provider failed"), FrameProcessor())
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures)


@pytest.mark.asyncio
async def test_provider_failure_during_context_setup_still_fails_startup():
    async with _started_session() as (session, _transport):
        client = session._client
        client._handshake_over.set()
        client._params.context = [{"role": "user", "content": "starting context"}]

        async def setup(*_args):
            await _pipeline_error(session, RuntimeError("provider failed"), FrameProcessor())

        client._send_eval = AsyncMock(side_effect=setup)
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures)


@pytest.mark.asyncio
async def test_cancelled_bot_ready_wait_cleans_up_all_waiters():
    async with _started_session() as (session, _transport):
        before = asyncio.all_tasks()
        task = asyncio.create_task(session._client.handshake())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_successful_or_timed_out_handshake_joins_both_waiters(ready):
    async with _started_session() as (session, _transport):
        before = asyncio.all_tasks()
        client = session._client
        client._session_params.bot_ready_timeout_s = 0.001
        if ready:
            client._handshake_over.set()
            await client.handshake()
        else:
            with pytest.raises(TimeoutError, match="bot-ready not received within 1ms"):
                await client.handshake()
        assert asyncio.all_tasks() == before


@pytest.mark.asyncio
async def test_provider_error_before_a_target_transport_error_keeps_its_cause():
    async with _started_session() as (session, transport):
        await _pipeline_error(session, RuntimeError("provider failed first"), FrameProcessor())
        await _pipeline_error(session, OSError("target failed later"), transport.input())
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["error", "connect_failed"])
async def test_simulation_retains_startup_category_through_worker_json_and_results_file(
    kind, tmp_path
):
    async with _started_session("simulation") as (session, transport):
        processor = transport.input() if kind == "connect_failed" else FrameProcessor()
        await _pipeline_error(session, OSError("connection failed"), processor)
        failures = await asyncio.wait_for(session._converse(), 0.5)
        assert [failure.kind for failure in failures] == [kind]
        result = session._result(time.monotonic(), failures)
        assert result.failure_kind == kind
        assert not result.passed

        rebuilt = _simulation_result_from_dict(json.loads(json.dumps(dataclasses.asdict(result))))
        assert rebuilt == result
        run = EvalRun(
            bot="agent.py",
            scenario="startup",
            scenario_path=tmp_path / "startup.yaml",
            kind="simulation",
            status="done",
            result=rebuilt,
        )
        path = tmp_path / "results.jsonl"
        _append_result(path, run, "agent_startup", tmp_path, None)
        record = json.loads(path.read_text())
        assert record["failure_kind"] == kind
        assert record["error"] == result.error


def test_old_worker_json_without_a_category_remains_readable():
    result = _simulation_result_from_dict(
        {"simulation_name": "old", "succeeded": False, "error": "old startup failure"}
    )
    assert result.failure_kind is None
    assert result.error == "old startup failure"

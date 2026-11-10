#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Startup speech failures are not evidence that the target agent is unavailable."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.evals.script import EvalScriptScenario, EvalScriptTurn
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.session import EvalSessionParams
from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor


@asynccontextmanager
async def _started_session():
    session = EvalScriptSession(
        EvalScriptScenario(name="startup-speech-error", turns=[EvalScriptTurn(user="hello")]),
        "ws://localhost:0",
        params=EvalSessionParams(bot_ready_timeout_s=30),
    )
    client = session._client
    transport = client._transport()
    # Register the actual client callback and build its real pipeline objects;
    # the runner is stopped at the network/lifecycle boundary for these tests.
    with (
        patch.object(client, "_transport", return_value=transport),
        patch("pipecat.evals.client.WorkerRunner.add_workers", new=AsyncMock()),
        patch("pipecat.evals.client.WorkerRunner.run", new=AsyncMock()),
    ):
        await client.start()
        await client._run_task
    client._sink.allow_answers = AsyncMock()
    yield session, transport


async def _pipeline_error(session, exception, processor):
    worker = session._client._worker
    callback = worker._event_handlers["on_pipeline_error"].handlers[0]
    frame = ErrorFrame(
        error="PRIVATE_PROVIDER_DIAGNOSTIC", exception=exception, processor=processor
    )
    await callback(worker, frame)
    assert frame.processor is processor


def _assert_service_failure(failures, client):
    assert [failure.kind for failure in failures] == ["error"]
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in failures[0].reason
    client._sink.allow_answers.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception", [OSError("provider DNS failed"), RuntimeError("provider failed")]
)
@pytest.mark.parametrize("during_wait", [False, True])
async def test_provider_startup_failure_interrupts_bot_ready_without_target_retry(
    exception, during_wait
):
    async with _started_session() as (session, _transport):
        client = session._client
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
        _assert_service_failure(failures, client)
        assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_unknown_error_origin_cannot_be_classified_as_target_connection_failure():
    async with _started_session() as (session, _transport):
        await _pipeline_error(session, OSError("unknown origin"), None)
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures, session._client)


@pytest.mark.asyncio
async def test_provider_error_before_target_closure_remains_the_first_cause():
    async with _started_session() as (session, _transport):
        client = session._client
        await _pipeline_error(session, RuntimeError("provider failed first"), FrameProcessor())
        client._refuse("closed_before_ready", "target closed afterwards")
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures, client)


@pytest.mark.asyncio
async def test_actual_target_closure_before_provider_error_keeps_its_first_cause():
    async with _started_session() as (session, _transport):
        client = session._client
        client._refuse("closed_before_ready", "target closed first")
        await _pipeline_error(session, RuntimeError("provider failed afterwards"), FrameProcessor())
        failures = await asyncio.wait_for(session._converse(), 0.5)
        assert [failure.kind for failure in failures] == ["closed_before_ready"]
        assert "target closed first" in failures[0].reason
        client._sink.allow_answers.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["input", "output"])
async def test_actual_target_transport_error_keeps_connect_failed(direction):
    async with _started_session() as (session, transport):
        await _pipeline_error(
            session, OSError("target connection failed"), getattr(transport, direction)()
        )
        failures = await asyncio.wait_for(session._converse(), 0.5)
        assert [failure.kind for failure in failures] == ["connect_failed"]
        session._client._sink.allow_answers.assert_not_awaited()


@pytest.mark.asyncio
async def test_simultaneous_ready_and_provider_failure_cannot_release_the_caller():
    async with _started_session() as (session, _transport):
        session._client._handshake_over.set()
        await _pipeline_error(session, RuntimeError("provider failed"), FrameProcessor())
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures, session._client)


@pytest.mark.asyncio
async def test_provider_failure_during_context_setup_cannot_release_the_caller():
    async with _started_session() as (session, _transport):
        client = session._client
        client._handshake_over.set()
        client._params.context = [{"role": "user", "content": "starting context"}]

        async def setup(*_args):
            await _pipeline_error(session, RuntimeError("provider failed"), FrameProcessor())

        client._send_eval = AsyncMock(side_effect=setup)
        failures = await asyncio.wait_for(session._converse(), 0.5)
        _assert_service_failure(failures, client)


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

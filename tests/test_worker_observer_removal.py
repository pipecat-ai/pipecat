#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer detachment, queued callbacks, and proxy task lifetime."""

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from pipecat.frames.frames import EndFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest_asyncio.fixture
async def task_manager() -> AsyncIterator[TaskManager]:
    manager = TaskManager()
    yield manager
    # Also release orphaned tasks when a regression assertion fails.
    for task in list(manager.current_tasks()):
        await manager.cancel_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_event", [False, True])
@pytest.mark.parametrize("reattach", [False, True])
async def test_self_removal_discards_pending_callbacks(task_manager, sync_event, reattach):
    calls = []
    detached = asyncio.Event()

    class Observer(BaseObserver):
        def __init__(self):
            super().__init__()
            self._register_event_handler("on_seen", sync=True)

        async def on_pipeline_started(self):
            calls.append(asyncio.current_task())
            if len(calls) == 1:
                if sync_event:
                    await self._call_event_handler("on_seen")
                else:
                    await detach(self)

    observer = Observer()
    # Keep another observer registered so reattachment does not depend on
    # registration into an empty observer set.
    worker = WorkerObserver(observers=[observer, BaseObserver()])

    @observer.event_handler("on_seen")
    async def detach(observer):
        await worker.remove_observer(observer)
        if reattach:
            worker.add_observer(observer)
        detached.set()

    await worker.setup(task_manager)
    try:
        for _ in range(3):
            await worker.on_pipeline_started()
        await asyncio.wait_for(detached.wait(), timeout=2)
        old_task = calls[0]
        done, _ = await asyncio.wait([old_task], timeout=1)
        assert old_task in done, "Detached proxy task is still running"
        assert not old_task.cancelled(), "Self-removal must let the current callback return"
        assert calls == [old_task], "Detached observer processed its old queue"

        if reattach:
            await worker.on_pipeline_started()
            # Queue joins wait for the new delivery, without relying on sleeps.
            for proxy in worker._proxies.values():
                await asyncio.wait_for(proxy.queue.join(), timeout=2)
            assert len(calls) == 2
            assert calls[1] is not old_task
    finally:
        await worker.cleanup()


@pytest.mark.asyncio
async def test_external_removal_cancels_inflight_callback(task_manager):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Observer(BaseObserver):
        async def on_pipeline_started(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    observer = Observer()
    worker = WorkerObserver(observers=[observer])
    await worker.setup(task_manager)
    try:
        await worker.on_pipeline_started()
        await asyncio.wait_for(entered.wait(), timeout=2)
        await worker.remove_observer(observer)
        assert cancelled.is_set()
        assert not task_manager.current_tasks()
    finally:
        await worker.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_event", [False, True])
async def test_pipeline_observer_removes_itself_from_event_handler(task_manager, sync_event):
    detached = asyncio.Event()
    callback_tasks = []

    class Observer(BaseObserver):
        def __init__(self):
            super().__init__()
            self._register_event_handler("on_ready", sync=sync_event)

        async def on_pipeline_started(self):
            callback_tasks.append(asyncio.current_task())
            await self._call_event_handler("on_ready")

    observer = Observer()
    worker = PipelineWorker(
        Pipeline([IdentityFilter()]),
        observers=[observer],
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
    )

    @observer.event_handler("on_ready")
    async def on_ready(observer):
        await worker.remove_observer(observer)
        detached.set()

    run = asyncio.create_task(worker.run(WorkerParams(task_manager=task_manager)))
    try:
        await asyncio.wait_for(detached.wait(), timeout=10)
        proxy_task = callback_tasks[0]
        done, _ = await asyncio.wait([proxy_task], timeout=1)
        assert proxy_task in done, "Detached proxy task is still running"
        assert proxy_task.cancelled() is not sync_event
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(run, timeout=10)
    assert not task_manager.current_tasks()

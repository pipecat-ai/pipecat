#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Idle timeout callbacks must not overlap while a handler is still running."""

import asyncio

import pytest

from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest.mark.asyncio
async def test_idle_timeout_waits_for_ongoing_handler():
    worker = PipelineWorker(
        Pipeline([IdentityFilter()]),
        idle_timeout_secs=0.05,
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
        enable_turn_tracking=False,
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    calls = 0

    @worker.event_handler("on_idle_timeout")
    async def on_idle_timeout(worker):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()

    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await asyncio.wait_for(first_started.wait(), timeout=10)
        # Leave the first handler active across several idle periods.
        await asyncio.sleep(0.2)
        assert calls == 1

        release_first.set()
        # A subsequent idle period still produces another event.
        await asyncio.wait_for(second_started.wait(), timeout=2)
        assert calls == 2
    finally:
        release_first.set()
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(run, timeout=10)

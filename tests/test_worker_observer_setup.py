#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Initialization and cancellation of dynamically registered observers."""

import asyncio
import unittest

from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.utils.asyncio.task_manager import TaskManager


class SetupObserver(BaseObserver):
    def __init__(self, *, gated=False):
        super().__init__()
        self.setup_entered = asyncio.Event()
        self.setup_release = asyncio.Event()
        self.setup_exited = asyncio.Event()
        self.received = asyncio.Event()
        self.setup_calls = 0
        self.ready = False
        self.calls = []
        if not gated:
            self.setup_release.set()

    async def setup(self, task_manager):
        await super().setup(task_manager)
        self.setup_calls += 1
        self.setup_entered.set()
        try:
            await self.setup_release.wait()
            self.ready = True
        finally:
            self.setup_exited.set()

    async def on_pipeline_started(self):
        self.calls.append(self.ready)
        self.received.set()


class TestObserverSetup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.survivor = SetupObserver()
        self.worker = WorkerObserver(observers=[self.survivor])
        await self.worker.setup(self.task_manager)

    async def asyncTearDown(self):
        await self.worker.cleanup()
        # Release any orphaned proxy tasks even when a regression assertion fails.
        for task in list(self.task_manager.current_tasks()):
            await self.task_manager.cancel_task(task)

    async def test_initial_observer_is_set_up_once(self):
        await self.worker.on_pipeline_started()
        await asyncio.wait_for(self.survivor.received.wait(), timeout=2)
        self.assertEqual(self.survivor.setup_calls, 1)
        self.assertIs(self.survivor.task_manager, self.task_manager)
        self.assertEqual(self.survivor.calls, [True])

    async def test_slow_setup_queues_events_without_blocking_other_observers(self):
        observer = SetupObserver(gated=True)
        self.worker.add_observer(observer)
        await self.worker.on_pipeline_started()
        await self.worker.on_pipeline_started()
        await asyncio.wait_for(self.survivor.received.wait(), timeout=2)
        self.assertEqual(self.survivor.calls, [True, True])
        await asyncio.wait_for(observer.setup_entered.wait(), timeout=2)
        self.assertEqual(observer.calls, [])

        observer.setup_release.set()
        await asyncio.wait_for(observer.received.wait(), timeout=2)
        self.assertEqual(observer.calls, [True, True])
        self.assertEqual(observer.setup_calls, 1)
        self.assertIs(observer.task_manager, self.task_manager)

    async def test_removal_cancels_pending_setup_and_drops_queued_events(self):
        observer = SetupObserver(gated=True)
        self.worker.add_observer(observer)
        await self.worker.on_pipeline_started()
        await asyncio.wait_for(observer.setup_entered.wait(), timeout=2)
        await self.worker.remove_observer(observer)
        self.assertTrue(observer.setup_exited.is_set())
        self.assertFalse(observer.ready)
        self.assertEqual(observer.calls, [])
        self.assertEqual(len(self.task_manager.current_tasks()), 1)

    async def test_removal_before_proxy_starts_does_not_run_setup(self):
        observer = SetupObserver()
        self.worker.add_observer(observer)
        await self.worker.on_pipeline_started()
        await self.worker.remove_observer(observer)
        self.assertEqual(observer.setup_calls, 0)
        self.assertEqual(observer.calls, [])
        self.assertEqual(len(self.task_manager.current_tasks()), 1)

    async def test_cleanup_cancels_pending_setup(self):
        observer = SetupObserver(gated=True)
        self.worker.add_observer(observer)
        await self.worker.on_pipeline_started()
        await asyncio.wait_for(observer.setup_entered.wait(), timeout=2)
        await self.worker.cleanup()
        self.assertTrue(observer.setup_exited.is_set())
        self.assertFalse(observer.ready)
        self.assertEqual(observer.calls, [])
        self.assertFalse(self.task_manager.current_tasks())

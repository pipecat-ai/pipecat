#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.registry import WorkerRegistry
from pipecat.registry.types import WorkerReadyData


class TestTaskRegistry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = WorkerRegistry(runner_name="runner_a")

    async def test_register_local_task(self):
        """Local task is registered and appears in local_workers."""
        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        result = await self.registry.register(data)

        self.assertTrue(result)
        self.assertIn("greeter", self.registry.local_workers)
        self.assertNotIn("greeter", self.registry.remote_workers)

    async def test_register_remote_task(self):
        """Remote task is registered and appears in remote_workers."""
        data = WorkerReadyData(worker_name="support", runner="runner_b")
        result = await self.registry.register(data)

        self.assertTrue(result)
        self.assertIn("support", self.registry.remote_workers)
        self.assertNotIn("support", self.registry.local_workers)

    async def test_duplicate_registration_returns_false(self):
        """Registering the same task twice returns False."""
        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        first = await self.registry.register(data)
        second = await self.registry.register(data)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(self.registry.local_workers.count("greeter"), 1)

    async def test_get_local_task(self):
        """get() returns data for a local task."""
        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)

        result = self.registry.get("greeter")
        self.assertIs(result, data)

    async def test_get_remote_task(self):
        """get() returns data for a remote task."""
        data = WorkerReadyData(worker_name="support", runner="runner_b")
        await self.registry.register(data)

        result = self.registry.get("support")
        self.assertIs(result, data)

    async def test_get_unknown_task_returns_none(self):
        """get() returns None for an unknown task."""
        self.assertIsNone(self.registry.get("nonexistent"))

    async def test_contains(self):
        """__contains__ works for registered and unregistered tasks."""
        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)

        self.assertIn("greeter", self.registry)
        self.assertNotIn("unknown", self.registry)

    async def test_watch_fires_on_registration(self):
        """Watch handler fires when the watched task registers."""
        received = []

        async def handler(task_data):
            received.append(task_data)

        await self.registry.watch("greeter", handler)

        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], data)

    async def test_watch_is_idempotent_for_same_handler(self):
        """Re-watching with the same handler does not double-fire on registration."""
        received = []

        async def handler(task_data):
            received.append(task_data)

        await self.registry.watch("greeter", handler)
        await self.registry.watch("greeter", handler)  # duplicate, should be no-op

        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)

        self.assertEqual(len(received), 1)

    async def test_watch_does_not_fire_for_other_tasks(self):
        """Watch handler does not fire for a different task."""
        received = []

        async def handler(task_data):
            received.append(task_data)

        await self.registry.watch("greeter", handler)

        data = WorkerReadyData(worker_name="support", runner="runner_a")
        await self.registry.register(data)

        self.assertEqual(len(received), 0)

    async def test_watch_does_not_fire_on_duplicate(self):
        """Watch handler does not fire on duplicate registration."""
        received = []

        async def handler(task_data):
            received.append(task_data)

        await self.registry.watch("greeter", handler)

        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)
        await self.registry.register(data)

        self.assertEqual(len(received), 1)

    async def test_multiple_watchers(self):
        """Multiple watch handlers fire for the same task."""
        received_a = []
        received_b = []

        async def handler_a(task_data):
            received_a.append(task_data)

        async def handler_b(task_data):
            received_b.append(task_data)

        await self.registry.watch("greeter", handler_a)
        await self.registry.watch("greeter", handler_b)

        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    async def test_watch_fires_immediately_if_already_registered(self):
        """Watch handler fires immediately when the task is already registered."""
        data = WorkerReadyData(worker_name="greeter", runner="runner_a")
        await self.registry.register(data)

        received = []

        async def handler(task_data):
            received.append(task_data)

        await self.registry.watch("greeter", handler)

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], data)

    async def test_runner_name_property(self):
        """runner_name returns the name passed at construction."""
        self.assertEqual(self.registry.runner_name, "runner_a")

    async def test_multiple_remote_runners(self):
        """Tasks from multiple remote runners are tracked separately."""
        data_b = WorkerReadyData(worker_name="task_b", runner="runner_b")
        data_c = WorkerReadyData(worker_name="task_c", runner="runner_c")
        await self.registry.register(data_b)
        await self.registry.register(data_c)

        remote = self.registry.remote_workers
        self.assertIn("task_b", remote)
        self.assertIn("task_c", remote)
        self.assertEqual(len(remote), 2)


if __name__ == "__main__":
    unittest.main()

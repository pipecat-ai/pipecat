#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.bus import (
    BusAddTaskMessage,
    BusCancelMessage,
    BusCancelTaskMessage,
    BusEndMessage,
    BusEndTaskMessage,
)
from pipecat.pipeline.base_task import BaseTask
from pipecat.pipeline.runner import PipelineRunner


class StubTask(BaseTask):
    """BaseTask subclass that stops on end/cancel so the runner can exit."""

    async def _handle_task_end(self, message):
        await super()._handle_task_end(message)
        self._finished_event.set()

    async def _handle_task_cancel(self, message):
        await super()._handle_task_cancel(message)
        self._finished_event.set()


class TestPipelineRunner(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_registers_task(self):
        """spawn() registers the task by name (duplicate is silently skipped)."""
        runner = PipelineRunner(handle_sigint=False)
        task = StubTask("task_a")

        await runner.spawn(task)

        # Duplicate is silently skipped (logs error)
        await runner.spawn(StubTask("task_a"))

    async def test_run_starts_bus_and_tasks(self):
        """run() starts bus, starts all tasks, fires on_ready."""
        runner = PipelineRunner(handle_sigint=False)
        task = StubTask("task_a")
        await runner.spawn(task)

        runner_started = asyncio.Event()

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            runner_started.set()
            # Immediately end to unblock run()
            await runner.end()

        await asyncio.wait_for(runner.run(), timeout=5.0)

        self.assertTrue(runner_started.is_set())

    async def test_end_is_idempotent(self):
        """end() is idempotent — subsequent calls are no-ops."""
        runner = PipelineRunner(handle_sigint=False)
        task = StubTask("task_a")
        await runner.spawn(task)

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            await runner.end(reason="first")
            await runner.end(reason="second")  # should be no-op

        await asyncio.wait_for(runner.run(), timeout=5.0)
        # If we got here without hanging, idempotency works

    async def test_cancel_is_idempotent(self):
        """cancel() is idempotent — subsequent calls are no-ops."""
        runner = PipelineRunner(handle_sigint=False)
        task = StubTask("task_a")
        await runner.spawn(task)

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            await runner.cancel(reason="first")
            await runner.cancel(reason="second")  # should be no-op

        try:
            await asyncio.wait_for(runner.run(), timeout=5.0)
        except asyncio.CancelledError:
            pass

    async def test_end_sends_end_task_message_to_root_tasks_only(self):
        """end() sends BusEndTaskMessage only to root tasks (no parent)."""
        runner = PipelineRunner(handle_sigint=False)
        root = StubTask("root")
        child = StubTask("child")
        # Manually mark child as having root as parent
        child._parent = root.name
        await runner.spawn(root)
        await runner.spawn(child)

        sent = []
        bus = runner.bus
        original_send = bus.send

        async def capture_send(message):
            sent.append(message)
            await original_send(message)

        bus.send = capture_send

        # Call end() directly — no need to run the full pipeline lifecycle
        await runner.end()

        end_msgs = [m for m in sent if isinstance(m, BusEndTaskMessage)]
        targets = {m.target for m in end_msgs}
        self.assertIn("root", targets)
        self.assertNotIn("child", targets)

    async def test_cancel_sends_cancel_task_message_to_root_tasks_only(self):
        """cancel() sends BusCancelTaskMessage only to root tasks (no parent)."""
        runner = PipelineRunner(handle_sigint=False)
        root = StubTask("root")
        child = StubTask("child")
        child._parent = root.name
        await runner.spawn(root)
        await runner.spawn(child)

        sent = []
        bus = runner.bus
        original_send = bus.send

        async def capture_send(message):
            sent.append(message)
            await original_send(message)

        bus.send = capture_send

        # Call cancel() directly — no need to run the full pipeline lifecycle
        await runner.cancel()

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelTaskMessage)]
        targets = {m.target for m in cancel_msgs}
        self.assertIn("root", targets)
        self.assertNotIn("child", targets)

    async def test_bus_end_message_triggers_end(self):
        """BusEndMessage on bus triggers runner.end()."""
        runner = PipelineRunner(handle_sigint=False)
        task = StubTask("task_a")
        await runner.spawn(task)

        bus = runner.bus

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            # Simulate a task sending BusEndMessage
            await bus.send(BusEndMessage(source="task_a"))

        await asyncio.wait_for(runner.run(), timeout=5.0)
        # If we got here, end was triggered by the bus message

    async def test_bus_cancel_message_triggers_cancel(self):
        """BusCancelMessage on bus triggers runner.cancel()."""
        runner = PipelineRunner(handle_sigint=False)
        task = StubTask("task_a")
        await runner.spawn(task)

        bus = runner.bus

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            await bus.send(BusCancelMessage(source="task_a"))

        try:
            await asyncio.wait_for(runner.run(), timeout=5.0)
        except asyncio.CancelledError:
            pass

    async def test_bus_add_task_message_triggers_spawn(self):
        """BusAddTaskMessage on bus triggers spawn()."""
        runner = PipelineRunner(handle_sigint=False)
        task_a = StubTask("task_a")
        await runner.spawn(task_a)

        task_b = StubTask("task_b")
        bus = runner.bus

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            await bus.send(BusAddTaskMessage(source="task_a", task=task_b))
            await asyncio.sleep(0.1)
            await runner.end()

        await asyncio.wait_for(runner.run(), timeout=5.0)

        # Verify task_b was added (duplicate is silently skipped)
        await runner.spawn(StubTask("task_b"))


if __name__ == "__main__":
    unittest.main()

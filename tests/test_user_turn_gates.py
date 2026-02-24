import asyncio

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TriggerStartStrategy(BaseUserTurnStartStrategy):
    async def process_frame(self, frame):
        await super().process_frame(frame)


class TriggerStopStrategy(BaseUserTurnStopStrategy):
    async def process_frame(self, frame):
        await super().process_frame(frame)


@pytest.mark.asyncio
async def test_start_gate_denied():
    events = []

    async def start_gate(ctx):
        return False

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    controller = UserTurnController(
        user_turn_strategies=UserTurnStrategies(
            start=[TriggerStartStrategy()],
            stop=[],
            start_gate=start_gate,
        ),
    )

    @controller.event_handler("on_user_turn_started")
    async def on_start(controller, strategy, params):
        events.append("started")

    await controller.setup(task_manager)
    await controller.process_frame(
        TranscriptionFrame(text="hello", finalized=True, user_id="user", timestamp="now")
    )
    await controller._user_turn_strategies.start[0].trigger_user_turn_started()

    assert events == []


@pytest.mark.asyncio
async def test_stop_gate_denied_keeps_turn_open():
    events = []

    async def stop_gate(ctx):
        return False

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    controller = UserTurnController(
        user_turn_strategies=UserTurnStrategies(
            start=[TriggerStartStrategy()],
            stop=[TriggerStopStrategy()],
            stop_gate=stop_gate,
        ),
    )

    @controller.event_handler("on_user_turn_started")
    async def on_start(controller, strategy, params):
        events.append("started")

    @controller.event_handler("on_user_turn_stopped")
    async def on_stop(controller, strategy, params):
        events.append("stopped")

    await controller.setup(task_manager)

    await controller._user_turn_strategies.start[0].trigger_user_turn_started()
    await controller._user_turn_strategies.stop[0].trigger_user_turn_stopped()

    assert events == ["started"]
    assert controller._user_turn is True


@pytest.mark.asyncio
async def test_start_gate_timeout_denies_by_default():
    events = []

    async def slow_gate(ctx):
        await asyncio.sleep(0.2)
        return True

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    controller = UserTurnController(
        user_turn_strategies=UserTurnStrategies(
            start=[TriggerStartStrategy()],
            stop=[],
            start_gate=slow_gate,
            gate_timeout_secs=0.01,
        ),
    )

    @controller.event_handler("on_user_turn_started")
    async def on_start(controller, strategy, params):
        events.append("started")

    await controller.setup(task_manager)
    await controller._user_turn_strategies.start[0].trigger_user_turn_started()

    assert events == []


@pytest.mark.asyncio
async def test_start_gate_error_denies_when_configured():
    events = []

    async def error_gate(ctx):
        raise RuntimeError("boom")

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    controller = UserTurnController(
        user_turn_strategies=UserTurnStrategies(
            start=[TriggerStartStrategy()],
            stop=[],
            start_gate=error_gate,
            start_gate_on_error=False,
        ),
    )

    @controller.event_handler("on_user_turn_started")
    async def on_start(controller, strategy, params):
        events.append("started")

    await controller.setup(task_manager)
    await controller._user_turn_strategies.start[0].trigger_user_turn_started()

    assert events == []

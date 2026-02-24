import asyncio

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TriggerStartStrategy(BaseUserTurnStartStrategy):
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

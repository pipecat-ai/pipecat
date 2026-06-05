import asyncio
import gc
import warnings

import pytest

from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


@pytest.mark.asyncio
async def test_create_task_closes_coroutine_cancelled_before_first_run():
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def never_started():
        await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)

        task = task_manager.create_task(never_started(), "never-started")
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        gc.collect()

    assert not any("never awaited" in str(w.message) for w in caught)

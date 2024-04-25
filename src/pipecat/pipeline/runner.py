#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import signal

from pipecat.pipeline.task import PipelineTask
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class PipelineRunner:

    def __init__(self, name: str | None = None, handle_sigint: bool = True):
        self.id: int = obj_id()
        self.name: str = name or f"{self.__class__.__name__}#{obj_count(self)}"
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

        self._tasks = {}
        self._running = True

        if handle_sigint:
            self._setup_sigint()

    async def run(self, task: PipelineTask):
        logger.debug(f"Runner {self} started running {task}")
        self._running = True
        self._tasks[task.name] = task
        await task.run()
        del self._tasks[task.name]
        self._running = False
        logger.debug(f"Runner {self} finished running {task}")

    async def stop_when_done(self):
        logger.debug(f"Runner {self} scheduled to stop when all tasks are done")
        await asyncio.gather(*[t.stop_when_done() for t in self._tasks.values()])

    async def cancel(self):
        logger.debug(f"Canceling runner {self}")
        await asyncio.gather(*[t.cancel() for t in self._tasks.values()])

    def is_active(self):
        return self._running

    def _setup_sigint(self):
        self._loop.add_signal_handler(
            signal.SIGINT,
            lambda *args: asyncio.create_task(self._sigint_handler())
        )

    async def _sigint_handler(self):
        logger.warning(f"Ctrl-C detected. Canceling runner {self}")
        await self.cancel()

    def __str__(self):
        return self.name

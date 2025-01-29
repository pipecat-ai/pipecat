#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import signal

from loguru import logger

from pipecat.pipeline.task import PipelineTask
from pipecat.utils.asyncio import current_tasks
from pipecat.utils.utils import obj_count, obj_id


class PipelineRunner:
    def __init__(self, *, name: str | None = None, handle_sigint: bool = True):
        self.id: int = obj_id()
        self.name: str = name or f"{self.__class__.__name__}#{obj_count(self)}"

        self._tasks = {}
        self._sig_task = None

        if handle_sigint:
            self._setup_sigint()

    async def run(self, task: PipelineTask):
        logger.debug(f"Runner {self} started running {task}")
        self._tasks[task.name] = task
        await task.run()
        del self._tasks[task.name]
        # If we are cancelling through a signal, make sure we wait for it so
        # everything gets cleaned up nicely.
        if self._sig_task:
            await self._sig_task
        self._print_dangling_tasks()
        logger.debug(f"Runner {self} finished running {task}")

    async def stop_when_done(self):
        logger.debug(f"Runner {self} scheduled to stop when all tasks are done")
        await asyncio.gather(*[t.stop_when_done() for t in self._tasks.values()])

    async def cancel(self):
        logger.debug(f"Canceling runner {self}")
        await asyncio.gather(*[t.cancel() for t in self._tasks.values()])

    def _setup_sigint(self):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda *args: self._sig_handler())
        loop.add_signal_handler(signal.SIGTERM, lambda *args: self._sig_handler())

    def _sig_handler(self):
        if not self._sig_task:
            self._sig_task = asyncio.create_task(self._sig_cancel())

    async def _sig_cancel(self):
        logger.warning(f"Interruption detected. Canceling runner {self}")
        await self.cancel()

    def _print_dangling_tasks(self):
        tasks = [t.get_name() for t in current_tasks()]
        if tasks:
            logger.warning(f"Dangling tasks detected: {tasks}")

    def __str__(self):
        return self.name

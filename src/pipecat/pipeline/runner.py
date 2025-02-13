#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import gc
import signal
from typing import Optional

from loguru import logger

from pipecat.pipeline.task import PipelineTask
from pipecat.utils.utils import obj_count, obj_id


class PipelineRunner:
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        handle_sigint: bool = True,
        force_gc: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.id: int = obj_id()
        self.name: str = name or f"{self.__class__.__name__}#{obj_count(self)}"

        self._tasks = {}
        self._sig_task = None
        self._force_gc = force_gc
        self._loop = loop or asyncio.get_running_loop()

        if handle_sigint:
            self._setup_sigint()

    async def run(self, task: PipelineTask):
        logger.debug(f"Runner {self} started running {task}")
        self._tasks[task.name] = task
        task.set_event_loop(self._loop)
        await task.run()
        del self._tasks[task.name]
        # If we are cancelling through a signal, make sure we wait for it so
        # everything gets cleaned up nicely.
        if self._sig_task:
            await self._sig_task
        if self._force_gc:
            self._gc_collect()
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

    def _gc_collect(self):
        collected = gc.collect()
        logger.debug(f"Garbage collector: collected {collected} objects.")
        logger.debug(f"Garbage collector: uncollectable objects {gc.garbage}")

    def __str__(self):
        return self.name

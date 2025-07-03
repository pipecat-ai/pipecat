#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline runner for managing pipeline task execution.

This module provides the PipelineRunner class that handles the execution
of pipeline tasks with signal handling, garbage collection, and lifecycle
management.
"""

import asyncio
import gc
import signal
from typing import Optional

from loguru import logger

from pipecat.pipeline.base_task import PipelineTaskParams
from pipecat.pipeline.task import PipelineTask
from pipecat.utils.base_object import BaseObject


class PipelineRunner(BaseObject):
    """Manages the execution of pipeline tasks with lifecycle and signal handling.

    Provides a high-level interface for running pipeline tasks with automatic
    signal handling (SIGINT/SIGTERM), optional garbage collection, and proper
    cleanup of resources.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        handle_sigint: bool = True,
        force_gc: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize the pipeline runner.

        Args:
            name: Optional name for the runner instance.
            handle_sigint: Whether to automatically handle SIGINT/SIGTERM signals.
            force_gc: Whether to force garbage collection after task completion.
            loop: Event loop to use. If None, uses the current running loop.
        """
        super().__init__(name=name)

        self._tasks = {}
        self._sig_task = None
        self._force_gc = force_gc
        self._loop = loop or asyncio.get_running_loop()

        if handle_sigint:
            self._setup_sigint()

    async def run(self, task: PipelineTask):
        """Run a pipeline task to completion.

        Args:
            task: The pipeline task to execute.
        """
        logger.debug(f"Runner {self} started running {task}")
        self._tasks[task.name] = task
        params = PipelineTaskParams(loop=self._loop)
        await task.run(params)
        del self._tasks[task.name]

        # Cleanup base object.
        await self.cleanup()

        # If we are cancelling through a signal, make sure we wait for it so
        # everything gets cleaned up nicely.
        if self._sig_task:
            await self._sig_task

        if self._force_gc:
            self._gc_collect()

        logger.debug(f"Runner {self} finished running {task}")

    async def stop_when_done(self):
        """Schedule all running tasks to stop when their current processing is complete."""
        logger.debug(f"Runner {self} scheduled to stop when all tasks are done")
        await asyncio.gather(*[t.stop_when_done() for t in self._tasks.values()])

    async def cancel(self):
        """Cancel all running tasks immediately."""
        logger.debug(f"Cancelling runner {self}")
        await asyncio.gather(*[t.cancel() for t in self._tasks.values()])

    def _setup_sigint(self):
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda *args: self._sig_handler())
        loop.add_signal_handler(signal.SIGTERM, lambda *args: self._sig_handler())

    def _sig_handler(self):
        """Handle interrupt signals by cancelling all tasks."""
        if not self._sig_task:
            self._sig_task = asyncio.create_task(self._sig_cancel())

    async def _sig_cancel(self):
        """Cancel all running tasks due to signal interruption."""
        logger.warning(f"Interruption detected. Cancelling runner {self}")
        await self.cancel()

    def _gc_collect(self):
        """Force garbage collection and log results."""
        collected = gc.collect()
        logger.debug(f"Garbage collector: collected {collected} objects.")
        logger.debug(f"Garbage collector: uncollectable objects {gc.garbage}")

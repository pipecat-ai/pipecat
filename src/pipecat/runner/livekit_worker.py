#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit Agent Worker runner for Pipecat.

This module provides a runner that allows Pipecat pipelines to operate as LiveKit
Agents (Workers), handling job assignment and room connections automatically.
"""

import asyncio
from typing import Awaitable, Callable, Tuple

from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.livekit.transport import LiveKitTransport

try:
    from livekit import rtc
    from livekit.agents import JobContext, WorkerOptions, cli
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use LiveKit agents, you need to `pip install livekit-agents`."
    )
    raise Exception(f"Missing module: {e}")


class LiveKitWorkerRunner:
    """Runner for LiveKit Agents (Workers).

    This runner wraps the `livekit-agents` worker loop to run Pipecat pipelines
    in response to LiveKit job assignments (e.g., when a user joins a room).
    """

    def __init__(self):
        """Initialize the LiveKitWorkerRunner."""
        self._pipeline_factory = None

    def set_pipeline_factory(
        self, factory: Callable[[rtc.Room], Awaitable[Tuple[LiveKitTransport, Pipeline]]]
    ):
        """Set the factory function to create pipelines for assigned rooms.

        Args:
           factory: A callable that takes an `rtc.Room` and returns a tuple of
               `(LiveKitTransport, Pipeline)`.
        """
        self._pipeline_factory = factory

    async def _entrypoint(self, ctx: JobContext):
        """Internal entrypoint for the LiveKit worker job.

        Args:
            ctx: The LiveKit job context.
        """
        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect()
        logger.info(f"Connected to room {ctx.room.name}")

        if not self._pipeline_factory:
            logger.error("No pipeline factory set")
            return

        transport, pipeline = await self._pipeline_factory(ctx.room)

        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        await runner.run(task)

    def run(self):
        """Start the worker to listen for jobs.

        This method blocks until the worker is stopped.
        """
        cli.run_app(WorkerOptions(entrypoint_fnc=self._entrypoint))

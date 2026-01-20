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
    from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
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
        self, factory: Callable[[rtc.Room], Awaitable[Tuple[LiveKitTransport, PipelineTask]]]
    ):
        """Set the factory function to create pipelines for assigned rooms.

        Args:
           factory: A callable that takes an `rtc.Room` and returns a tuple of
               `(LiveKitTransport, PipelineTask)`.
        """
        self._pipeline_factory = factory

    async def _request_fnc(self, req: JobRequest):
        """Handle incoming job requests.
        
        This function is called when the LiveKit server dispatches a job to this worker.
        By default, we accept all jobs.
        
        Args:
            req: The job request from LiveKit.
        """
        logger.info(f"🔔 JOB REQUEST RECEIVED: room={req.room.name}, id={req.id}")
        logger.info(f"   Job metadata: {req.room.metadata}")
        logger.info(f"   Accepting job...")
        await req.accept(
            name="Pipecat Agent",
            identity=f"pipecat-agent-{req.id[:8]}",
        )
        logger.info(f"✅ Job accepted for room: {req.room.name}")

    async def _entrypoint(self, ctx: JobContext):
        """Internal entrypoint for the LiveKit worker job.

        Args:
            ctx: The LiveKit job context.
        """
        logger.info(f"🚀 ENTRYPOINT CALLED for room {ctx.room.name}")
        logger.info(f"   Job ID: {ctx.job.id}")
        
        logger.info(f"🔗 Connecting to room {ctx.room.name}...")
        await ctx.connect()
        logger.info(f"✅ Connected to room {ctx.room.name}")
        
        room_sid = await ctx.room.sid
        logger.info(f"   Room SID: {room_sid}")

        if not self._pipeline_factory:
            logger.error("❌ No pipeline factory set")
            return

        logger.info("🏗️ Creating pipeline via factory...")
        transport, task = await self._pipeline_factory(ctx.room)
        logger.info("✅ Pipeline created, starting runner...")

        runner = PipelineRunner()
        await runner.run(task)
        logger.info("🏁 Pipeline runner finished")

    def run(self):
        """Start the worker to listen for jobs.

        This method blocks until the worker is stopped.
        """
        logger.info("🤖 Starting LiveKit Worker Runner...")
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=self._entrypoint,
                request_fnc=self._request_fnc,
            )
        )

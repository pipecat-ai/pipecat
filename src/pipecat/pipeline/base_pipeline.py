#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base pipeline implementation for frame processing."""

import asyncio
import time
from collections.abc import Sequence

from loguru import logger

from pipecat.observers.base_observer import ProcessorSetUp
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup


class BasePipeline(FrameProcessor):
    """Base class for all pipeline implementations."""

    def __init__(self, **kwargs):
        """Initialize the base pipeline."""
        super().__init__(**kwargs)

    async def _setup_processors(
        self, processors: Sequence[FrameProcessor], setup: FrameProcessorSetup
    ):
        """Set up the given processors concurrently.

        A processor that fails to set up reports it as an error and leaves the
        rest of the pipeline to carry on, the same way a failure while handling
        a frame is reported. Every failure is reported, not just the first.

        The failure is permanent whatever caused it, since setting up is not
        attempted again: the processor is left half-built for the rest of the
        session, so it loses its usability and a switcher can move off it.

        Call it right after this pipeline's own setup, once it has released
        itself with ``_setup_finished()``. Each processor holds the StartFrame
        until its own setup is done, so one slow processor doesn't block the
        ones ahead of it. The processors are held before anything else runs,
        so the StartFrame can't get past one whose setup() hasn't begun yet.

        Args:
            processors: The processors to set up.
            setup: Configuration for frame processor setup.
        """
        tasks = [self._create_setup_task(p, setup) for p in processors]
        await asyncio.gather(*tasks)

    def _create_setup_task(
        self, processor: FrameProcessor, setup: FrameProcessorSetup
    ) -> asyncio.Task:
        """Hold the processor's StartFrame and set it up in a task of its own."""
        processor._setup_started()
        task = self.create_task(
            self._setup_processor(processor, setup), f"{self}::setup::{processor}"
        )
        task.add_done_callback(lambda task: self._setup_task_done(processor, task))
        return task

    def _setup_task_done(self, processor: FrameProcessor, task: asyncio.Task):
        """Release the processor once it is set up, unless setting it up was cancelled."""
        if not task.cancelled():
            processor._setup_finished()

    async def _setup_processor(self, processor: FrameProcessor, setup: FrameProcessorSetup):
        """Set the processor up, reporting a failure as a permanent error."""
        started_at_ns = time.monotonic_ns()
        try:
            await processor.setup(setup)
        except Exception as e:
            await processor.push_error(
                f"Error setting up processor: {e}", exception=e, force_treat_as_permanent=True
            )
        if setup.observer:
            await setup.observer.on_processor_setup(
                ProcessorSetUp(
                    processor=processor,
                    started_at_ns=started_at_ns,
                    finished_at_ns=time.monotonic_ns(),
                )
            )

    async def _cleanup_processors(self, processors: Sequence[FrameProcessor]):
        """Clean up the given processors concurrently.

        Tearing down is best effort: a processor that raises is logged and the
        rest are still released. Cleaning up runs after a processor failed to
        set up, or was abandoned part-way through being set up, so some of them
        are released from a state they never finished reaching.

        Args:
            processors: The processors to clean up.
        """

        async def cleanup_processor(processor: FrameProcessor):
            try:
                await processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {processor}: {e}")

        await asyncio.gather(*[cleanup_processor(p) for p in processors])

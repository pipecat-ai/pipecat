#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base pipeline implementation for frame processing."""

import asyncio
from collections.abc import Sequence

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

        Args:
            processors: The processors to set up.
            setup: Configuration for frame processor setup.
        """

        async def setup_processor(processor: FrameProcessor):
            try:
                await processor.setup(setup)
            except Exception as e:
                await processor.push_error(f"Error setting up processor: {e}", exception=e)

        await asyncio.gather(*[setup_processor(p) for p in processors])

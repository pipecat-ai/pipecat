#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sequential pipeline merging for Pipecat.

This module provides a pipeline implementation that sequentially merges
the output from multiple pipelines, processing them one after another
in a specified order.
"""

from typing import List

from pipecat.frames.frames import EndFrame, EndPipeFrame
from pipecat.pipeline.pipeline import Pipeline


class SequentialMergePipeline(Pipeline):
    """Pipeline that sequentially merges output from multiple pipelines.

    This pipeline merges the sink queues from a list of pipelines by processing
    frames from each pipeline's sink sequentially in the order specified. Each
    pipeline runs to completion before the next one begins processing.
    """

    def __init__(self, pipelines: List[Pipeline]):
        """Initialize the sequential merge pipeline.

        Args:
            pipelines: List of pipelines to merge sequentially. Pipelines will
                      be processed in the order they appear in this list.
        """
        super().__init__([])
        self.pipelines = pipelines

    async def run_pipeline(self):
        """Run all pipelines sequentially and merge their output.

        Processes each pipeline in order, consuming all frames from each
        pipeline's sink until an EndFrame or EndPipeFrame is encountered,
        then moves to the next pipeline. After all pipelines complete,
        sends a final EndFrame to signal completion.
        """
        for idx, pipeline in enumerate(self.pipelines):
            while True:
                frame = await pipeline.sink.get()
                if isinstance(frame, EndFrame) or isinstance(frame, EndPipeFrame):
                    break
                await self.sink.put(frame)

        await self.sink.put(EndFrame())

#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from tests.utils import run_test


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_single(self):
        pipeline = Pipeline([IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(pipeline, frames_to_send, expected_returned_frames)

    async def test_pipeline_multiple(self):
        identity1 = IdentityFilter()
        identity2 = IdentityFilter()
        identity3 = IdentityFilter()

        pipeline = Pipeline([identity1, identity2, identity3])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(pipeline, frames_to_send, expected_returned_frames)


class TestParallelPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_single(self):
        pipeline = ParallelPipeline([IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(pipeline, frames_to_send, expected_returned_frames)

    async def test_parallel_multiple(self):
        """Should only passthrough one instance of TextFrame."""
        pipeline = ParallelPipeline([IdentityFilter()], [IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(pipeline, frames_to_send, expected_returned_frames)

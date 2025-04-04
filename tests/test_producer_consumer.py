#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import Frame, InputAudioRawFrame, TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.consumer_processor import ConsumerProcessor
from pipecat.processors.producer_processor import ProducerProcessor
from pipecat.tests.utils import SleepFrame, run_test


async def text_frame_filter(frame: Frame):
    return isinstance(frame, TextFrame)


class TestProducerConsumerProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_produce_passthrough(self):
        producer = ProducerProcessor(filter=text_frame_filter)
        consumer = ConsumerProcessor(producer=producer)
        pipeline = Pipeline([producer, consumer])
        frames_to_send = [
            TextFrame("Hello!"),
            SleepFrame(),  # So we let the consumer go first.
        ]
        expected_down_frames = [
            TextFrame,  # Consumer frame
            TextFrame,  # Pass-through frame
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_produce_no_passthrough(self):
        producer = ProducerProcessor(filter=text_frame_filter, passthrough=False)
        consumer = ConsumerProcessor(producer=producer)
        pipeline = Pipeline([producer, consumer])
        frames_to_send = [TextFrame("Hello!")]
        expected_down_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_produce_multiple_consumer_no_passthrough(self):
        producer = ProducerProcessor(filter=text_frame_filter, passthrough=False)
        consumer1 = ConsumerProcessor(producer=producer)
        consumer2 = ConsumerProcessor(producer=producer)
        pipeline = Pipeline([producer, consumer1, consumer2])
        frames_to_send = [TextFrame("Hello!")]
        expected_down_frames = [
            TextFrame,  # From consumer1 or consumer2 (depending on who runs first)
            TextFrame,  # From consumer1 or consumer2 (depending on who runs first)
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_produce_parallel_pipeline_no_passthrough(self):
        producer = ProducerProcessor(filter=text_frame_filter, passthrough=False)
        consumer = ConsumerProcessor(producer=producer)
        pipeline = Pipeline([ParallelPipeline([producer], [consumer])])
        frames_to_send = [TextFrame("Hello!")]
        expected_down_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_produce_passthrough_transform(self):
        async def audio_transformer(_: Frame) -> Frame:
            return InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1)

        producer = ProducerProcessor(filter=text_frame_filter, transformer=audio_transformer)
        consumer = ConsumerProcessor(producer=producer)
        pipeline = Pipeline([producer, consumer])
        frames_to_send = [
            TextFrame("Hello!"),
            SleepFrame(),  # So we let the consumer go first.
        ]
        expected_down_frames = [
            InputAudioRawFrame,  # Consumer frame
            TextFrame,  # Pass-through frame
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_produce_passthrough_consumer_transform(self):
        async def audio_transformer(_: Frame) -> Frame:
            return InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1)

        producer = ProducerProcessor(filter=text_frame_filter)
        consumer = ConsumerProcessor(producer=producer, transformer=audio_transformer)
        pipeline = Pipeline([producer, consumer])
        frames_to_send = [
            TextFrame("Hello!"),
            SleepFrame(),  # So we let the consumer go first.
        ]
        expected_down_frames = [
            InputAudioRawFrame,  # Consumer frame
            TextFrame,  # Pass-through frame
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

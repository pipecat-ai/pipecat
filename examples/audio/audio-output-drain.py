#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Drain a final partial audio chunk before a downstream marker.

Run with `uv run python examples/audio/audio-output-drain.py` from the
repository root. The example writes to a mock transport and needs no credentials.
"""

import asyncio

from pipecat.frames.frames import EndFrame, Frame, OutputAudioDrainFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.base_worker import WorkerParams


class PrintingOutput(BaseOutputTransport):
    """Print audio writes instead of sending audio to a client."""

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        print(f"Wrote {len(frame.audio)} bytes of PCM")
        return True


class DrainObserver(FrameProcessor):
    """Observe the drain marker after the output transport."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioDrainFrame):
            print("Received drain marker after the PCM writes")
        await self.push_frame(frame, direction)


async def main():
    sample_rate = 16000
    output = PrintingOutput(
        TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=sample_rate,
            audio_out_end_silence_secs=0,
        )
    )
    worker = PipelineWorker(Pipeline([output, DrainObserver()]))

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        half_chunk_samples = output.audio_chunk_size // 4
        audio = b"\x01\x00" * (3 * half_chunk_samples)
        await worker.queue_frames(
            [
                OutputAudioRawFrame(audio, sample_rate, 1),
                OutputAudioDrainFrame(),
                EndFrame(),
            ]
        )

    await worker.run(WorkerParams(task_manager=TaskManager()))


if __name__ == "__main__":
    asyncio.run(main())

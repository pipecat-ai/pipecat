"""Echo bot for the transport latency benchmark.

One bot file for both transports under test: ``-t moq`` and ``-t webrtc``.
The pipeline is the identical minimal loop for both (fairness control #1):

    transport.input() -> EchoProcessor -> transport.output()

No VAD, no AI services, no RTVI. 48 kHz mono in both directions on both
transports so neither path pays a resampler (fairness control #2).

A ``LatencyObserver`` stamps every audio frame entering from the input
transport and leaving toward the output transport; per-chunk bot-internal
times are written as JSON to ``$BENCH_BOT_STATS`` at shutdown so the client's
round-trip numbers can be decomposed into wire+codec vs pipeline overhead.

Env vars:
    BENCH_JITTER_MS   MoQ receive jitter buffer (audio_in_max_latency_ms),
                      default 60. The benchmark client pins its own MoQ
                      subscribe latency to the same value.
    BENCH_BOT_STATS   Path to write the observer's JSON stats.

Usage:
    uv run python benchmarks/transport_latency/echo_bot.py -t webrtc
    uv run python benchmarks/transport_latency/echo_bot.py -t moq [--moq-connect ...]
"""

import json
import os
import time

import numpy as np
from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.moq.transport import MOQParams
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 48000
JITTER_MS = int(os.getenv("BENCH_JITTER_MS", "60"))

transport_params = {
    "moq": lambda: MOQParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_passthrough=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_sample_rate=SAMPLE_RATE,
        audio_in_max_latency_ms=JITTER_MS,
        audio_out_frame_ms=20,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_passthrough=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_sample_rate=SAMPLE_RATE,
    ),
}


class EchoProcessor(FrameProcessor):
    """Turn input audio straight into output audio; pass everything else through."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
        else:
            await self.push_frame(frame, direction)


class LatencyObserver(BaseObserver):
    """Stamp audio frames at the pipeline's edges, keyed by cumulative samples."""

    def __init__(self):
        super().__init__()
        self._in: list[tuple[int, float]] = []  # (cum samples, t_mono)
        self._out: list[tuple[int, float]] = []
        self._in_samples = 0
        self._out_samples = 0

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, InputAudioRawFrame):
            self._in_samples += len(frame.audio) // 2
            self._in.append((self._in_samples, time.monotonic()))
            # Periodic dump: shutdown-time writes can lose the race against
            # uvicorn's own SIGINT handling.
            stats_path = os.getenv("BENCH_BOT_STATS")
            if stats_path and len(self._in) % 250 == 0:
                self.dump(stats_path)
        elif isinstance(frame, OutputAudioRawFrame) and not isinstance(frame, InputAudioRawFrame):
            # Stamp only the push into the output transport (destination side).
            if data.destination.__class__.__name__.endswith("OutputTransport"):
                self._out_samples += len(frame.audio) // 2
                self._out.append((self._out_samples, time.monotonic()))

    def dump(self, path: str) -> None:
        n = min(len(self._in), len(self._out))
        deltas_ms = [
            (self._out[i][1] - self._in[i][1]) * 1000.0
            for i in range(n)
            if self._out[i][0] == self._in[i][0]
        ]
        stats = {
            "frames_in": len(self._in),
            "frames_out": len(self._out),
            "matched": len(deltas_ms),
            "internal_ms": {
                "p50": float(np.percentile(deltas_ms, 50)) if deltas_ms else None,
                "p95": float(np.percentile(deltas_ms, 95)) if deltas_ms else None,
                "max": float(max(deltas_ms)) if deltas_ms else None,
            },
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"LatencyObserver stats -> {path}: {stats['internal_ms']}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Echo bot starting ({transport.__class__.__name__}, {JITTER_MS}ms jitter)")

    observer = LatencyObserver()
    pipeline = Pipeline([transport.input(), EchoProcessor(), transport.output()])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(),
        observers=[observer],
        idle_timeout_secs=None,  # a pure echo loop generates no "activity"
    )

    transport_name = transport.__class__.__name__
    if transport_name == "MOQTransport":

        @transport.event_handler("on_disconnected")
        async def on_disconnected(transport):
            logger.info("MoQ disconnected")
            await worker.cancel()

    else:

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        stats_path = os.getenv("BENCH_BOT_STATS")
        if stats_path:
            observer.dump(stats_path)
        if transport_name == "MOQTransport":
            await transport.disconnect()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

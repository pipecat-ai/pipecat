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
    BENCH_AUDIO_OUT_10MS_CHUNKS  Overrides TransportParams.audio_out_10ms_chunks
                         (MediaSender's accumulate-then-flush size, shared
                         by both transports — base_transport.py). Unset
                         uses the framework default (4 = 40ms).
    BENCH_JITTER_MS      MoQ receive jitter buffer (audio_in_max_latency_ms),
                         default 60. The benchmark client pins its own MoQ
                         subscribe latency to the same value.
    BENCH_WEBRTC_PREFETCH  Overrides aiortc's hardcoded audio JitterBuffer
                         prefetch (see webrtc_client.configure_audio_jitter_prefetch)
                         for THIS process's receiver — i.e. the bot's own
                         receive side. Unset by default (stock aiortc
                         behavior, prefetch=4); the benchmark harness sets
                         it to match the client's --webrtc-prefetch so both
                         legs of a webrtc-* scenario measure the same
                         configuration, not an unpatched bot against a
                         patched client.
    BENCH_RTT_LOG        Set to "1" to instrument this process's own RTT
                         layers and fold them into BENCH_BOT_STATS as
                         "rtt_breakdown": aiortc RTCP RTT, jitter-buffer
                         hold time, and opus encode/decode cost on the
                         bot's receiver/sender (webrtc only — a no-op on
                         moq, which has none of these), plus
                         BaseOutputTransport.MediaSender buffer occupancy
                         and flush events (both transports — MediaSender
                         is shared code), and pipecat's send-side adapter
                         (RawAudioTrack on webrtc, MOQOutputTransport.
                         write_audio_frame on moq) queue depth/send
                         events. The flush/sent events are position-keyed
                         (cum_samples, t_mono) using the same numbering as
                         LatencyObserver's out_raw, so a specific marker's
                         wait at each pipeline stage can be measured
                         directly instead of inferred from occupancy
                         snapshots. Off by default since it monkeypatches
                         process-wide state; opt in for diagnostic runs.
    BENCH_BOT_STATS      Path to write the observer's JSON stats.

Usage:
    uv run python benchmarks/transport_latency/echo_bot.py -t webrtc
    uv run python benchmarks/transport_latency/echo_bot.py -t moq [--moq-connect ...]
"""

import asyncio
import json
import os
import time
from collections import deque

import numpy as np
from loguru import logger
from stats import summarize

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.moq.transport import MOQParams
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 48000
JITTER_MS = int(os.getenv("BENCH_JITTER_MS", "60"))

_webrtc_prefetch = os.getenv("BENCH_WEBRTC_PREFETCH")
if _webrtc_prefetch is not None:
    from webrtc_client import configure_audio_jitter_prefetch

    configure_audio_jitter_prefetch(int(_webrtc_prefetch))

# Opt-in RTT-layer instrumentation for this process (bot side) — see
# BENCH_RTT_LOG in the module docstring. Applied at import time so it's
# active before any pipeline/track construction; harmless no-op on moq for
# the aiortc-specific pieces.
RTT_LOG = os.getenv("BENCH_RTT_LOG") == "1"

_jb_delays: dict[int, list[float]] | None = None
_jb_creation_order: list[int] | None = None
_media_sender_samples: list[float] | None = None

_opus_encode_ms: list[float] | None = None
_opus_decode_ms: list[float] | None = None

if RTT_LOG:
    from webrtc_client import instrument_jitter_buffer_timing, instrument_opus_codec_timing

    _jb_delays, _jb_creation_order, _ = instrument_jitter_buffer_timing()
    _opus_encode_ms, _opus_decode_ms, _ = instrument_opus_codec_timing()

    _media_sender_samples = []
    # (cum_samples, t_mono) at the moment each flushed chunk is actually
    # queued for send — common to both transports (base_output.py's
    # MediaSender). Position-keyed like LatencyObserver's out_raw, so a
    # given marker's exact wait *inside* this accumulate-then-flush buffer
    # can be measured directly (t_flush - t_bot_out for the same cum_samples)
    # instead of inferred from occupancy snapshots, which measure the wrong
    # thing: a chunk arriving at LOW occupancy is the one that waits LONGEST
    # for its block to fill, not the one that gets through fastest.
    _media_sender_flush_events: list[tuple[float, int]] = []
    _media_sender_cum_samples = [0]
    _instrumented_media_queues: set[int] = set()
    _orig_media_sender_handle = BaseOutputTransport.MediaSender.handle_audio_frame

    async def _patched_media_sender_handle(self, frame):
        # Buffer occupancy (ms of audio already queued) just before this
        # frame is folded in — see base_output.py's accumulate-then-flush
        # MediaSender, shared by every output transport (audio_out_10ms_chunks).
        bytes_per_sample = 2 * (getattr(self._params, "audio_out_channels", 1) or 1)
        ms_before = (len(self._audio_buffer) / bytes_per_sample) / self._sample_rate * 1000.0
        _media_sender_samples.append(ms_before)

        qid = id(self._audio_queue)
        if qid not in _instrumented_media_queues:
            _instrumented_media_queues.add(qid)
            orig_put = self._audio_queue.put

            async def _patched_put(chunk, _orig_put=orig_put):
                # //2 (not channel-aware) to match LatencyObserver's own
                # cum_samples convention exactly — it always counts raw
                # int16 words regardless of channel layout, and marker
                # positions are keyed against that same convention.
                _media_sender_cum_samples[0] += len(chunk.audio) // 2
                _media_sender_flush_events.append((_media_sender_cum_samples[0], time.monotonic()))
                return await _orig_put(chunk)

            self._audio_queue.put = _patched_put

        return await _orig_media_sender_handle(self, frame)

    BaseOutputTransport.MediaSender.handle_audio_frame = _patched_media_sender_handle

    # RawAudioTrack (webrtc only) — pipecat's own send-side adapter feeding
    # aiortc's RTCRtpSender (src/pipecat/transports/smallwebrtc/transport.py),
    # not part of aiortc itself. Tracks queue depth (in 10ms chunks) at every
    # fill (add_audio_bytes, driven by MediaSender's 40ms flush) and drain
    # (recv, driven by aiortc's RTCRtpSender pulling one 10ms frame at a
    # time) to see whether the queue grows over a run rather than staying
    # steady-state, plus (cum_samples, t_mono) for every REAL (non-silence)
    # chunk actually handed to aiortc, position-keyed like the MediaSender
    # flush events above.
    _raw_audio_track_events: list[tuple[float, int, str]] = []
    _raw_audio_track_sent_events: list[tuple[float, int]] = []
    _raw_audio_track_cum_samples = [0]
    try:
        from pipecat.transports.smallwebrtc.transport import RawAudioTrack

        # recv() awaits a wall-clock pacing sleep BEFORE checking whether
        # self._chunk_queue is empty, so a pre-call bool(self._chunk_queue)
        # check races with add_audio_bytes filling the queue during that
        # sleep — a "was_real" guess taken beforehand can miss real chunks.
        # Swap in a deque subclass so popleft() itself (the actual
        # consumption point, unaffected by that race) does the recording.
        class _InstrumentedChunkQueue(deque):
            def popleft(self):
                chunk, fut = super().popleft()
                _raw_audio_track_cum_samples[0] += len(chunk) // 2
                _raw_audio_track_sent_events.append(
                    (_raw_audio_track_cum_samples[0], time.monotonic())
                )
                return chunk, fut

        _orig_add_audio_bytes = RawAudioTrack.add_audio_bytes
        _orig_track_recv = RawAudioTrack.recv

        def _patched_add_audio_bytes(self, audio_bytes):
            if not isinstance(self._chunk_queue, _InstrumentedChunkQueue):
                self._chunk_queue = _InstrumentedChunkQueue(self._chunk_queue)
            result = _orig_add_audio_bytes(self, audio_bytes)
            _raw_audio_track_events.append((time.monotonic(), len(self._chunk_queue), "add"))
            return result

        async def _patched_track_recv(self):
            frame = await _orig_track_recv(self)
            _raw_audio_track_events.append((time.monotonic(), len(self._chunk_queue), "recv"))
            return frame

        RawAudioTrack.add_audio_bytes = _patched_add_audio_bytes
        RawAudioTrack.recv = _patched_track_recv
    except ImportError:
        pass  # moq bot process never imports smallwebrtc — nothing to instrument

    # MOQOutputTransport (moq only) — the shared-code counterpart to
    # RawAudioTrack: where a flushed MediaSender chunk actually leaves the
    # bot process, via moq.AudioProducer.write() (see publish_audio in
    # transport.py). Confirmed by reading that method: it only blocks if the
    # pacing clock gets audio_out_max_buffer_ms (25s) ahead of wall-clock —
    # never in a real-time echo stream — so unlike RawAudioTrack it has no
    # per-chunk real-time throttle of its own.
    _moq_send_events: list[tuple[float, int]] = []
    _moq_send_cum_samples = [0]
    try:
        from pipecat.transports.moq.transport import MOQOutputTransport

        _orig_moq_write_audio_frame = MOQOutputTransport.write_audio_frame

        async def _patched_moq_write_audio_frame(self, frame):
            result = await _orig_moq_write_audio_frame(self, frame)
            _moq_send_cum_samples[0] += len(frame.audio) // 2  # s16 mono
            _moq_send_events.append((_moq_send_cum_samples[0], time.monotonic()))
            return result

        MOQOutputTransport.write_audio_frame = _patched_moq_write_audio_frame
    except ImportError:
        pass

# base_transport.py's MediaSender only flushes once it has accumulated a
# full audio_out_10ms_chunks-sized block (default 4 = 40ms), batching
# audio before either transport's send-adapter ever sees it — see the
# rtt_breakdown leg2b/leg2c analysis in RUNBOOK for why this drives most
# of webrtc's non-jitter-buffer send-side latency. Override for diagnostic
# runs; unset uses base_transport.py's own default (4).
_audio_out_10ms_chunks_env = os.getenv("BENCH_AUDIO_OUT_10MS_CHUNKS")
_audio_out_10ms_chunks_kwargs = (
    {"audio_out_10ms_chunks": int(_audio_out_10ms_chunks_env)}
    if _audio_out_10ms_chunks_env is not None
    else {}
)

transport_params = {
    "moq": lambda: MOQParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_passthrough=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_sample_rate=SAMPLE_RATE,
        audio_in_max_latency_ms=JITTER_MS,
        audio_out_frame_ms=20,
        **_audio_out_10ms_chunks_kwargs,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_passthrough=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_sample_rate=SAMPLE_RATE,
        **_audio_out_10ms_chunks_kwargs,
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

    def dump(self, path: str, extra: dict | None = None) -> None:
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
            # Raw per-frame (cumulative sample count, monotonic timestamp)
            # log, keyed the same way as the aggregates above. Lets an
            # external client — on the same machine, sharing the OS's
            # monotonic clock — correlate a specific marker's sample offset
            # to this bot's own receive/send stamps for ladder-diagram
            # reconstruction.
            "in_raw": self._in,
            "out_raw": self._out,
        }
        if extra:
            stats["rtt_breakdown"] = extra
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

    rtcp_rtt_samples: list[float] = []
    rtcp_stop = asyncio.Event()
    rtcp_poll_task: asyncio.Task | None = None

    async def poll_bot_rtcp_rtt(pc) -> None:
        while not rtcp_stop.is_set():
            try:
                report = await pc.getStats()
            except Exception:
                report = {}
            for stat in report.values():
                if stat.type == "remote-inbound-rtp" and stat.roundTripTime is not None:
                    rtcp_rtt_samples.append(stat.roundTripTime * 1000.0)
            try:
                await asyncio.wait_for(rtcp_stop.wait(), timeout=0.5)
            except TimeoutError:
                pass

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

        if RTT_LOG:

            @transport.event_handler("on_client_connected")
            async def on_client_connected(transport, connection):
                nonlocal rtcp_poll_task
                rtcp_poll_task = asyncio.ensure_future(poll_bot_rtcp_rtt(connection.pc))

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        rtcp_stop.set()
        if rtcp_poll_task:
            await rtcp_poll_task
        stats_path = os.getenv("BENCH_BOT_STATS")
        if stats_path:
            extra = None
            if RTT_LOG:
                # Bot's own receive-side jitter buffer — the first (and
                # only, for this echo pipeline) audio buffer created in
                # creation_order; drop fill-up samples like the floor
                # breakdown script does.
                jb_hold: list[float] = []
                if _jb_creation_order:
                    vals = (_jb_delays or {}).get(_jb_creation_order[0], [])
                    jb_hold = vals[5:] if len(vals) > 10 else vals
                extra = {
                    "rtcp_rtt_ms": summarize(rtcp_rtt_samples),
                    "jitter_buffer_hold_ms": summarize(jb_hold),
                    "media_sender_buffer_ms": summarize(_media_sender_samples or []),
                    "raw_audio_track_depth_10ms_chunks": summarize(
                        [depth for _, depth, _ in _raw_audio_track_events]
                    ),
                    # Raw (t_mono, depth, "add"|"recv") events — lets an
                    # external analysis correlate queue depth against wall
                    # clock over the run, not just an aggregate distribution.
                    "raw_audio_track_events": _raw_audio_track_events,
                    "opus_encode_ms": summarize(_opus_encode_ms or []),
                    "opus_decode_ms": summarize(_opus_decode_ms or []),
                    # Position-keyed (cum_samples, t_mono) checkpoints — same
                    # cum_samples numbering as LatencyObserver's out_raw, so
                    # a marker's exact wait at each stage of the send path
                    # can be measured directly rather than inferred.
                    "media_sender_flush_events": _media_sender_flush_events,
                    "raw_audio_track_sent_events": _raw_audio_track_sent_events,
                    "moq_send_events": _moq_send_events,
                }
            observer.dump(stats_path, extra=extra)
        if transport_name == "MOQTransport":
            await transport.disconnect()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

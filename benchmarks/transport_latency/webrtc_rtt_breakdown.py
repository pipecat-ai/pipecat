"""Decompose the aiortc webrtc-local floor into its RTT components.

Runs the same in-process ``LoopbackConnector`` (two RTCPeerConnections
echoing a track, a -> b -> a, zero real network) used for the ``floor-webrtc``
trial, and instruments three layers simultaneously so they can be compared
against the same chirp-based end-to-end RTT the benchmark harness reports:

1. RTCP RTT (``remote-inbound-rtp.roundTripTime`` from ``pc.getStats()``) —
   pure transport round-trip, computed from RTCP SR/RR LSR/DLSR exchange.
   Polled periodically since aiortc's stats report only holds the latest
   value and RTCP fires every 0.5-1.5s.
2. Jitter-buffer hold time — aiortc's stats API does not expose this
   (confirmed: no ``jitterBufferDelay`` field in ``aiortc.stats``), so it's
   measured directly by wrapping ``JitterBuffer.add`` and timing the gap
   between a packet's first arrival at a given RTP timestamp and the moment
   ``add()`` releases the frame carrying that timestamp. There are two
   independent jitter buffers on the loopback path (bot's receiver on hop
   a->b, client's receiver on hop b->a); reported separately by creation
   order.
3. Chirp end-to-end RTT — the harness's own measurement
   (``client_core.run_trial``), included for direct comparison.

No ICE candidate-pair RTT: aiortc's stats module has no
``RTCIceCandidatePairStats`` at all (confirmed by reading ``aiortc/stats.py``)
— it only exposes ``RTCInboundRtpStreamStats``, ``RTCRemoteInboundRtpStreamStats``,
``RTCOutboundRtpStreamStats``, ``RTCRemoteOutboundRtpStreamStats``, and
``RTCTransportStats``. There is no browser-equivalent ICE-layer number to pull.

Use ``--prefetch`` to reproduce the investigation that found aiortc's
hardcoded audio ``prefetch=4`` (``aiortc/rtcrtpreceiver.py``) contributes a
fixed ~80ms/hop hold, independent of real network jitter — e.g.
``--prefetch 1`` for the architectural floor (one packetization interval,
~20ms/hop) that ``configure_audio_jitter_prefetch`` applies by default
elsewhere in this benchmark.

Usage:
    uv run python benchmarks/transport_latency/webrtc_rtt_breakdown.py --prefetch 1
"""

import argparse
import asyncio
import time
from collections import defaultdict

import numpy as np
from aiortc.jitterbuffer import JitterBuffer
from client_core import run_trial
from webrtc_client import LoopbackConnector, configure_audio_jitter_prefetch


def instrument_jitter_buffer_timing():
    """Wrap JitterBuffer.add to time packet-arrival -> frame-release.

    Keyed by id(self) so the two independent buffers on the loopback path
    (hop a->b and hop b->a) are reported separately, in creation order.
    Prefetch itself is set via ``configure_audio_jitter_prefetch`` (the same
    knob the rest of the benchmark uses) — this only adds timing.
    """
    orig_add = JitterBuffer.add
    arrival_times: dict[int, dict[int, float]] = defaultdict(dict)
    delays_by_buffer: dict[int, list[float]] = defaultdict(list)
    creation_order: list[int] = []

    def patched_add(self, packet):
        buf_id = id(self)
        if buf_id not in creation_order:
            creation_order.append(buf_id)
        now = time.monotonic()
        arrivals = arrival_times[buf_id]
        if packet.timestamp not in arrivals:
            arrivals[packet.timestamp] = now
        pli_flag, frame = orig_add(self, packet)
        if frame is not None:
            t0 = arrivals.pop(frame.timestamp, None)
            if t0 is not None:
                delays_by_buffer[buf_id].append((time.monotonic() - t0) * 1000.0)
        return pli_flag, frame

    JitterBuffer.add = patched_add

    def restore():
        JitterBuffer.add = orig_add

    return delays_by_buffer, creation_order, restore


async def poll_rtcp_rtt(
    connector: LoopbackConnector, samples: dict[str, list[float]], stop: asyncio.Event
):
    """Poll getStats() periodically; RTCP RR fires every 0.5-1.5s so the
    stats report's single 'latest value' slot would otherwise only give us
    the last sample — poll to build a distribution.
    """
    pcs = {"a_to_b": connector._pcs[0], "b_to_a": connector._pcs[1]}
    while not stop.is_set():
        for label, pc in pcs.items():
            try:
                report = await pc.getStats()
            except Exception:
                continue
            for stat in report.values():
                if stat.type == "remote-inbound-rtp" and stat.roundTripTime is not None:
                    samples[label].append(stat.roundTripTime * 1000.0)
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.4)
        except TimeoutError:
            pass


def summarize(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: no samples")
        return
    arr = np.array(values)
    print(
        f"  {label}: n={len(arr)} p50={np.percentile(arr, 50):.2f}ms "
        f"p95={np.percentile(arr, 95):.2f}ms mean={arr.mean():.2f}ms "
        f"min={arr.min():.2f}ms max={arr.max():.2f}ms"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="aiortc audio JitterBuffer prefetch override (default 1, this "
        "benchmark's tuned floor; aiortc's own stock default is 4).",
    )
    args = parser.parse_args()
    duration_s = args.duration
    warmup_s = args.warmup

    configure_audio_jitter_prefetch(args.prefetch)
    delays_by_buffer, creation_order, restore_timing = instrument_jitter_buffer_timing()

    connector = LoopbackConnector()
    rtcp_samples: dict[str, list[float]] = defaultdict(list)
    stop = asyncio.Event()

    # run_trial() calls connector.start() internally, but we need the RTCP
    # poller running against connector._pcs once they exist. LoopbackConnector
    # sets self._pcs during start(); start the poller as a background task
    # that begins polling once _pcs is available.
    async def poller_when_ready():
        while not hasattr(connector, "_pcs"):
            await asyncio.sleep(0.05)
        await poll_rtcp_rtt(connector, rtcp_samples, stop)

    poller_task = asyncio.create_task(poller_when_ready())

    try:
        result = await run_trial(connector, duration_s=duration_s, warmup_s=warmup_s)
    finally:
        stop.set()
        await poller_task
        restore_timing()

    chirp_rtts = np.array(result.rtts_ms)

    print(
        f"=== webrtc loopback floor RTT breakdown "
        f"(duration={duration_s}s, warmup={warmup_s}s, prefetch={args.prefetch}) ===\n"
    )

    print("[1] Chirp end-to-end RTT (client_core.run_trial, the harness's own metric)")
    print("    -> what the benchmark reports as 'floor-webrtc'")
    if len(chirp_rtts):
        summarize("chirp RTT", list(chirp_rtts))
    else:
        print("  no chirp measurements (unexpected)")
    print(f"  drops={result.drops} ambiguous={result.ambiguous}\n")

    print("[2] RTCP RTT (remote-inbound-rtp.roundTripTime, polled every ~0.4s)")
    print("    -> pure transport round-trip; expect ~0 on in-process loopback")
    for label in ("a_to_b", "b_to_a"):
        summarize(label, rtcp_samples.get(label, []))
    print()

    print("[3] Jitter-buffer hold time (packet arrival -> frame release, instrumented)")
    print("    -> aiortc exposes no jitterBufferDelay stat; measured directly")
    hop_labels = ["hop1 (bot receiving probe, a->b)", "hop2 (client receiving echo, b->a)"]
    for i, buf_id in enumerate(creation_order):
        label = hop_labels[i] if i < len(hop_labels) else f"buffer {i}"
        vals = delays_by_buffer[buf_id]
        # Drop the first few (buffer fill-up) samples so this reflects
        # steady-state hold time, not the one-time prefetch ramp.
        steady = vals[5:] if len(vals) > 10 else vals
        summarize(label, steady)
    print()

    total_jb = sum(
        np.percentile(delays_by_buffer[buf_id][5:], 50) if len(delays_by_buffer[buf_id]) > 10 else 0
        for buf_id in creation_order
    )
    chirp_p50 = np.percentile(chirp_rtts, 50) if len(chirp_rtts) else float("nan")
    print("[4] Reconciliation")
    print(f"    chirp RTT p50:                  {chirp_p50:.2f} ms")
    print(f"    sum of both jitter-buffer p50s:  {total_jb:.2f} ms")
    print(f"    unaccounted (encode/decode/resample/scheduling): {chirp_p50 - total_jb:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())

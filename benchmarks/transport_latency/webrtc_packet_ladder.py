"""Per-packet ladder through aiortc's audio receive path — where does the
fixed ~20ms (prefetch=1) hold actually come from, packet by packet?

Sends a short burst of N 20ms silent chunks through the same
``LoopbackConnector`` a->b hop used elsewhere in this benchmark (the "bot's
receiver" direction — see webrtc_rtt_breakdown.py), and prints one row per
packet: RTP timestamp, send time, arrival time (the moment
``JitterBuffer.add`` first sees a packet carrying that timestamp), release
time (the moment ``add`` hands back the frame), and both deltas.

With prefetch=1, aiortc's ``_remove_frame`` only releases frame i once it has
counted 1 complete frame ahead of it — i.e. once packet i+1 arrives. The
ladder makes that mechanism visible directly: t_release[i] tracks
t_arrival[i+1], not t_arrival[i] + a flat 20ms.

Usage:
    uv run python benchmarks/transport_latency/webrtc_packet_ladder.py --packets 10 --prefetch 1
"""

import argparse
import asyncio
import time

from aiortc.jitterbuffer import JitterBuffer
from webrtc_client import CHUNK_SAMPLES, LoopbackConnector, configure_audio_jitter_prefetch


def instrument_packet_ladder():
    """Like ``webrtc_client.instrument_jitter_buffer_timing``, but keeps raw
    per-packet arrival/release wall-clock times (not just the hold duration),
    restricted to the first JitterBuffer instance created — hop a->b, the
    bot's receiver in the LoopbackConnector topology.
    """
    orig_add = JitterBuffer.add
    creation_order: list[int] = []
    arrival_times: dict[int, dict[int, float]] = {}
    rows: list[dict] = []

    def patched_add(self, packet):
        buf_id = id(self)
        if buf_id not in creation_order:
            creation_order.append(buf_id)
            arrival_times[buf_id] = {}
        if buf_id != creation_order[0]:
            return orig_add(self, packet)  # hop b->a (echo) — not our target
        now = time.monotonic()
        arrivals = arrival_times[buf_id]
        if packet.timestamp not in arrivals:
            arrivals[packet.timestamp] = now
        pli_flag, frame = orig_add(self, packet)
        if frame is not None:
            t_arrival = arrivals.pop(frame.timestamp, None)
            if t_arrival is not None:
                rows.append(
                    {
                        "rtp_ts": frame.timestamp,
                        "t_arrival": t_arrival,
                        "t_release": time.monotonic(),
                    }
                )
        return pli_flag, frame

    JitterBuffer.add = patched_add

    def restore():
        JitterBuffer.add = orig_add

    return rows, restore


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packets", type=int, default=10, help="number of 20ms chunks to send (10 = 200ms)"
    )
    parser.add_argument("--prefetch", type=int, default=1)
    args = parser.parse_args()

    configure_audio_jitter_prefetch(args.prefetch)
    rows, restore = instrument_packet_ladder()

    connector = LoopbackConnector()
    await connector.start()

    async def drain_echo():
        # Hop b->a (the echo back) has to be consumed or aiortc's send queue
        # backs up; we only care about hop a->b's arrival/release above.
        try:
            async for _ in connector.recv_chunks():
                pass
        except asyncio.CancelledError:
            pass

    drainer = asyncio.ensure_future(drain_echo())

    silence = b"\x00\x00" * CHUNK_SAMPLES  # 20ms of silence, matches real chunk size
    send_times: list[float] = []
    for _ in range(args.packets):
        send_times.append(time.monotonic())
        await connector.send_chunk(silence)
        await asyncio.sleep(0.02)  # pace like real 20ms audio frames

    await asyncio.sleep(0.5)  # let the last packet's prefetch backlog drain
    drainer.cancel()
    await asyncio.gather(drainer, return_exceptions=True)
    restore()
    await connector.stop()

    print(
        f"=== packet ladder: {args.packets} packets ({args.packets * 20}ms), "
        f"prefetch={args.prefetch} ===\n"
    )
    header = (
        f"{'#':>3} {'rtp_ts':>8} {'t_send':>9} {'t_arrival':>10} {'t_release':>10} "
        f"{'send->arrival':>14} {'arrival->release (hold)':>24}"
    )
    print(header)
    print("-" * len(header))
    t0 = send_times[0] if send_times else 0.0
    # Packets arrive/release in FIFO order on this zero-jitter loopback, so
    # row i corresponds to send_times[i] by position, not by matching RTP
    # timestamp (aiortc may start the RTP clock at a random offset).
    for i, row in enumerate(rows):
        t_send = send_times[i] if i < len(send_times) else None
        send_arrival_ms = (
            (row["t_arrival"] - t_send) * 1000.0 if t_send is not None else float("nan")
        )
        hold_ms = (row["t_release"] - row["t_arrival"]) * 1000.0
        print(
            f"{i:>3} {row['rtp_ts']:>8} "
            f"{((t_send - t0) * 1000.0 if t_send is not None else float('nan')):>9.2f} "
            f"{(row['t_arrival'] - t0) * 1000.0:>10.2f} "
            f"{(row['t_release'] - t0) * 1000.0:>10.2f} "
            f"{send_arrival_ms:>14.2f} "
            f"{hold_ms:>24.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())

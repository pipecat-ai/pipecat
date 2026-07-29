"""Transport-agnostic round-trip measurement loop.

One process, one ``time.monotonic()`` clock stamps both directions — no clock
sync. The connector is a thin adapter; everything timing-related lives here so
both transports are measured identically.

Timing convention (identical for every connector, so quantization is fair):
- t_send(k)  = handoff time of the 20 ms chunk containing marker k's onset.
  The transport consumes the whole chunk at once — the marker's bytes leave
  in that chunk's packet(s) regardless of where the onset sits inside it.
- t_recv(k)  = arrival stamp of the received chunk containing the onset (the
  onset is not observable before its chunk arrives).
Both sides are chunk-granular, so per-marker RTT carries up to ±one chunk of
quantization, identically for both transports; the offline WAV
cross-correlation check bounds the residual.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from probe import SAMPLE_RATE, detect_onsets, gen_probe


class Connector(Protocol):
    """Adapter interface a transport connector must implement."""

    async def start(self) -> None: ...

    async def send_chunk(self, pcm: bytes) -> None:
        """Hand one chunk of s16 mono PCM to the transport."""
        ...

    def recv_chunks(self) -> AsyncIterator[bytes]:
        """Yield received s16 mono PCM chunks as they arrive."""
        ...

    async def stop(self) -> None: ...


@dataclass
class TrialResult:
    rtts_ms: list[float]
    drops: int
    ambiguous: int
    markers_sent: int
    sent_wav: np.ndarray = field(repr=False, default=None)
    recv_wav: np.ndarray = field(repr=False, default=None)
    recv_chunk_stats: dict = field(default_factory=dict)


async def run_trial(
    connector: Connector,
    duration_s: float = 60.0,
    warmup_s: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
    chunk_ms: int = 20,
    period_ms: int = 250,
    tail_s: float = 1.0,
) -> TrialResult:
    signal, marker_positions = gen_probe(duration_s, sample_rate, period_ms=period_ms)
    chunk_samples = sample_rate * chunk_ms // 1000
    chunk_s = chunk_ms / 1000.0

    send_handoffs: list[float] = []  # t_handoff per chunk index
    recv_chunks: list[np.ndarray] = []
    recv_stamps: list[float] = []  # arrival stamp per received chunk

    await connector.start()

    async def receiver() -> None:
        async for pcm in connector.recv_chunks():
            recv_stamps.append(time.monotonic())
            recv_chunks.append(np.frombuffer(pcm, dtype=np.int16))

    recv_task = asyncio.create_task(receiver())

    # Paced sender: absolute schedule, no drift accumulation.
    n_chunks = len(signal) // chunk_samples
    t0 = time.monotonic()
    for i in range(n_chunks):
        target = t0 + i * chunk_s
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        send_handoffs.append(time.monotonic())
        chunk = signal[i * chunk_samples : (i + 1) * chunk_samples]
        await connector.send_chunk(chunk.tobytes())

    await asyncio.sleep(tail_s)
    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass
    await connector.stop()

    # --- Post-hoc analysis ------------------------------------------------
    if not recv_chunks:
        return TrialResult([], len(marker_positions), 0, len(marker_positions))

    received = np.concatenate(recv_chunks)
    onsets = detect_onsets(received, sample_rate, period_ms=period_ms)

    # Map each onset sample position to the arrival stamp of its chunk.
    chunk_bounds = np.cumsum([len(c) for c in recv_chunks])  # end positions
    onset_recv_times = [
        recv_stamps[int(np.searchsorted(chunk_bounds, pos, side="right"))] for pos in onsets
    ]

    # Marker send times: handoff of the chunk containing the onset.
    marker_send_times = [
        send_handoffs[int(pos) // chunk_samples]
        for pos in marker_positions
        if int(pos) // chunk_samples < len(send_handoffs)
    ]

    # Pair each received onset with the unique sent marker in the window
    # (~0, period). Ambiguity or no candidate => not measured.
    window_s = (period_ms - 10) / 1000.0
    window_lo = -0.001
    warmup_end = send_handoffs[0] + warmup_s
    rtts_ms: list[float] = []
    ambiguous = 0
    used: set[int] = set()
    for t_r in onset_recv_times:
        cands = [
            k
            for k, t_s in enumerate(marker_send_times)
            if k not in used and window_lo < t_r - t_s < window_s
        ]
        if len(cands) == 1:
            used.add(cands[0])
            if marker_send_times[cands[0]] >= warmup_end:
                rtts_ms.append((t_r - marker_send_times[cands[0]]) * 1000.0)
        elif len(cands) > 1:
            ambiguous += 1

    markers_after_warmup = sum(1 for t in marker_send_times if t >= warmup_end)
    drops = markers_after_warmup - len(rtts_ms) - ambiguous

    return TrialResult(
        rtts_ms=rtts_ms,
        drops=max(0, drops),
        ambiguous=ambiguous,
        markers_sent=markers_after_warmup,
        sent_wav=signal,
        recv_wav=received,
        recv_chunk_stats={
            "n_chunks": len(recv_chunks),
            "median_chunk_ms": float(np.median([len(c) for c in recv_chunks]))
            / sample_rate
            * 1000.0,
        },
    )


class NullConnector:
    """In-process loopback: send queue == receive queue. Measures the harness floor."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def send_chunk(self, pcm: bytes) -> None:
        self._q.put_nowait(pcm)

    async def recv_chunks(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._q.get()

    async def stop(self) -> None:
        pass


async def _floor_test() -> None:
    result = await run_trial(NullConnector(), duration_s=15.0, warmup_s=2.0)
    arr = np.array(result.rtts_ms)
    print(
        f"null floor: n={len(arr)} drops={result.drops} ambiguous={result.ambiguous} "
        f"p50={np.percentile(arr, 50):.3f} ms p95={np.percentile(arr, 95):.3f} ms "
        f"max={arr.max():.3f} ms"
    )
    assert result.drops == 0 and result.ambiguous == 0
    assert np.percentile(arr, 95) < 25.0, "harness floor above one chunk duration"
    print("client_core floor test OK")


if __name__ == "__main__":
    asyncio.run(_floor_test())

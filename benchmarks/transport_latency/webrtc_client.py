"""aiortc offerer connector for the transport latency benchmark.

Dials the echo bot's SmallWebRTC signalling endpoint (``POST /api/offer``) as
the browser normally would, sends the probe as a real Opus-encoded WebRTC
track, and yields received (decoded) audio chunks with arrival stamps taken by
the shared client core.

Topology enforcement (fairness control #10):
- no ``turn`` server: only host candidates exist; the selected pair is
  asserted host/host on loopback.
- with a ``turn`` server (local coturn or a hosted service): it is the only
  ICE server and non-relay candidates are stripped from the client's offer
  (aiortc has no ``iceTransportPolicy``), so media must traverse the TURN
  allocation both directions; the selected local candidate is asserted
  type=relay.
"""

import asyncio
import fractions
from collections.abc import AsyncIterator

import aiohttp
import av
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamTrack

SAMPLE_RATE = 48000
CHUNK_SAMPLES = SAMPLE_RATE * 20 // 1000  # 20 ms


class _ProbeTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._q: asyncio.Queue[bytes] = asyncio.Queue()
        self._pts = 0

    def push(self, pcm: bytes) -> None:
        self._q.put_nowait(pcm)

    async def recv(self) -> av.AudioFrame:
        pcm = await self._q.get()
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.planes[0].update(pcm)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += len(pcm) // 2
        return frame


def _strip_non_relay_candidates(sdp: str) -> str:
    lines = [
        ln
        for ln in sdp.splitlines()
        if not (ln.startswith("a=candidate") and " typ relay " not in ln + " ")
    ]
    return "\r\n".join(lines) + "\r\n"


def _selected_pair(pc: RTCPeerConnection) -> tuple[str, str] | None:
    """(local_type, remote_type) of the nominated ICE pair, or None.

    aiortc exposes no public selected-pair API; reach into aioice's nominated
    map (stable across aiortc 1.x).
    """
    for transceiver in pc.getTransceivers():
        transport = transceiver.sender.transport
        ice = getattr(transport, "transport", transport)  # DTLS -> ICE
        conn = getattr(ice, "_connection", None)
        nominated = getattr(conn, "_nominated", None)
        if nominated:
            pair = next(iter(nominated.values()))
            return (pair.local_candidate.type, pair.remote_candidate.type)
    return None


class WebRTCConnector:
    def __init__(
        self,
        offer_url: str = "http://localhost:7860/api/offer",
        turn: RTCIceServer | None = None,
    ) -> None:
        self._offer_url = offer_url
        self._turn = turn  # None: direct (host/host); set: TURN-forced relay
        self._pc: RTCPeerConnection | None = None
        self._probe = _ProbeTrack()
        self._remote_track: asyncio.Future[MediaStreamTrack] = asyncio.Future()

    async def start(self) -> None:
        config = RTCConfiguration(iceServers=[self._turn] if self._turn else [])
        pc = RTCPeerConnection(config)
        self._pc = pc
        pc.addTrack(self._probe)

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio" and not self._remote_track.done():
                self._remote_track.set_result(track)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)  # waits for ICE gathering
        sdp = pc.localDescription.sdp
        if self._turn:
            # Stripping the offer SDP only hides candidates from the bot; the
            # client would still pair its own local host candidates with the
            # bot's. Remove non-relay local candidates inside aioice so the
            # only formable pair is client-relay <-> bot-host, i.e. all media
            # traverses the TURN allocation.
            sdp = _strip_non_relay_candidates(sdp)
            filtered = False
            for transceiver in pc.getTransceivers():
                transport = transceiver.sender.transport
                ice = getattr(transport, "transport", transport)
                conn = getattr(ice, "_connection", None)
                # aioice builds candidate pairs from its per-candidate
                # _protocols list (and local_candidates returns a copy), so
                # both structures must be pruned to relay-only.
                cands = getattr(conn, "_local_candidates", None)
                protos = getattr(conn, "_protocols", None)
                if cands:
                    relay_only = [c for c in cands if c.type == "relay"]
                    if not relay_only:
                        raise RuntimeError(
                            f"no relay candidates gathered — is the TURN server up? "
                            f"({self._turn.urls})"
                        )
                    cands[:] = relay_only
                    if protos is not None:
                        protos[:] = [
                            p
                            for p in protos
                            if getattr(p, "local_candidate", None) is not None
                            and p.local_candidate.type == "relay"
                        ]
                    filtered = True
            if not filtered:
                raise RuntimeError("could not reach aioice local candidate list")

        async with aiohttp.ClientSession() as http:
            async with http.post(
                self._offer_url, json={"sdp": sdp, "type": pc.localDescription.type}
            ) as resp:
                resp.raise_for_status()
                answer = await resp.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        # Wait for media before measuring; then enforce topology.
        await asyncio.wait_for(self._remote_track, timeout=15.0)
        for _ in range(50):
            pair = _selected_pair(self._pc)
            if pair:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("no nominated ICE pair found")
        local_type, remote_type = pair
        if self._turn is None and (local_type, remote_type) != ("host", "host"):
            raise RuntimeError(f"direct scenario got ICE pair {pair}, expected host/host")
        if self._turn is not None and local_type != "relay":
            raise RuntimeError(f"TURN scenario got ICE pair {pair}, expected local relay")
        self.selected_ice_pair = pair

    async def send_chunk(self, pcm: bytes) -> None:
        self._probe.push(pcm)

    async def recv_chunks(self) -> AsyncIterator[bytes]:
        track = await self._remote_track
        while True:
            frame = await track.recv()
            arr = frame.to_ndarray().reshape(-1)
            if arr.dtype.kind == "f":
                arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
            channels = len(frame.layout.channels)
            if channels > 1:
                arr = arr.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
            if frame.sample_rate != SAMPLE_RATE:
                raise RuntimeError(f"unexpected sample rate {frame.sample_rate}")
            yield arr.astype(np.int16).tobytes()

    async def stop(self) -> None:
        if self._pc:
            await self._pc.close()


class LoopbackConnector:
    """Two in-process RTCPeerConnections echoing a track back.

    Measures the aiortc stack floor (two encode/decode cycles + two receive
    jitter buffers — the same count as the real scenario) with no bot and no
    real network, so scenario RTTs can be decomposed into client-stack vs
    bot-path time.
    """

    def __init__(self) -> None:
        self._probe = _ProbeTrack()
        self._remote: asyncio.Future[MediaStreamTrack] = asyncio.Future()
        self._back: MediaStreamTrack | None = None

    async def start(self) -> None:
        a, b = RTCPeerConnection(), RTCPeerConnection()
        self._pcs = (a, b)
        a.addTrack(self._probe)
        remote = self._remote

        @b.on("track")
        def on_track(t: MediaStreamTrack) -> None:
            if not remote.done():
                remote.set_result(t)

        class _EchoTrack(MediaStreamTrack):
            kind = "audio"

            async def recv(self) -> av.AudioFrame:
                t = await remote
                return await t.recv()

        b.addTrack(_EchoTrack())

        @a.on("track")
        def on_track_a(t: MediaStreamTrack) -> None:
            self._back = t

        offer = await a.createOffer()
        await a.setLocalDescription(offer)
        await b.setRemoteDescription(a.localDescription)
        answer = await b.createAnswer()
        await b.setLocalDescription(answer)
        await a.setRemoteDescription(b.localDescription)
        for _ in range(100):
            if self._back is not None:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("loopback track never arrived")

    async def send_chunk(self, pcm: bytes) -> None:
        self._probe.push(pcm)

    async def recv_chunks(self) -> AsyncIterator[bytes]:
        while True:
            frame = await self._back.recv()
            arr = frame.to_ndarray().reshape(-1)
            if arr.dtype.kind == "f":
                arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
            channels = len(frame.layout.channels)
            if channels > 1:
                arr = arr.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
            yield arr.astype(np.int16).tobytes()

    async def stop(self) -> None:
        for pc in self._pcs:
            await pc.close()


async def _main() -> None:
    import argparse

    from client_core import run_trial

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turn-url", help="TURN URL; omit for direct host/host (local coturn: turn:127.0.0.1:3478)"
    )
    parser.add_argument("--turn-username", default="pipecat")
    parser.add_argument("--turn-credential", default="pipecat")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    turn = None
    if args.turn_url:
        turn = RTCIceServer(
            urls=args.turn_url, username=args.turn_username, credential=args.turn_credential
        )
    connector = WebRTCConnector(turn=turn)
    result = await run_trial(connector, duration_s=args.duration, warmup_s=5.0)
    arr = np.array(result.rtts_ms)
    print(f"ice pair: {connector.selected_ice_pair}")
    print(
        f"webrtc/{'turn' if turn else 'direct'}: n={len(arr)} drops={result.drops} "
        f"ambiguous={result.ambiguous} p50={np.percentile(arr, 50):.2f} ms "
        f"p95={np.percentile(arr, 95):.2f} ms max={arr.max():.2f} ms"
    )


if __name__ == "__main__":
    asyncio.run(_main())

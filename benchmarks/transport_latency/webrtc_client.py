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
import time
from collections import defaultdict
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
from aiortc.jitterbuffer import JitterBuffer
from aiortc.mediastreams import MediaStreamTrack

SAMPLE_RATE = 48000
CHUNK_SAMPLES = SAMPLE_RATE * 20 // 1000  # 20 ms

# aiortc hardcodes prefetch=4 for its audio JitterBuffer
# (aiortc/rtcrtpreceiver.py) with no public API to change it: the receiver
# won't release a frame until it has counted 4 complete frames ahead of it,
# a fixed ~80ms hold per receive hop regardless of actual network jitter
# (confirmed by instrumenting JitterBuffer.add — see RUNBOOK "floor"
# investigation). prefetch=1 is the architectural minimum (a frame can only
# be confirmed complete once the next RTP timestamp arrives, ~20ms) and
# removes that fixed tax, at the cost of removing aiortc's jitter
# tolerance — on scenarios with real network jitter (webrtc-turn-*) this
# will surface as reordering/drops a stock aiortc client would have
# absorbed. Default here; override via --webrtc-prefetch.
DEFAULT_AUDIO_JITTER_PREFETCH = 1

_ORIGINAL_JITTER_BUFFER_INIT = JitterBuffer.__init__


def configure_audio_jitter_prefetch(prefetch: int) -> None:
    """Monkeypatch aiortc's hardcoded audio JitterBuffer prefetch.

    Always re-derives from the pristine constructor, so repeated calls (e.g.
    once per scenario) don't stack patches. Video's buffer (prefetch=0,
    is_video=True) is untouched.
    """

    forced_prefetch = prefetch

    def patched_init(self, capacity, prefetch=0, is_video=False):
        if not is_video:
            prefetch = forced_prefetch
        _ORIGINAL_JITTER_BUFFER_INIT(self, capacity, prefetch=prefetch, is_video=is_video)

    JitterBuffer.__init__ = patched_init


_ORIGINAL_JITTER_BUFFER_ADD = JitterBuffer.add


def instrument_jitter_buffer_timing():
    """Wrap ``JitterBuffer.add`` to time packet-arrival -> frame-release.

    aiortc exposes no ``jitterBufferDelay`` stat, so this measures the hold
    directly: the gap between a packet's first arrival at a given RTP
    timestamp and the moment ``add()`` releases the frame carrying that
    timestamp. Idempotent like ``configure_audio_jitter_prefetch`` — always
    re-wraps the pristine ``add``, so repeated calls don't stack patches.

    Returns ``(delays_by_buffer, creation_order, restore)``:
    ``delays_by_buffer[id(buffer)]`` is the list of hold times (ms) in
    release order; ``creation_order`` lists buffer ids in first-seen order
    (one per receive direction — e.g. a single entry for a connector with one
    audio receiver).
    """
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
        pli_flag, frame = _ORIGINAL_JITTER_BUFFER_ADD(self, packet)
        if frame is not None:
            t0 = arrivals.pop(frame.timestamp, None)
            if t0 is not None:
                delays_by_buffer[buf_id].append((time.monotonic() - t0) * 1000.0)
        return pli_flag, frame

    JitterBuffer.add = patched_add

    def restore():
        JitterBuffer.add = _ORIGINAL_JITTER_BUFFER_ADD

    return delays_by_buffer, creation_order, restore


def instrument_opus_codec_timing():
    """Wrap aiortc's ``OpusEncoder.encode``/``OpusDecoder.decode`` to time
    each call's wall-clock cost.

    Encode runs in ``RTCRtpSender``'s thread-pool executor (before a packet
    is sent); decode runs in ``RTCRtpReceiver``'s dedicated decoder thread,
    *after* the jitter buffer releases a frame — so decode time is additive
    to the measured jitter-buffer hold, not overlapping with it. Idempotent
    like the other instrument_* helpers here.

    Returns ``(encode_ms_samples, decode_ms_samples, restore)``.
    """
    from aiortc.codecs.opus import OpusDecoder, OpusEncoder

    orig_encode = OpusEncoder.encode
    orig_decode = OpusDecoder.decode
    encode_ms: list[float] = []
    decode_ms: list[float] = []

    def patched_encode(self, frame, force_keyframe=False):
        t0 = time.perf_counter()
        result = orig_encode(self, frame, force_keyframe)
        encode_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    def patched_decode(self, encoded_frame):
        t0 = time.perf_counter()
        result = orig_decode(self, encoded_frame)
        decode_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    OpusEncoder.encode = patched_encode
    OpusDecoder.decode = patched_decode

    def restore():
        OpusEncoder.encode = orig_encode
        OpusDecoder.decode = orig_decode

    return encode_ms, decode_ms, restore


def instrument_decoder_handoff():
    """Time the hop opus-decode timing doesn't cover: aiortc decodes audio
    on a dedicated background thread (``decoder_worker`` in
    ``rtcrtpreceiver.py``) and hands each decoded frame back to the asyncio
    event loop via ``asyncio.run_coroutine_threadsafe(output_q.put(frame),
    loop)`` — a cross-thread scheduling handoff, separate from decode
    duration itself, before ``RemoteStreamTrack.recv()`` can return it.

    Reimplements ``decoder_worker``'s body (copied from aiortc's source,
    which this depends on structurally — a version bump could drift) so a
    timestamp can be recorded right at the scheduling call, matched against
    ``RemoteStreamTrack.recv()``'s return by the decoded frame's ``pts``.

    Returns ``(handoff_ms_samples, restore)``.
    """
    import aiortc.rtcrtpreceiver as rtcrtpreceiver_module
    from aiortc.codecs import get_decoder
    from aiortc.rtcrtpreceiver import RemoteStreamTrack

    orig_decoder_worker = rtcrtpreceiver_module.decoder_worker
    orig_track_recv = RemoteStreamTrack.recv
    scheduled_times: dict[int, float] = {}
    handoff_ms: list[float] = []

    def patched_decoder_worker(loop, input_q, output_q):
        codec_name = None
        decoder = None
        while True:
            task = input_q.get()
            if task is None:
                asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
                break
            codec, encoded_frame = task
            if codec.name != codec_name:
                decoder = get_decoder(codec)
                codec_name = codec.name
            for frame in decoder.decode(encoded_frame):
                scheduled_times[frame.pts] = time.monotonic()
                asyncio.run_coroutine_threadsafe(output_q.put(frame), loop)
        if decoder is not None:
            del decoder

    async def patched_track_recv(self):
        frame = await orig_track_recv(self)
        t0 = scheduled_times.pop(getattr(frame, "pts", None), None)
        if t0 is not None:
            handoff_ms.append((time.monotonic() - t0) * 1000.0)
        return frame

    rtcrtpreceiver_module.decoder_worker = patched_decoder_worker
    RemoteStreamTrack.recv = patched_track_recv

    def restore():
        rtcrtpreceiver_module.decoder_worker = orig_decoder_worker
        RemoteStreamTrack.recv = orig_track_recv

    return handoff_ms, restore


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
        # RTCP SR/RR round-trip time (remote-inbound-rtp.roundTripTime) —
        # transport-level, upstream of the jitter buffer. aiortc exposes no
        # ICE candidate-pair stats at all, so this is the only wire-level RTT
        # available; RTCP only fires every 0.5-1.5s so it's polled to build a
        # distribution rather than read once at teardown.
        self.rtp_rtt_ms_samples: list[float] = []
        self._rtcp_poll_task: asyncio.Task | None = None
        self._rtcp_stop = asyncio.Event()
        # Client-side jitter-buffer hold (receiving the bot's echo) — see
        # instrument_jitter_buffer_timing. Populated at stop(); this
        # connector only has one audio receiver, so creation_order has a
        # single entry.
        self.jitter_buffer_hold_ms_samples: list[float] = []
        self._jb_delays: dict[int, list[float]] | None = None
        self._jb_creation_order: list[int] | None = None
        self._jb_restore = None

    async def start(self) -> None:
        self._jb_delays, self._jb_creation_order, self._jb_restore = (
            instrument_jitter_buffer_timing()
        )
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

        self._rtcp_poll_task = asyncio.ensure_future(self._poll_rtcp_rtt())

    async def _poll_rtcp_rtt(self) -> None:
        while not self._rtcp_stop.is_set():
            try:
                report = await self._pc.getStats()
            except Exception:
                report = {}
            for stat in report.values():
                if stat.type == "remote-inbound-rtp" and stat.roundTripTime is not None:
                    self.rtp_rtt_ms_samples.append(stat.roundTripTime * 1000.0)
            try:
                await asyncio.wait_for(self._rtcp_stop.wait(), timeout=0.5)
            except TimeoutError:
                pass

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
        self._rtcp_stop.set()
        if self._rtcp_poll_task:
            await self._rtcp_poll_task
        if self._pc:
            await self._pc.close()
        if self._jb_creation_order:
            buf_id = self._jb_creation_order[0]
            vals = self._jb_delays.get(buf_id, [])
            # Drop the first few (buffer fill-up) samples — steady-state
            # hold time, not the one-time prefetch ramp.
            self.jitter_buffer_hold_ms_samples = vals[5:] if len(vals) > 10 else vals
        if self._jb_restore:
            self._jb_restore()


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
        # RTCP RTT from both hops (a->b, b->a), merged — see WebRTCConnector.
        self.rtp_rtt_ms_samples: list[float] = []
        self._rtcp_poll_task: asyncio.Task | None = None
        self._rtcp_stop = asyncio.Event()

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
                self._rtcp_poll_task = asyncio.ensure_future(self._poll_rtcp_rtt())
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("loopback track never arrived")

    async def _poll_rtcp_rtt(self) -> None:
        while not self._rtcp_stop.is_set():
            for pc in self._pcs:
                try:
                    report = await pc.getStats()
                except Exception:
                    continue
                for stat in report.values():
                    if stat.type == "remote-inbound-rtp" and stat.roundTripTime is not None:
                        self.rtp_rtt_ms_samples.append(stat.roundTripTime * 1000.0)
            try:
                await asyncio.wait_for(self._rtcp_stop.wait(), timeout=0.5)
            except TimeoutError:
                pass

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
        self._rtcp_stop.set()
        if self._rtcp_poll_task:
            await self._rtcp_poll_task
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
    parser.add_argument(
        "--prefetch",
        type=int,
        default=DEFAULT_AUDIO_JITTER_PREFETCH,
        help="aiortc audio JitterBuffer prefetch override (default 1; aiortc's own default is 4)",
    )
    args = parser.parse_args()

    configure_audio_jitter_prefetch(args.prefetch)

    turn = None
    if args.turn_url:
        turn = RTCIceServer(
            urls=args.turn_url, username=args.turn_username, credential=args.turn_credential
        )
    connector = WebRTCConnector(turn=turn)
    result = await run_trial(connector, duration_s=args.duration, warmup_s=5.0)
    arr = np.array(result.rtts_ms)
    rtp_rtt = np.array(connector.rtp_rtt_ms_samples)
    jb_hold = np.array(connector.jitter_buffer_hold_ms_samples)
    print(f"ice pair: {connector.selected_ice_pair}")
    if len(rtp_rtt):
        print(f"rtcp rtt: p50={np.percentile(rtp_rtt, 50):.2f} ms n={len(rtp_rtt)}")
    if len(jb_hold):
        print(
            f"jitter buffer hold (client receive): p50={np.percentile(jb_hold, 50):.2f} ms n={len(jb_hold)}"
        )
    print(
        f"webrtc/{'turn' if turn else 'direct'}: n={len(arr)} drops={result.drops} "
        f"ambiguous={result.ambiguous} p50={np.percentile(arr, 50):.2f} ms "
        f"p95={np.percentile(arr, 95):.2f} ms max={arr.max():.2f} ms"
    )


if __name__ == "__main__":
    asyncio.run(_main())

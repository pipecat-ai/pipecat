#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTP session: UDP send/receive with 20ms timing and G.711 codec.

Manages per-call RTP media over UDP. Receives G.711-encoded audio, decodes to
PCM16, and queues for pipeline consumption. Sends PCM16 audio from the pipeline
after encoding to G.711 with drift-corrected 20ms pacing.
"""

from __future__ import annotations

import asyncio
import logging
import random
import struct
import time
from typing import Optional, Tuple

import numpy as np

from pipecat.transports.sip.codecs import G711Codec

logger = logging.getLogger(__name__)

RTP_HEADER_SIZE = 12
FRAME_DURATION = 0.020  # 20ms
SAMPLES_PER_FRAME = 160  # @ 8kHz
PCM16_FRAME_SIZE = 320  # bytes (160 samples * 2 bytes)
SILENCE_PCM16 = b"\x00" * PCM16_FRAME_SIZE
PREBUFFER_FRAMES = 3  # ~60ms

# RFC 2833 DTMF
DTMF_PAYLOAD_TYPE = 101
DTMF_DIGITS = "0123456789*#ABCD"


def pack_rtp_header(*, seq: int, timestamp: int, ssrc: int, payload_type: int = 0) -> bytes:
    """Pack a 12-byte RTP header (V=2, no padding/ext/CSRC).

    Args:
        seq: Sequence number (16-bit, wraps).
        timestamp: RTP timestamp (32-bit, wraps).
        ssrc: Synchronization source identifier.
        payload_type: RTP payload type (0=PCMU, 8=PCMA, 101=DTMF).

    Returns:
        12-byte RTP header.
    """
    return struct.pack(
        "!BBHII",
        0x80,
        payload_type & 0x7F,
        seq & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc,
    )


def unpack_rtp_header(data: bytes) -> Tuple[int, int, int, int]:
    """Unpack RTP header fields.

    Args:
        data: At least 12 bytes of RTP packet.

    Returns:
        Tuple of (payload_type, sequence, timestamp, ssrc).
    """
    _, byte1, seq, timestamp, ssrc = struct.unpack("!BBHII", data[:12])
    payload_type = byte1 & 0x7F
    return payload_type, seq, timestamp, ssrc


def parse_dtmf_event(payload: bytes) -> Tuple[str, bool, int] | None:
    """Parse an RFC 2833 DTMF event payload.

    Args:
        payload: 4-byte DTMF event payload after RTP header.

    Returns:
        Tuple of (digit, end_bit, duration) or None if invalid.
    """
    if len(payload) < 4:
        return None
    event = payload[0]
    end_bit = bool(payload[1] & 0x80)
    duration = struct.unpack("!H", payload[2:4])[0]
    if event < len(DTMF_DIGITS):
        digit = DTMF_DIGITS[event]
        return digit, end_bit, duration
    return None


class RTPSession:
    """Manages RTP send/receive over UDP with precise 20ms timing.

    Args:
        local_port: UDP port to bind for RTP media.
        prebuffer_frames: Number of frames to buffer before TX playback.
        dtmf_enabled: Whether to detect RFC 2833 DTMF events.
        rx_maxsize: Maximum size of the receive queue.
        tx_maxsize: Maximum size of the transmit queue.
    """

    def __init__(
        self,
        local_port: int,
        *,
        prebuffer_frames: int = PREBUFFER_FRAMES,
        dtmf_enabled: bool = True,
        rx_maxsize: int = 50,
        tx_maxsize: int = 250,
    ):
        """Initialize the RTP session.

        Args:
            local_port: UDP port to bind for RTP media.
            prebuffer_frames: Number of frames to buffer before TX playback.
            dtmf_enabled: Whether to detect RFC 2833 DTMF events.
            rx_maxsize: Maximum size of the receive queue.
            tx_maxsize: Maximum size of the transmit queue.
        """
        self.local_port = local_port
        self.rx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=rx_maxsize)
        self.tx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=tx_maxsize)
        self.dtmf_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50)

        self._prebuffer_frames = prebuffer_frames
        self._dtmf_enabled = dtmf_enabled
        self._ssrc = random.randint(0, 0xFFFFFFFF)
        self._seq = random.randint(0, 0xFFFF)
        self._timestamp = random.randint(0, 0xFFFFFFFF)
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._remote_addr: Optional[Tuple[str, int]] = None
        self._running = False
        self._codec = G711Codec.get_instance()
        self._sent = 0
        self._received = 0
        self._last_dtmf_ts = -1
        self._last_rtp_time: float = 0.0

    async def start(self, remote_addr: Tuple[str, int]):
        """Bind UDP socket and prepare for send/receive.

        Args:
            remote_addr: Remote RTP endpoint (ip, port).
        """
        self._remote_addr = remote_addr
        self._running = True
        self._last_rtp_time = time.monotonic()

        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _RTPProtocol(self),
            local_addr=("0.0.0.0", self.local_port),
        )
        actual_addr = self._transport.get_extra_info("sockname")
        if actual_addr:
            self.local_port = actual_addr[1]

        logger.info("RTP session started on port %d -> %s", self.local_port, remote_addr)

    async def run(self):
        """Run the send loop. Call after start()."""
        try:
            await self._send_loop()
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Stop the session and close the UDP transport."""
        self._running = False
        if self._transport:
            self._transport.close()
            self._transport = None

    def _handle_packet(self, data: bytes, addr: Tuple[str, int]):
        """Process an incoming RTP packet."""
        if len(data) < RTP_HEADER_SIZE:
            return
        pt, seq, ts, ssrc = unpack_rtp_header(data)
        payload = data[RTP_HEADER_SIZE:]
        if len(payload) == 0:
            return

        self._received += 1
        self._last_rtp_time = time.monotonic()

        # RFC 2833 DTMF
        if self._dtmf_enabled and pt == DTMF_PAYLOAD_TYPE:
            result = parse_dtmf_event(payload)
            if result:
                digit, end_bit, duration = result
                if end_bit and ts != self._last_dtmf_ts:
                    self._last_dtmf_ts = ts
                    try:
                        self.dtmf_queue.put_nowait(digit)
                    except asyncio.QueueFull:
                        pass
            return

        # Audio: decode G.711 to PCM16
        ulaw = np.frombuffer(payload, dtype=np.uint8)
        pcm16 = self._codec.decode(ulaw)
        pcm_bytes = pcm16.tobytes()

        if self.rx_queue.full():
            try:
                self.rx_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.rx_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            pass

    async def _send_loop(self):
        """Send RTP frames at precise 20ms intervals with pre-buffering."""
        loop = asyncio.get_running_loop()
        next_send = loop.time()
        playing = False
        prebuf_deadline = 0.0

        while self._running:
            next_send += FRAME_DURATION
            qsize = self.tx_queue.qsize()

            if playing:
                if qsize > 0:
                    pcm_bytes = self.tx_queue.get_nowait()
                else:
                    try:
                        pcm_bytes = await asyncio.wait_for(
                            self.tx_queue.get(), timeout=FRAME_DURATION * 2
                        )
                    except asyncio.TimeoutError:
                        pcm_bytes = SILENCE_PCM16
                        playing = False
            elif qsize >= self._prebuffer_frames or (qsize > 0 and loop.time() >= prebuf_deadline):
                playing = True
                pcm_bytes = self.tx_queue.get_nowait()
            else:
                if qsize == 0:
                    prebuf_deadline = loop.time() + self._prebuffer_frames * FRAME_DURATION
                pcm_bytes = SILENCE_PCM16

            pcm16 = np.frombuffer(pcm_bytes[:PCM16_FRAME_SIZE], dtype=np.int16)
            ulaw = self._codec.encode(pcm16)

            header = pack_rtp_header(seq=self._seq, timestamp=self._timestamp, ssrc=self._ssrc)
            packet = header + ulaw.tobytes()

            if self._transport and self._remote_addr:
                try:
                    self._transport.sendto(packet, self._remote_addr)
                except (OSError, AttributeError):
                    break

            self._seq = (self._seq + 1) & 0xFFFF
            self._timestamp = (self._timestamp + SAMPLES_PER_FRAME) & 0xFFFFFFFF
            self._sent += 1

            now = loop.time()
            sleep_time = next_send - now
            if sleep_time > FRAME_DURATION * 2:
                next_send = now + FRAME_DURATION
                sleep_time = FRAME_DURATION
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def wait_for_drain(self, timeout: float = 10.0) -> bool:
        """Wait for tx_queue to drain.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if drained, False on timeout.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while self.tx_queue.qsize() > 0:
            if asyncio.get_running_loop().time() > deadline:
                return False
            await asyncio.sleep(FRAME_DURATION)
        await asyncio.sleep(0.5)
        return True


class _RTPProtocol(asyncio.DatagramProtocol):
    """Asyncio UDP protocol for RTP packet reception."""

    def __init__(self, session: RTPSession):
        self._session = session

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        self._session._handle_packet(data, addr)

    def error_received(self, exc: Exception):
        logger.error("RTP error: %s", exc)

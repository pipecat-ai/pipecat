#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FreeSWITCH SIP/RTP transport for Pipecat.

Provides FreeSwitchSIPServerTransport (SIP listener + port manager) that
produces FreeSwitchSIPCallTransport instances (BaseTransport) per incoming
call. Each call gets its own input/output processors for pipeline
integration. Scoped for LAN use with FreeSWITCH (no NAT/STUN, no
REGISTER, dial-in only).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.sip.codecs import G711Codec, resample_down, resample_up
from pipecat.transports.sip.params import FreeSwitchSIPParams
from pipecat.transports.sip.rtp import RTPSession
from pipecat.transports.sip.sdp import parse_sdp
from pipecat.transports.sip.signaling import (
    SIPMessage,
    SIPMethod,
    build_100_trying,
    build_200_ok,
    build_200_ok_bye,
    build_200_ok_options,
    build_bye,
)
from pipecat.utils.base_object import BaseObject

logger = logging.getLogger(__name__)


@dataclass
class FreeSwitchSIPSession:
    """Per-call SIP session state.

    Parameters:
        call_id: SIP Call-ID header value.
        local_tag: Our tag from the 200 OK response.
        remote_tag: Remote party's tag from the INVITE From header.
        from_header: Original INVITE From header (remote party).
        to_header: Original INVITE To header (us).
        via_header: Original INVITE Via header.
        cseq: Original INVITE CSeq header.
        remote_rtp_addr: Remote RTP endpoint (ip, port).
        local_rtp_port: Our allocated RTP port.
        local_ip: Our IP address.
        local_sip_port: Our SIP listener port.
        codec: Negotiated codec name.
    """

    call_id: str
    local_tag: str
    remote_tag: str
    from_header: str
    to_header: str
    via_header: str
    cseq: str
    remote_rtp_addr: Tuple[str, int]
    local_rtp_port: int
    local_ip: str
    local_sip_port: int
    codec: str = "PCMU"

    rtp_session: RTPSession = field(init=False)
    stopped_event: asyncio.Event = field(init=False)
    _stopped: bool = field(init=False, default=False)
    _bye_sent: bool = field(init=False, default=False)
    _sip_transport: Any = field(init=False, default=None)
    _sip_addr: Tuple[str, int] = field(init=False, default=("", 0))

    prebuffer_frames: int = 3
    dtmf_enabled: bool = True

    def __post_init__(self):
        self.rtp_session = RTPSession(
            local_port=self.local_rtp_port,
            prebuffer_frames=self.prebuffer_frames,
            dtmf_enabled=self.dtmf_enabled,
        )
        self.stopped_event = asyncio.Event()
        self._stopped = False
        self._bye_sent = False

    def set_sip_transport(self, transport: asyncio.DatagramTransport, addr: Tuple[str, int]):
        """Set the SIP transport for sending BYE requests."""
        self._sip_transport = transport
        self._sip_addr = addr

    async def start_rtp(self):
        """Start the RTP session."""
        await self.rtp_session.start(self.remote_rtp_addr)

    def send_bye(self):
        """Send SIP BYE to the remote party."""
        if self._bye_sent:
            return
        self._bye_sent = True
        if self._sip_transport and self._sip_addr[0]:
            bye_msg = build_bye(
                call_id=self.call_id,
                from_header=self.from_header,
                to_header=self.to_header,
                local_tag=self.local_tag,
                local_ip=self.local_ip,
                local_sip_port=self.local_sip_port,
            )
            try:
                self._sip_transport.sendto(bye_msg, self._sip_addr)
                logger.info("SIP BYE sent for call %s", self.call_id)
            except (OSError, AttributeError):
                pass

    async def stop(self):
        """Stop the call: send BYE, stop RTP."""
        if self._stopped:
            return
        self._stopped = True
        self.stopped_event.set()
        self.send_bye()
        await self.rtp_session.stop()


class FreeSwitchSIPInputTransport(BaseInputTransport):
    """Pulls PCM16 from RTP rx_queue, resamples 8k->16k, pushes to pipeline.

    Args:
        session: The FreeSwitchSIPSession for this call.
        params: Transport parameters.
    """

    def __init__(self, session: FreeSwitchSIPSession, params: FreeSwitchSIPParams, **kwargs):
        """Initialize the SIP input transport.

        Args:
            session: The FreeSwitchSIPSession for this call.
            params: Transport parameters.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(params=params, **kwargs)
        self._session = session
        self._rx_task: Optional[asyncio.Task] = None
        self._dtmf_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        """Start the SIP input transport and begin receiving RTP audio.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._rx_task = self.create_task(self._rx_loop(), "sip_rx_loop")
        if self._params.dtmf_enabled:
            self._dtmf_task = self.create_task(self._dtmf_loop(), "sip_dtmf_loop")

    async def stop(self, frame: EndFrame):
        """Stop the SIP input transport and cancel receive tasks.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        if self._rx_task:
            await self.cancel_task(self._rx_task)
        if self._dtmf_task:
            await self.cancel_task(self._dtmf_task)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the SIP input transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        if self._rx_task:
            await self.cancel_task(self._rx_task)
        if self._dtmf_task:
            await self.cancel_task(self._dtmf_task)
        await super().cancel(frame)

    async def _rx_loop(self):
        """Pull from RTP rx_queue, resample 8k->16k, push audio frames."""
        while True:
            try:
                pcm_bytes = await self._session.rtp_session.rx_queue.get()
                pcm_8k = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm_16k = resample_up(pcm_8k, factor=2)
                frame = InputAudioRawFrame(
                    audio=pcm_16k.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                )
                await self.push_audio_frame(frame)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("SIP input error: %s", e)
                await self.push_error(f"SIP input error: {e}", exception=e)

    async def _dtmf_loop(self):
        """Pull DTMF digits from RTP session and push as transport messages."""
        while True:
            try:
                digit = await self._session.rtp_session.dtmf_queue.get()
                frame = InputTransportMessageFrame(message={"type": "dtmf", "digit": digit})
                await self.push_frame(frame)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("SIP DTMF error: %s", e)
                await self.push_error(f"SIP DTMF error: {e}", exception=e)


class FreeSwitchSIPOutputTransport(BaseOutputTransport):
    """Receives pipeline audio, resamples to 8k, encodes G.711, sends via RTP.

    Args:
        session: The FreeSwitchSIPSession for this call.
        params: Transport parameters.
    """

    def __init__(self, session: FreeSwitchSIPSession, params: FreeSwitchSIPParams, **kwargs):
        """Initialize the SIP output transport.

        Args:
            session: The FreeSwitchSIPSession for this call.
            params: Transport parameters.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(params=params, **kwargs)
        self._session = session
        self._codec = G711Codec.get_instance()
        self._rtp_buffer = bytearray()

    async def start(self, frame: StartFrame):
        """Start the SIP output transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write audio to RTP. Resamples to 8kHz and encodes as G.711.

        Args:
            frame: Output audio frame from the pipeline.

        Returns:
            True on success.
        """
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        sample_rate = frame.sample_rate

        if sample_rate == 24000:
            pcm_8k = resample_down(pcm, factor=3)
        elif sample_rate == 16000:
            pcm_8k = resample_down(pcm, factor=2)
        elif sample_rate == 8000:
            pcm_8k = pcm
        else:
            logger.warning("Unexpected sample rate %d, skipping", sample_rate)
            return True

        self._rtp_buffer.extend(pcm_8k.tobytes())

        while len(self._rtp_buffer) >= 320:
            chunk = bytes(self._rtp_buffer[:320])
            del self._rtp_buffer[:320]
            await self._session.rtp_session.tx_queue.put(chunk)

        return True


class FreeSwitchSIPCallTransport(BaseTransport):
    """Per-call transport wrapping SIP session with input/output processors.

    Created by FreeSwitchSIPServerTransport for each incoming call. Provides standard
    BaseTransport interface with input()/output() for pipeline integration.

    Args:
        session: The FreeSwitchSIPSession for this call.
        params: Transport parameters.
    """

    def __init__(self, session: FreeSwitchSIPSession, params: FreeSwitchSIPParams, **kwargs):
        """Initialize the SIP call transport.

        Args:
            session: The FreeSwitchSIPSession for this call.
            params: Transport parameters.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._session = session
        self._params = params
        self._input: Optional[FreeSwitchSIPInputTransport] = None
        self._output: Optional[FreeSwitchSIPOutputTransport] = None

    @property
    def session(self) -> FreeSwitchSIPSession:
        """Access the SIP session state (call_id, headers, etc.)."""
        return self._session

    def input(self) -> FreeSwitchSIPInputTransport:
        """Get the input frame processor for this call transport.

        Returns:
            The SIP input transport that handles incoming RTP audio.
        """
        if not self._input:
            self._input = FreeSwitchSIPInputTransport(
                self._session, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> FreeSwitchSIPOutputTransport:
        """Get the output frame processor for this call transport.

        Returns:
            The SIP output transport that handles outgoing RTP audio.
        """
        if not self._output:
            self._output = FreeSwitchSIPOutputTransport(
                self._session, self._params, name=self._output_name
            )
        return self._output

    async def hangup(self):
        """Initiate a UAS-side hangup (send SIP BYE, stop RTP)."""
        await self._session.stop()


class _FreeSwitchSIPServerProtocol(asyncio.DatagramProtocol):
    """Asyncio UDP protocol for SIP signaling."""

    def __init__(self, server: FreeSwitchSIPServerTransport):
        self._server = server

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        self._server._handle_message(data, addr)

    def error_received(self, exc: Exception):
        logger.error("SIP protocol error: %s", exc)


class FreeSwitchSIPServerTransport(BaseObject):
    """SIP UAS server that listens for incoming calls.

    Manages a UDP SIP listener and RTP port pool. Each incoming INVITE
    creates a FreeSwitchSIPCallTransport and fires the on_call_started event.

    Event handlers available:

    - on_call_started: Fired when a new call is ready.
    - on_call_ended: Fired when a call ends.
    - on_call_failed: Fired on call error.

    Args:
        params: SIP transport parameters.

    Example::

        server = FreeSwitchSIPServerTransport(params=FreeSwitchSIPParams())

        @server.event_handler("on_call_started")
        async def on_call(server, call_transport):
            pipeline = Pipeline([
                call_transport.input(), stt, llm, tts, call_transport.output()
            ])
            task = PipelineTask(pipeline)
            await runner.run(task)

        await server.start()
    """

    def __init__(self, params: FreeSwitchSIPParams, **kwargs):
        """Initialize the SIP server transport.

        Args:
            params: SIP transport parameters.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._params = params
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._active_calls: Dict[str, Tuple[FreeSwitchSIPSession, FreeSwitchSIPCallTransport]] = {}
        self._pending_acks: Dict[str, asyncio.Task] = {}
        self._background_tasks: Set[asyncio.Task] = set()
        self._used_ports: Set[int] = set()
        self._running = False
        self._local_port = 0

        # Register supported event handlers
        self._register_event_handler("on_call_started")
        self._register_event_handler("on_call_ended")
        self._register_event_handler("on_call_failed")

    async def start(self):
        """Start the SIP UDP listener."""
        self._running = True
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _FreeSwitchSIPServerProtocol(self),
            local_addr=(self._params.sip_listen_host, self._params.sip_listen_port),
        )
        actual_addr = self._transport.get_extra_info("sockname")
        if actual_addr:
            self._local_port = actual_addr[1]
        logger.info(
            "SIP server started on %s:%d",
            self._params.sip_listen_host,
            self._local_port,
        )

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create and track a background task."""
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def stop(self):
        """Stop the SIP server and all active calls."""
        self._running = False
        # Cancel pending ACK timers
        for task in self._pending_acks.values():
            task.cancel()
        self._pending_acks.clear()
        # Cancel background tasks
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        # Stop all active calls and release ports
        for session, call_transport in list(self._active_calls.values()):
            self._release_rtp_port(session.local_rtp_port)
            await session.stop()
        self._active_calls.clear()
        if self._transport:
            self._transport.close()
            self._transport = None
        logger.info("SIP server stopped")

    @property
    def local_port(self) -> int:
        """The actual port the server is bound to."""
        return self._local_port

    def _allocate_rtp_port(self) -> int:
        """Allocate an RTP port from the configured range."""
        lo, hi = self._params.rtp_port_range
        for _ in range(100):
            port = random.randint(lo, hi)
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        for port in range(lo, hi + 1):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        raise RuntimeError("No RTP ports available")

    def _release_rtp_port(self, port: int):
        """Return an RTP port to the pool."""
        self._used_ports.discard(port)

    def _handle_message(self, data: bytes, addr: Tuple[str, int]):
        """Dispatch incoming SIP messages."""
        try:
            msg = SIPMessage.parse(data)
        except Exception:
            logger.error("SIP parse error from %s", addr)
            return

        if msg.method == SIPMethod.INVITE:
            if self._running:
                self._handle_invite(msg, addr)
        elif msg.method == SIPMethod.BYE:
            self._handle_bye(msg, addr)
        elif msg.method == SIPMethod.ACK:
            self._handle_ack(msg, addr)
        elif msg.method == SIPMethod.OPTIONS:
            self._handle_options(msg, addr)
        else:
            logger.debug("SIP unhandled method: %s", msg.method)

    def _handle_invite(self, msg: SIPMessage, addr: Tuple[str, int]):
        """Handle INVITE: send 100 Trying, parse SDP, allocate RTP, send 200 OK."""
        if msg.call_id in self._active_calls:
            logger.warning("Duplicate INVITE for call %s", msg.call_id)
            return

        if len(self._active_calls) >= self._params.max_calls:
            logger.warning("Max calls reached, rejecting %s", msg.call_id)
            return

        trying = build_100_trying(invite=msg)
        self._transport.sendto(trying, addr)

        remote_ip, remote_port, codecs = parse_sdp(msg.body or "")
        if not remote_ip or not remote_port:
            logger.error("Invalid SDP in INVITE for call %s", msg.call_id)
            return

        negotiated_codec = "PCMU"
        for pref in self._params.codec_preferences:
            for pt, name in codecs.items():
                if name == pref:
                    negotiated_codec = name
                    break
            else:
                continue
            break

        try:
            local_rtp_port = self._allocate_rtp_port()
        except RuntimeError:
            logger.error("No RTP ports available for call %s", msg.call_id)
            return

        local_ip = self._transport.get_extra_info("sockname")[0]
        if local_ip == "0.0.0.0":
            local_ip = "127.0.0.1"

        session_id = random.randint(1, 0xFFFFFFFF)
        local_tag = f"bot-{session_id}"

        # Extract remote tag from From header
        remote_tag = ""
        if "tag=" in msg.from_header:
            for part in msg.from_header.split(";"):
                if part.strip().startswith("tag="):
                    remote_tag = part.strip()[4:]
                    break

        session = FreeSwitchSIPSession(
            call_id=msg.call_id,
            local_tag=local_tag,
            remote_tag=remote_tag,
            from_header=msg.from_header,
            to_header=msg.to_header,
            via_header=msg.via,
            cseq=msg.cseq,
            remote_rtp_addr=(remote_ip, remote_port),
            local_rtp_port=local_rtp_port,
            local_ip=local_ip,
            local_sip_port=self._local_port,
            codec=negotiated_codec,
            prebuffer_frames=self._params.rtp_prebuffer_frames,
            dtmf_enabled=self._params.dtmf_enabled,
        )
        session.set_sip_transport(self._transport, addr)

        ok = build_200_ok(
            invite=msg,
            local_ip=local_ip,
            local_port=local_rtp_port,
            session_id=session_id,
            local_sip_port=self._local_port,
        )
        self._transport.sendto(ok, addr)

        call_transport = FreeSwitchSIPCallTransport(session=session, params=self._params)

        self._active_calls[msg.call_id] = (session, call_transport)

        ack_task = self._create_background_task(self._ack_timeout(msg.call_id))
        self._pending_acks[msg.call_id] = ack_task

        logger.info(
            "SIP INVITE accepted: call=%s remote_rtp=%s:%d",
            msg.call_id,
            remote_ip,
            remote_port,
        )

    def _handle_ack(self, msg: SIPMessage, addr: Tuple[str, int]):
        """Handle ACK: cancel timeout, start RTP, fire on_call_started."""
        call_id = msg.call_id

        ack_task = self._pending_acks.pop(call_id, None)
        if ack_task:
            ack_task.cancel()

        entry = self._active_calls.get(call_id)
        if not entry:
            return
        session, call_transport = entry

        self._create_background_task(self._start_call(session, call_transport))

    async def _start_call(
        self, session: FreeSwitchSIPSession, call_transport: FreeSwitchSIPCallTransport
    ):
        """Start RTP session and fire on_call_started event."""
        try:
            await session.start_rtp()
            self._create_background_task(self._run_rtp(session))
            if self._params.rtp_dead_timeout_ms > 0:
                self._create_background_task(self._rtp_dead_monitor(session, call_transport))
            await self._call_event_handler("on_call_started", call_transport)
        except Exception as e:
            logger.error("Call start error: %s", e)
            await self._call_event_handler("on_call_failed", call_transport, e)
            await self._cleanup_call(session.call_id)

    async def _run_rtp(self, session: FreeSwitchSIPSession):
        """Run RTP send loop, cleanup on completion."""
        try:
            await session.rtp_session.run()
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("RTP send error for call %s: %s", session.call_id, e)
        # If we exit normally (send loop broke, e.g. OSError), clean up the call
        if session.call_id in self._active_calls:
            entry = self._active_calls.get(session.call_id)
            if entry:
                _, call_transport = entry
                await self._call_event_handler("on_call_ended", call_transport)
            await self._cleanup_call(session.call_id)

    async def _rtp_dead_monitor(
        self, session: FreeSwitchSIPSession, call_transport: FreeSwitchSIPCallTransport
    ):
        """Monitor RTP liveness; teardown call if no packets received within timeout."""
        timeout_s = self._params.rtp_dead_timeout_ms / 1000
        poll_interval = min(timeout_s / 2, 1.0)
        try:
            while session.call_id in self._active_calls:
                await asyncio.sleep(poll_interval)
                elapsed = time.monotonic() - session.rtp_session._last_rtp_time
                if elapsed >= timeout_s:
                    logger.warning("RTP dead timeout (%.1fs) for call %s", elapsed, session.call_id)
                    await self._call_event_handler("on_call_ended", call_transport)
                    await self._cleanup_call(session.call_id)
                    return
        except asyncio.CancelledError:
            pass

    async def _ack_timeout(self, call_id: str):
        """Wait for ACK timeout, then teardown the call."""
        try:
            await asyncio.sleep(self._params.ack_timeout_ms / 1000)
            logger.warning("ACK timeout for call %s", call_id)
            await self._cleanup_call(call_id)
        except asyncio.CancelledError:
            pass

    def _handle_options(self, msg: SIPMessage, addr: Tuple[str, int]):
        """Handle OPTIONS: respond 200 OK to keep SBC probes happy."""
        ok = build_200_ok_options(options=msg)
        self._transport.sendto(ok, addr)
        logger.debug("SIP OPTIONS 200 OK sent to %s", addr)

    def _handle_bye(self, msg: SIPMessage, addr: Tuple[str, int]):
        """Handle BYE: respond 200 OK, teardown call."""
        ok = build_200_ok_bye(bye=msg)
        self._transport.sendto(ok, addr)

        # Cancel pending ACK timeout if still waiting
        ack_task = self._pending_acks.pop(msg.call_id, None)
        if ack_task:
            ack_task.cancel()

        entry = self._active_calls.get(msg.call_id)
        if entry:
            session, call_transport = entry
            self._create_background_task(self._on_bye_received(session, call_transport))
        else:
            logger.debug("BYE for unknown call %s", msg.call_id)

    async def _on_bye_received(
        self, session: FreeSwitchSIPSession, call_transport: FreeSwitchSIPCallTransport
    ):
        """Handle BYE received: stop session, fire event, cleanup."""
        await session.stop()
        await self._call_event_handler("on_call_ended", call_transport)
        self._active_calls.pop(session.call_id, None)
        self._release_rtp_port(session.local_rtp_port)

    async def _cleanup_call(self, call_id: str):
        """Clean up a call: stop session, release port, remove from tracking."""
        entry = self._active_calls.pop(call_id, None)
        if entry:
            session, call_transport = entry
            await session.stop()
            self._release_rtp_port(session.local_rtp_port)
        self._pending_acks.pop(call_id, None)

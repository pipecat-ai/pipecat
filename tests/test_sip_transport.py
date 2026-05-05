#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for FreeSWITCH SIP transport: FreeSwitchSIPServerTransport,
FreeSwitchSIPSession, port allocation, call lifecycle, ACK timeout,
duplicate INVITE rejection, and OPTIONS handling.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.transports.sip.params import FreeSwitchSIPParams
from pipecat.transports.sip.rtp import RTPSession
from pipecat.transports.sip.signaling import SIPMessage, build_200_ok_options
from pipecat.transports.sip.transport import (
    FreeSwitchSIPCallTransport,
    FreeSwitchSIPServerTransport,
    FreeSwitchSIPSession,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SDP_BODY = (
    "v=0\r\n"
    "o=- 123 456 IN IP4 10.0.0.1\r\n"
    "s=-\r\n"
    "c=IN IP4 10.0.0.1\r\n"
    "t=0 0\r\n"
    "m=audio 4000 RTP/AVP 0\r\n"
    "a=rtpmap:0 PCMU/8000\r\n"
)


def _make_invite(call_id: str = "call-001@test", remote_rtp_port: int = 4000) -> bytes:
    """Build a raw INVITE SIP message with SDP."""
    sdp = SDP_BODY.replace("4000", str(remote_rtp_port))
    sdp_bytes = sdp.encode("utf-8")
    msg = (
        f"INVITE sip:bot@192.168.1.1:5060 SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK776\r\n"
        f"From: <sip:alice@10.0.0.1>;tag=abc123\r\n"
        f"To: <sip:bot@192.168.1.1>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 INVITE\r\n"
        f"Content-Type: application/sdp\r\n"
        f"Content-Length: {len(sdp_bytes)}\r\n"
        f"\r\n"
    ).encode("utf-8") + sdp_bytes
    return msg


def _make_ack(call_id: str = "call-001@test") -> bytes:
    msg = (
        f"ACK sip:bot@192.168.1.1 SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bKack\r\n"
        f"From: <sip:alice@10.0.0.1>;tag=abc123\r\n"
        f"To: <sip:bot@192.168.1.1>;tag=bot-1\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 ACK\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    ).encode("utf-8")
    return msg


def _make_bye(call_id: str = "call-001@test") -> bytes:
    msg = (
        f"BYE sip:bot@192.168.1.1 SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bKbye\r\n"
        f"From: <sip:alice@10.0.0.1>;tag=abc123\r\n"
        f"To: <sip:bot@192.168.1.1>;tag=bot-1\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 2 BYE\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    ).encode("utf-8")
    return msg


def _make_options() -> bytes:
    msg = (
        "OPTIONS sip:bot@192.168.1.1 SIP/2.0\r\n"
        "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bKopts\r\n"
        "From: <sip:sbc@10.0.0.1>;tag=sbc1\r\n"
        "To: <sip:bot@192.168.1.1>\r\n"
        "Call-ID: options-001\r\n"
        "CSeq: 1 OPTIONS\r\n"
        "Content-Length: 0\r\n"
        "\r\n"
    ).encode("utf-8")
    return msg


def _mock_transport():
    """Create a mock asyncio.DatagramTransport."""
    transport = MagicMock(spec=asyncio.DatagramTransport)
    transport.get_extra_info.return_value = ("127.0.0.1", 5060)
    transport.sendto = MagicMock()
    transport.close = MagicMock()
    return transport


REMOTE_ADDR = ("10.0.0.1", 5060)


# ---------------------------------------------------------------------------
# FreeSwitchSIPSession unit tests
# ---------------------------------------------------------------------------


class TestFreeSwitchSIPSession:
    def test_session_creation(self):
        """FreeSwitchSIPSession creates an RTPSession with correct port."""
        session = FreeSwitchSIPSession(
            call_id="call-1",
            local_tag="bot-1",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10000,
            local_ip="192.168.1.1",
            local_sip_port=5060,
        )
        assert session.rtp_session.local_port == 10000
        assert session.call_id == "call-1"
        assert not session._stopped

    def test_session_prebuffer_forwarded(self):
        """FreeSwitchSIPSession forwards prebuffer_frames to RTPSession."""
        session = FreeSwitchSIPSession(
            call_id="call-2",
            local_tag="bot-2",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10001,
            local_ip="192.168.1.1",
            local_sip_port=5060,
            prebuffer_frames=5,
        )
        assert session.rtp_session._prebuffer_frames == 5

    def test_session_dtmf_forwarded(self):
        """FreeSwitchSIPSession forwards dtmf_enabled to RTPSession."""
        session = FreeSwitchSIPSession(
            call_id="call-3",
            local_tag="bot-3",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10002,
            local_ip="192.168.1.1",
            local_sip_port=5060,
            dtmf_enabled=False,
        )
        assert session.rtp_session._dtmf_enabled is False

    @pytest.mark.asyncio
    async def test_session_stop_idempotent(self):
        """Calling stop() twice is safe."""
        session = FreeSwitchSIPSession(
            call_id="call-4",
            local_tag="bot-4",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10003,
            local_ip="192.168.1.1",
            local_sip_port=5060,
        )
        await session.stop()
        await session.stop()  # Should not raise
        assert session._stopped
        assert session.stopped_event.is_set()

    def test_send_bye_uses_transport(self):
        """send_bye() sends a BYE message via the SIP transport."""
        session = FreeSwitchSIPSession(
            call_id="call-5",
            local_tag="bot-5",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10004,
            local_ip="192.168.1.1",
            local_sip_port=5060,
        )
        mock_t = _mock_transport()
        session.set_sip_transport(mock_t, REMOTE_ADDR)
        session.send_bye()
        assert mock_t.sendto.call_count == 1
        sent_data = mock_t.sendto.call_args[0][0]
        assert b"BYE" in sent_data

    def test_send_bye_idempotent(self):
        """send_bye() only sends once."""
        session = FreeSwitchSIPSession(
            call_id="call-6",
            local_tag="bot-6",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10005,
            local_ip="192.168.1.1",
            local_sip_port=5060,
        )
        mock_t = _mock_transport()
        session.set_sip_transport(mock_t, REMOTE_ADDR)
        session.send_bye()
        session.send_bye()
        assert mock_t.sendto.call_count == 1


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


class TestPortAllocation:
    def test_allocate_and_release(self):
        """Ports are allocated and released correctly."""
        params = FreeSwitchSIPParams(rtp_port_range=(30000, 30002), max_calls=10)
        server = FreeSwitchSIPServerTransport(params=params)
        p1 = server._allocate_rtp_port()
        p2 = server._allocate_rtp_port()
        p3 = server._allocate_rtp_port()
        assert {p1, p2, p3} == {30000, 30001, 30002}

        with pytest.raises(RuntimeError, match="No RTP ports available"):
            server._allocate_rtp_port()

        server._release_rtp_port(p2)
        p4 = server._allocate_rtp_port()
        assert p4 == p2

    def test_release_unknown_port_safe(self):
        """Releasing a port not in use does not raise."""
        params = FreeSwitchSIPParams(rtp_port_range=(30000, 30010))
        server = FreeSwitchSIPServerTransport(params=params)
        server._release_rtp_port(99999)  # Should not raise


# ---------------------------------------------------------------------------
# FreeSwitchSIPServerTransport message handling
# ---------------------------------------------------------------------------


class TestFreeSwitchSIPServerTransportHandleMessage:
    def _make_server(self, **param_overrides) -> FreeSwitchSIPServerTransport:
        """Create a server with a mock UDP transport attached."""
        params = FreeSwitchSIPParams(
            rtp_port_range=(30000, 30100),
            ack_timeout_ms=500,
            rtp_dead_timeout_ms=0,  # Disable for non-timeout tests
            **param_overrides,
        )
        server = FreeSwitchSIPServerTransport(params=params)
        server._transport = _mock_transport()
        server._local_port = 5060
        server._running = True
        return server

    @pytest.mark.asyncio
    async def test_invite_creates_active_call(self):
        """INVITE creates an entry in _active_calls and sends 100+200."""
        server = self._make_server()
        invite_data = _make_invite("call-100")
        server._handle_message(invite_data, REMOTE_ADDR)

        assert "call-100" in server._active_calls
        session, call_transport = server._active_calls["call-100"]
        assert isinstance(session, FreeSwitchSIPSession)
        assert isinstance(call_transport, FreeSwitchSIPCallTransport)
        # Should have sent 100 Trying + 200 OK = 2 messages
        assert server._transport.sendto.call_count == 2
        await server.stop()

    @pytest.mark.asyncio
    async def test_duplicate_invite_rejected(self):
        """Second INVITE with same Call-ID is rejected."""
        server = self._make_server()
        server._handle_message(_make_invite("call-dup"), REMOTE_ADDR)
        assert "call-dup" in server._active_calls
        send_count = server._transport.sendto.call_count

        server._handle_message(_make_invite("call-dup"), REMOTE_ADDR)
        # No additional messages sent for duplicate
        assert server._transport.sendto.call_count == send_count
        await server.stop()

    @pytest.mark.asyncio
    async def test_max_calls_enforced(self):
        """INVITE rejected when max_calls reached."""
        server = self._make_server(max_calls=1)
        server._handle_message(_make_invite("call-A"), REMOTE_ADDR)
        assert "call-A" in server._active_calls
        send_count = server._transport.sendto.call_count

        server._handle_message(_make_invite("call-B"), REMOTE_ADDR)
        assert "call-B" not in server._active_calls
        # No messages sent for rejected call
        assert server._transport.sendto.call_count == send_count
        await server.stop()

    def test_invite_not_accepted_when_stopped(self):
        """INVITE is ignored when server is not running."""
        server = self._make_server()
        server._running = False
        server._handle_message(_make_invite("call-stopped"), REMOTE_ADDR)
        assert "call-stopped" not in server._active_calls

    def test_options_response(self):
        """OPTIONS probe gets a 200 OK response."""
        server = self._make_server()
        server._handle_message(_make_options(), REMOTE_ADDR)
        assert server._transport.sendto.call_count == 1
        sent_data = server._transport.sendto.call_args[0][0]
        text = sent_data.decode("utf-8")
        assert "200 OK" in text
        assert "Allow:" in text

    @pytest.mark.asyncio
    async def test_bye_for_active_call(self):
        """BYE for an active call sends 200 OK and triggers cleanup."""
        server = self._make_server()
        server._handle_message(_make_invite("call-bye"), REMOTE_ADDR)
        assert "call-bye" in server._active_calls

        server._handle_message(_make_bye("call-bye"), REMOTE_ADDR)
        # Should have sent 200 OK for BYE
        last_sent = server._transport.sendto.call_args[0][0]
        assert b"200 OK" in last_sent
        await server.stop()

    def test_bye_for_unknown_call(self):
        """BYE for unknown call sends 200 OK but doesn't crash."""
        server = self._make_server()
        server._handle_message(_make_bye("call-unknown"), REMOTE_ADDR)
        # Should still send 200 OK
        assert server._transport.sendto.call_count == 1

    def test_invalid_sip_message_ignored(self):
        """Garbled data is silently ignored."""
        server = self._make_server()
        server._handle_message(b"NOT A SIP MESSAGE", REMOTE_ADDR)
        assert server._transport.sendto.call_count == 0


# ---------------------------------------------------------------------------
# ACK timeout
# ---------------------------------------------------------------------------


class TestACKTimeout:
    @pytest.mark.asyncio
    async def test_ack_timeout_cleans_up_call(self):
        """Call is cleaned up after ACK timeout."""
        params = FreeSwitchSIPParams(
            rtp_port_range=(30000, 30100),
            ack_timeout_ms=100,  # Very short for test
            rtp_dead_timeout_ms=0,
        )
        server = FreeSwitchSIPServerTransport(params=params)
        server._transport = _mock_transport()
        server._local_port = 5060
        server._running = True

        server._handle_message(_make_invite("call-timeout"), REMOTE_ADDR)
        assert "call-timeout" in server._active_calls
        assert "call-timeout" in server._pending_acks

        # Wait for the ACK timeout to fire
        await asyncio.sleep(0.3)

        assert "call-timeout" not in server._active_calls
        assert "call-timeout" not in server._pending_acks

    @pytest.mark.asyncio
    async def test_ack_cancels_timeout(self):
        """ACK cancels the timeout timer."""
        params = FreeSwitchSIPParams(
            rtp_port_range=(30000, 30100),
            ack_timeout_ms=5000,  # Long timeout
            rtp_dead_timeout_ms=0,
        )
        server = FreeSwitchSIPServerTransport(params=params)
        server._transport = _mock_transport()
        server._local_port = 5060
        server._running = True

        server._handle_message(_make_invite("call-ack"), REMOTE_ADDR)
        assert "call-ack" in server._pending_acks

        # Send ACK — should cancel timeout and start call
        with patch.object(FreeSwitchSIPSession, "start_rtp", new_callable=AsyncMock):
            server._handle_message(_make_ack("call-ack"), REMOTE_ADDR)

        assert "call-ack" not in server._pending_acks
        # Call should still be in active_calls (started, not timed out)
        assert "call-ack" in server._active_calls

        await server.stop()


# ---------------------------------------------------------------------------
# RTP dead timeout
# ---------------------------------------------------------------------------


class TestRTPDeadTimeout:
    @pytest.mark.asyncio
    async def test_rtp_dead_timeout_triggers_cleanup(self):
        """Call is cleaned up when no RTP received within timeout."""
        params = FreeSwitchSIPParams(
            rtp_port_range=(30000, 30100),
            ack_timeout_ms=5000,
            rtp_dead_timeout_ms=200,  # Very short for test
        )
        server = FreeSwitchSIPServerTransport(params=params)
        server._transport = _mock_transport()
        server._local_port = 5060
        server._running = True

        call_ended = asyncio.Event()

        @server.event_handler("on_call_ended")
        async def on_ended(server_ref, transport):
            call_ended.set()

        server._handle_message(_make_invite("call-dead"), REMOTE_ADDR)

        # Simulate ACK with mocked start_rtp (avoid real UDP binding)
        with patch.object(FreeSwitchSIPSession, "start_rtp", new_callable=AsyncMock):
            server._handle_message(_make_ack("call-dead"), REMOTE_ADDR)

        # Wait enough for the monitor to trigger. The _last_rtp_time was
        # set by the mocked start_rtp through RTPSession.__init__ -> 0.0,
        # so the monitor should detect it as dead quickly.
        # But we need to ensure _last_rtp_time is old enough. Since
        # start_rtp was mocked, the RTPSession.start() was never called,
        # so _last_rtp_time is 0.0 which is definitely old enough.
        await asyncio.sleep(0.5)

        assert "call-dead" not in server._active_calls
        assert call_ended.is_set()

        await server.stop()


# ---------------------------------------------------------------------------
# FreeSwitchSIPCallTransport
# ---------------------------------------------------------------------------


class TestFreeSwitchSIPCallTransport:
    def test_input_output_creation(self):
        """FreeSwitchSIPCallTransport creates input/output transports."""
        session = FreeSwitchSIPSession(
            call_id="call-ct",
            local_tag="bot-ct",
            remote_tag="abc",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            via_header="SIP/2.0/UDP 10.0.0.1:5060",
            cseq="1 INVITE",
            remote_rtp_addr=("10.0.0.1", 4000),
            local_rtp_port=10006,
            local_ip="192.168.1.1",
            local_sip_port=5060,
        )
        params = FreeSwitchSIPParams()
        ct = FreeSwitchSIPCallTransport(session=session, params=params)
        inp = ct.input()
        out = ct.output()
        assert inp is ct.input()  # Same instance
        assert out is ct.output()  # Same instance
        assert ct.session is session


# ---------------------------------------------------------------------------
# build_200_ok_options
# ---------------------------------------------------------------------------


class TestBuild200OkOptions:
    def test_options_response_format(self):
        """200 OK for OPTIONS includes Allow header."""
        data = _make_options()
        msg = SIPMessage.parse(data)
        response = build_200_ok_options(options=msg)
        text = response.decode("utf-8")
        assert "SIP/2.0 200 OK" in text
        assert "Call-ID: options-001" in text
        assert "Allow:" in text
        assert "INVITE" in text
        assert "OPTIONS" in text
        assert "Content-Length: 0" in text


# ---------------------------------------------------------------------------
# Server stop
# ---------------------------------------------------------------------------


class TestServerStop:
    @pytest.mark.asyncio
    async def test_stop_cleans_all(self):
        """stop() cleans up all active calls and pending ACKs."""
        params = FreeSwitchSIPParams(
            rtp_port_range=(30000, 30100),
            ack_timeout_ms=60000,
            rtp_dead_timeout_ms=0,
        )
        server = FreeSwitchSIPServerTransport(params=params)
        server._transport = _mock_transport()
        server._local_port = 5060
        server._running = True

        server._handle_message(_make_invite("call-s1"), REMOTE_ADDR)
        server._handle_message(_make_invite("call-s2"), REMOTE_ADDR)
        assert len(server._active_calls) == 2
        assert len(server._pending_acks) == 2

        await server.stop()

        assert len(server._active_calls) == 0
        assert len(server._pending_acks) == 0
        assert server._transport is None
        assert not server._running

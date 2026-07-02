#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SIP message parsing and response building."""

import pytest

from pipecat.transports.sip.signaling import (
    SIPMessage,
    SIPMethod,
    build_100_trying,
    build_200_ok,
    build_200_ok_bye,
    build_bye,
)


class TestSIPMessageParse:
    def test_parse_invite(self):
        """Parse a basic INVITE request."""
        data = (
            "INVITE sip:bot@192.168.1.1:5060 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK776\r\n"
            "From: <sip:alice@10.0.0.1>;tag=abc123\r\n"
            "To: <sip:bot@192.168.1.1>\r\n"
            "Call-ID: call-001@10.0.0.1\r\n"
            "CSeq: 1 INVITE\r\n"
            "Content-Type: application/sdp\r\n"
            "Content-Length: 10\r\n"
            "\r\n"
            "v=0\r\ntest"
        ).encode()
        msg = SIPMessage.parse(data)
        assert msg.method == SIPMethod.INVITE
        assert msg.call_id == "call-001@10.0.0.1"
        assert "alice" in msg.from_header
        assert msg.body is not None

    def test_parse_bye(self):
        """Parse a BYE request."""
        data = (
            "BYE sip:bot@192.168.1.1 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK999\r\n"
            "From: <sip:alice@10.0.0.1>;tag=abc\r\n"
            "To: <sip:bot@192.168.1.1>;tag=bot-1\r\n"
            "Call-ID: call-002\r\n"
            "CSeq: 2 BYE\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        ).encode()
        msg = SIPMessage.parse(data)
        assert msg.method == SIPMethod.BYE
        assert msg.call_id == "call-002"

    def test_parse_ack(self):
        """Parse an ACK request."""
        data = (
            "ACK sip:bot@192.168.1.1 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bKack\r\n"
            "From: <sip:alice@10.0.0.1>;tag=abc\r\n"
            "To: <sip:bot@192.168.1.1>;tag=bot-1\r\n"
            "Call-ID: call-003\r\n"
            "CSeq: 1 ACK\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        ).encode()
        msg = SIPMessage.parse(data)
        assert msg.method == SIPMethod.ACK

    def test_parse_response(self):
        """Parse a SIP response."""
        data = (
            "SIP/2.0 200 OK\r\n"
            "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK776\r\n"
            "From: <sip:alice@10.0.0.1>;tag=abc\r\n"
            "To: <sip:bot@192.168.1.1>;tag=bot-1\r\n"
            "Call-ID: call-004\r\n"
            "CSeq: 1 INVITE\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        ).encode()
        msg = SIPMessage.parse(data)
        assert msg.method is None
        assert msg.status_code == 200

    def test_parse_unknown_method(self):
        """Unknown method is parsed with method=None."""
        data = (
            "REGISTER sip:reg@10.0.0.1 SIP/2.0\r\nCall-ID: reg-001\r\nCSeq: 1 REGISTER\r\n\r\n"
        ).encode()
        msg = SIPMessage.parse(data)
        assert msg.method is None


class TestSIPResponseBuilding:
    def _make_invite_msg(self) -> SIPMessage:
        data = (
            "INVITE sip:bot@192.168.1.1:5060 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK776\r\n"
            "From: <sip:alice@10.0.0.1>;tag=abc123\r\n"
            "To: <sip:bot@192.168.1.1>\r\n"
            "Call-ID: call-100\r\n"
            "CSeq: 1 INVITE\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        ).encode()
        return SIPMessage.parse(data)

    def test_100_trying(self):
        """100 Trying has correct status line and headers."""
        msg = self._make_invite_msg()
        response = build_100_trying(invite=msg)
        text = response.decode()
        assert text.startswith("SIP/2.0 100 Trying")
        assert "Call-ID: call-100" in text
        assert "Content-Length: 0" in text

    def test_200_ok_contains_sdp(self):
        """200 OK includes SDP body."""
        msg = self._make_invite_msg()
        response = build_200_ok(invite=msg, local_ip="192.168.1.1", local_port=10000, session_id=42)
        text = response.decode()
        assert text.startswith("SIP/2.0 200 OK")
        assert "Content-Type: application/sdp" in text
        assert "m=audio 10000" in text
        assert "tag=bot-42" in text

    def test_200_ok_contact_includes_sip_port(self):
        """200 OK Contact header includes the SIP port so SBC sends ACK to the correct port."""
        msg = self._make_invite_msg()
        response = build_200_ok(
            invite=msg,
            local_ip="172.16.204.30",
            local_port=10000,
            session_id=1,
            local_sip_port=6060,
        )
        text = response.decode()
        assert "Contact: <sip:pipecat@172.16.204.30:6060>" in text

    def test_200_ok_contact_default_port(self):
        """200 OK Contact header uses default SIP port 5060 when not specified."""
        msg = self._make_invite_msg()
        response = build_200_ok(invite=msg, local_ip="10.0.0.1", local_port=10000, session_id=1)
        text = response.decode()
        assert "Contact: <sip:pipecat@10.0.0.1:5060>" in text

    def test_200_ok_contact_non_standard_port(self):
        """200 OK Contact header correctly propagates non-standard SIP port."""
        msg = self._make_invite_msg()
        # Simulate a bot listening on port 9060
        response = build_200_ok(
            invite=msg,
            local_ip="192.168.1.100",
            local_port=12000,
            session_id=7,
            local_sip_port=9060,
        )
        text = response.decode()
        assert "Contact: <sip:pipecat@192.168.1.100:9060>" in text

    def test_200_ok_bye(self):
        """200 OK to BYE has no body."""
        data = (
            "BYE sip:bot@192.168.1.1 SIP/2.0\r\n"
            "Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK999\r\n"
            "From: <sip:alice@10.0.0.1>;tag=abc\r\n"
            "To: <sip:bot@192.168.1.1>;tag=bot-1\r\n"
            "Call-ID: call-200\r\n"
            "CSeq: 2 BYE\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        ).encode()
        msg = SIPMessage.parse(data)
        response = build_200_ok_bye(bye=msg)
        text = response.decode()
        assert "SIP/2.0 200 OK" in text
        assert "Content-Length: 0" in text
        assert "Call-ID: call-200" in text

    def test_build_bye(self):
        """UAS-initiated BYE swaps From/To correctly."""
        response = build_bye(
            call_id="call-300",
            from_header="<sip:alice@10.0.0.1>;tag=abc",
            to_header="<sip:bot@192.168.1.1>",
            local_tag="bot-42",
            local_ip="192.168.1.1",
            local_sip_port=5060,
        )
        text = response.decode()
        assert text.startswith("BYE sip:alice@10.0.0.1 SIP/2.0")
        # From = us (original To + our tag)
        assert "From: <sip:bot@192.168.1.1>;tag=bot-42" in text
        # To = remote (original From, has their tag)
        assert "To: <sip:alice@10.0.0.1>;tag=abc" in text
        assert "Call-ID: call-300" in text

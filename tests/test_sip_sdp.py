#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SDP parsing and generation."""

import pytest

from pipecat.transports.sip.sdp import generate_sdp, parse_sdp


class TestParseSdp:
    def test_parse_basic_sdp(self):
        """Parse SDP with PCMU codec."""
        sdp = (
            "v=0\r\n"
            "o=user 123 456 IN IP4 192.168.1.10\r\n"
            "s=Session\r\n"
            "c=IN IP4 192.168.1.10\r\n"
            "t=0 0\r\n"
            "m=audio 20000 RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
        )
        ip, port, codecs = parse_sdp(sdp)
        assert ip == "192.168.1.10"
        assert port == 20000
        assert "PCMU" in codecs.values()

    def test_parse_multiple_codecs(self):
        """Parse SDP with both PCMU and PCMA."""
        sdp = (
            "v=0\r\n"
            "c=IN IP4 10.0.0.5\r\n"
            "m=audio 30000 RTP/AVP 0 8\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=rtpmap:8 PCMA/8000\r\n"
        )
        ip, port, codecs = parse_sdp(sdp)
        assert ip == "10.0.0.5"
        assert port == 30000
        assert codecs == {0: "PCMU", 8: "PCMA"}

    def test_parse_missing_connection(self):
        """Missing c= line returns empty IP."""
        sdp = "v=0\r\nm=audio 5000 RTP/AVP 0\r\n"
        ip, port, codecs = parse_sdp(sdp)
        assert ip == ""
        assert port == 5000

    def test_parse_missing_media(self):
        """Missing m=audio line returns port 0."""
        sdp = "v=0\r\nc=IN IP4 1.2.3.4\r\n"
        ip, port, codecs = parse_sdp(sdp)
        assert ip == "1.2.3.4"
        assert port == 0

    def test_parse_newline_variants(self):
        """Handle both \\r\\n and \\n line endings."""
        sdp = "v=0\nc=IN IP4 10.0.0.1\nm=audio 8000 RTP/AVP 0\n"
        ip, port, _ = parse_sdp(sdp)
        assert ip == "10.0.0.1"
        assert port == 8000


class TestGenerateSdp:
    def test_generate_contains_required_fields(self):
        """Generated SDP has all required lines."""
        sdp = generate_sdp(local_ip="192.168.1.1", local_port=10000, session_id=42)
        assert "v=0" in sdp
        assert "c=IN IP4 192.168.1.1" in sdp
        assert "m=audio 10000 RTP/AVP 0" in sdp
        assert "a=rtpmap:0 PCMU/8000" in sdp
        assert "a=sendrecv" in sdp

    def test_generate_session_id_in_origin(self):
        """Session ID appears in o= line."""
        sdp = generate_sdp(local_ip="10.0.0.1", local_port=5000, session_id=99)
        assert "o=pipecat 99 99 IN IP4 10.0.0.1" in sdp

    def test_generate_roundtrip(self):
        """Parse a generated SDP back."""
        sdp = generate_sdp(local_ip="172.16.0.1", local_port=12000, session_id=1)
        ip, port, codecs = parse_sdp(sdp)
        assert ip == "172.16.0.1"
        assert port == 12000
        assert 0 in codecs

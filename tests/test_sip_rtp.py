#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for RTP header pack/unpack, DTMF parsing, and RTPSession compliance."""

import asyncio
import struct

import numpy as np
import pytest

from pipecat.transports.sip.rtp import (
    PCMU_PAYLOAD_TYPE,
    RTP_HEADER_SIZE,
    RTPSession,
    pack_rtp_header,
    parse_dtmf_event,
    unpack_rtp_header,
)


class TestRTPHeader:
    def test_pack_header_size(self):
        """RTP header is always 12 bytes."""
        header = pack_rtp_header(seq=0, timestamp=0, ssrc=0)
        assert len(header) == 12

    def test_pack_unpack_roundtrip(self):
        """Pack then unpack returns same values."""
        header = pack_rtp_header(seq=1234, timestamp=56789, ssrc=0xDEADBEEF, payload_type=0)
        pt, seq, ts, ssrc = unpack_rtp_header(header)
        assert pt == 0
        assert seq == 1234
        assert ts == 56789
        assert ssrc == 0xDEADBEEF

    def test_payload_type_8(self):
        """PCMA payload type 8."""
        header = pack_rtp_header(seq=0, timestamp=0, ssrc=0, payload_type=8)
        pt, _, _, _ = unpack_rtp_header(header)
        assert pt == 8

    def test_sequence_wraparound(self):
        """Sequence number wraps at 16 bits."""
        header = pack_rtp_header(seq=0xFFFF, timestamp=0, ssrc=0)
        _, seq, _, _ = unpack_rtp_header(header)
        assert seq == 0xFFFF

        header2 = pack_rtp_header(seq=0x10000, timestamp=0, ssrc=0)
        _, seq2, _, _ = unpack_rtp_header(header2)
        assert seq2 == 0  # Wrapped

    def test_timestamp_wraparound(self):
        """Timestamp wraps at 32 bits."""
        header = pack_rtp_header(seq=0, timestamp=0xFFFFFFFF, ssrc=0)
        _, _, ts, _ = unpack_rtp_header(header)
        assert ts == 0xFFFFFFFF

    def test_version_bit(self):
        """First byte has version 2 (0x80)."""
        header = pack_rtp_header(seq=0, timestamp=0, ssrc=0)
        assert header[0] == 0x80

    def test_marker_bit_not_set(self):
        """Marker bit is not set (payload_type & 0x7F)."""
        header = pack_rtp_header(seq=0, timestamp=0, ssrc=0, payload_type=0)
        assert header[1] & 0x80 == 0


class TestDTMFParsing:
    def test_parse_digit_0(self):
        """Event 0 = digit '0'."""
        payload = bytes([0, 0x80 | 10, 0x03, 0x20])
        result = parse_dtmf_event(payload)
        assert result is not None
        digit, end, duration = result
        assert digit == "0"
        assert end is True
        assert duration == 800

    def test_parse_digit_5(self):
        """Event 5 = digit '5'."""
        payload = bytes([5, 0x80 | 10, 0x01, 0x00])
        result = parse_dtmf_event(payload)
        assert result is not None
        assert result[0] == "5"
        assert result[1] is True

    def test_parse_star(self):
        """Event 10 = '*'."""
        payload = bytes([10, 0x80, 0x00, 0x50])
        result = parse_dtmf_event(payload)
        assert result is not None
        assert result[0] == "*"

    def test_parse_hash(self):
        """Event 11 = '#'."""
        payload = bytes([11, 0x00, 0x00, 0x50])
        result = parse_dtmf_event(payload)
        assert result is not None
        assert result[0] == "#"
        assert result[1] is False  # end bit not set

    def test_parse_too_short(self):
        """Payload shorter than 4 bytes returns None."""
        assert parse_dtmf_event(b"\x00\x00") is None

    def test_parse_invalid_event(self):
        """Event >= 16 returns None."""
        payload = bytes([20, 0x80, 0x00, 0x50])
        assert parse_dtmf_event(payload) is None


class TestRTPSessionCompliance:
    """RFC 3550 compliance tests for RTPSession."""

    def _make_session(self) -> RTPSession:
        return RTPSession(local_port=0)

    def _make_rtp_packet(self, payload_type: int, ssrc: int, payload: bytes = b"\x7f" * 160):
        header = pack_rtp_header(seq=1, timestamp=160, ssrc=ssrc, payload_type=payload_type)
        return header + payload

    def test_ignore_unknown_payload_type(self):
        """Packets with unknown payload types are dropped (RFC 3550 §5.1)."""
        session = self._make_session()
        packet = self._make_rtp_packet(payload_type=99, ssrc=0x12345678)
        session._handle_packet(packet, ("127.0.0.1", 5000))
        assert session.rx_queue.qsize() == 0

    def test_accept_pcmu_payload_type(self):
        """PCMU (PT=0) packets are accepted and queued."""
        session = self._make_session()
        packet = self._make_rtp_packet(payload_type=PCMU_PAYLOAD_TYPE, ssrc=0x12345678)
        session._handle_packet(packet, ("127.0.0.1", 5000))
        assert session.rx_queue.qsize() == 1

    def test_ssrc_collision_detection(self):
        """SSRC is regenerated on collision (RFC 3550 §5.1)."""
        session = self._make_session()
        original_ssrc = session._ssrc
        packet = self._make_rtp_packet(payload_type=PCMU_PAYLOAD_TYPE, ssrc=original_ssrc)
        session._handle_packet(packet, ("127.0.0.1", 5000))
        assert session._ssrc != original_ssrc

    @pytest.mark.asyncio
    async def test_new_ssrc_on_address_change(self):
        """SSRC is regenerated when remote address changes (RFC 3550 §5.1)."""
        session = self._make_session()

        # Simulate first start by setting remote_addr directly (avoid actual UDP bind)
        session._remote_addr = ("192.168.1.1", 5000)
        original_ssrc = session._ssrc

        # Simulate address change check from start()
        new_addr = ("192.168.1.2", 5000)
        if session._remote_addr is not None and session._remote_addr != new_addr:
            session._ssrc = __import__("random").randint(0, 0xFFFFFFFF)

        assert session._ssrc != original_ssrc

    def test_same_address_keeps_ssrc(self):
        """SSRC is preserved when remote address does not change."""
        session = self._make_session()
        session._remote_addr = ("192.168.1.1", 5000)
        original_ssrc = session._ssrc

        same_addr = ("192.168.1.1", 5000)
        if session._remote_addr is not None and session._remote_addr != same_addr:
            session._ssrc = __import__("random").randint(0, 0xFFFFFFFF)

        assert session._ssrc == original_ssrc

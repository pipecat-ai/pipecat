#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for RTP header pack/unpack and DTMF parsing."""

import struct

import numpy as np
import pytest

from pipecat.transports.sip.rtp import pack_rtp_header, parse_dtmf_event, unpack_rtp_header


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

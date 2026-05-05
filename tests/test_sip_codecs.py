#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for G.711 PCMU codec."""

import numpy as np
import pytest

from pipecat.transports.sip.codecs import G711Codec


class TestG711Codec:
    def setup_method(self):
        G711Codec._instance = None
        self.codec = G711Codec()

    def test_silence_encodes_to_0xff(self):
        """Silence (0) encodes to 0xFF in mu-law."""
        silence = np.zeros(160, dtype=np.int16)
        encoded = self.codec.encode(silence)
        assert encoded.dtype == np.uint8
        assert len(encoded) == 160
        assert np.all(encoded == 0xFF)

    def test_encode_decode_roundtrip(self):
        """Encode then decode should approximate original within G.711 quantization error."""
        original = np.linspace(-16000, 16000, 160, dtype=np.int16)
        encoded = self.codec.encode(original)
        decoded = self.codec.decode(encoded)
        assert decoded.dtype == np.int16
        assert len(decoded) == 160
        max_error = np.max(np.abs(original.astype(np.int32) - decoded.astype(np.int32)))
        assert max_error < 1000

    def test_decode_encode_roundtrip(self):
        """Decode then encode should produce exact same bytes (idempotent).

        Note: mu-law byte 0x7F ("negative zero") decodes to 0 which re-encodes
        as 0xFF ("positive zero"). This is an inherent G.711 property, so we
        exclude that single byte from the exact-match check.
        """
        ulaw_bytes = np.arange(256, dtype=np.uint8)
        decoded = self.codec.decode(ulaw_bytes)
        re_encoded = self.codec.encode(decoded)
        # Exclude byte 127 (0x7F) which is "negative zero" — it round-trips to 0xFF
        mask = ulaw_bytes != 127
        assert np.array_equal(ulaw_bytes[mask], re_encoded[mask])
        # Verify 0x7F specifically maps to 0xFF (positive zero)
        assert re_encoded[127] == 0xFF

    def test_encode_output_shape(self):
        """Output shape matches input shape."""
        pcm = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        encoded = self.codec.encode(pcm)
        assert len(encoded) == 320

    def test_decode_output_shape(self):
        """Output shape matches input shape."""
        ulaw = np.random.randint(0, 255, size=160, dtype=np.uint8)
        decoded = self.codec.decode(ulaw)
        assert len(decoded) == 160

    def test_singleton(self):
        """get_instance returns the same object."""
        a = G711Codec.get_instance()
        b = G711Codec.get_instance()
        assert a is b

    def test_positive_max_clipping(self):
        """Values above +32635 are clipped."""
        pcm = np.array([32767, 32635, 32636], dtype=np.int16)
        encoded = self.codec.encode(pcm)
        assert encoded[0] == encoded[1]

    def test_negative_values(self):
        """Negative values encode and decode correctly."""
        pcm = np.array([-1000, -5000, -16000], dtype=np.int16)
        encoded = self.codec.encode(pcm)
        decoded = self.codec.decode(encoded)
        for i in range(3):
            assert abs(int(pcm[i]) - int(decoded[i])) < 500

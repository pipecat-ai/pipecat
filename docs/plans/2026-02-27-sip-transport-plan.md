# SIP/RTP Transport Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a SIP UAS transport to Pipecat enabling PBX/SBC calls to enter pipelines directly via G.711 RTP.

**Architecture:** `SIPServerTransport` (BaseObject) manages a UDP SIP listener and RTP port pool. Each incoming call produces a `SIPCallTransport` (BaseTransport) with per-call `input()`/`output()` processors. Audio flows as 20ms G.711 RTP frames, decoded/resampled to 16kHz PCM for the pipeline.

**Tech Stack:** Python 3.10+, asyncio UDP, numpy (G.711 LUT codec + resampling), struct (RTP headers)

**Reference implementation:** `/Users/rasonyang/workspaces/mlx/pipecat-bot` — proven SIP/RTP code that this plan adapts to Pipecat's transport architecture.

---

### Task 1: Project Setup

**Files:**
- Create: `src/pipecat/transports/sip/__init__.py`
- Create: `src/pipecat/transports/sip/params.py`

**Step 1: Create branch**

```bash
git checkout -b feature/sip-transport
```

**Step 2: Create directory structure**

```bash
mkdir -p src/pipecat/transports/sip
```

**Step 3: Write `params.py`**

```python
"""SIP transport parameters."""

from __future__ import annotations

from typing import List, Tuple

from pipecat.transports.base_transport import TransportParams


class SIPParams(TransportParams):
    """Parameters for SIP/RTP transport.

    Extends TransportParams with SIP-specific configuration for signaling,
    RTP media, and codec settings.

    Parameters:
        sip_listen_host: Bind address for SIP UDP listener.
        sip_listen_port: Port for SIP UDP listener.
        rtp_port_range: Range of UDP ports for RTP media allocation.
        codec_preferences: Ordered list of preferred codecs for SDP negotiation.
        ptime_ms: Packetization time in milliseconds.
        rtp_prebuffer_frames: Number of frames to buffer before TX playback starts.
        rtp_dead_timeout_ms: Teardown call if no RTP received for this duration.
        ack_timeout_ms: Teardown call if ACK not received after 200 OK.
        max_calls: Maximum number of concurrent calls.
        dtmf_enabled: Enable RFC 2833 DTMF digit detection.
    """

    sip_listen_host: str = "0.0.0.0"
    sip_listen_port: int = 5060
    rtp_port_range: Tuple[int, int] = (10000, 20000)
    codec_preferences: List[str] = ["PCMU", "PCMA"]
    ptime_ms: int = 20
    rtp_prebuffer_frames: int = 3
    rtp_dead_timeout_ms: int = 5000
    ack_timeout_ms: int = 3000
    max_calls: int = 100
    dtmf_enabled: bool = True

    # Override TransportParams defaults for SIP
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 16000
```

**Step 4: Write empty `__init__.py`**

```python
"""SIP/RTP transport for Pipecat."""
```

(Exports will be added in Task 11 after all classes exist.)

**Step 5: Commit**

```bash
git add src/pipecat/transports/sip/
git commit -m "feat(sip): scaffold SIP transport directory and params"
```

---

### Task 2: G.711 Codec

**Files:**
- Create: `src/pipecat/transports/sip/codecs.py`
- Create: `tests/test_sip_codecs.py`

**Step 1: Write the failing tests**

```python
"""Tests for G.711 PCMU codec."""

import numpy as np
import pytest

from pipecat.transports.sip.codecs import G711Codec


class TestG711Codec:
    def setup_method(self):
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
        # Use a ramp signal covering the dynamic range
        original = np.linspace(-16000, 16000, 160, dtype=np.int16)
        encoded = self.codec.encode(original)
        decoded = self.codec.decode(encoded)
        assert decoded.dtype == np.int16
        assert len(decoded) == 160
        # G.711 has ~13-bit precision, max quantization error ~256 for large values
        max_error = np.max(np.abs(original.astype(np.int32) - decoded.astype(np.int32)))
        assert max_error < 1000

    def test_decode_encode_roundtrip(self):
        """Decode then encode should produce exact same bytes (idempotent)."""
        ulaw_bytes = np.arange(256, dtype=np.uint8)
        decoded = self.codec.decode(ulaw_bytes)
        re_encoded = self.codec.encode(decoded)
        assert np.array_equal(ulaw_bytes, re_encoded)

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
        # 32767 and 32636 both clip to same value as 32635
        assert encoded[0] == encoded[1]

    def test_negative_values(self):
        """Negative values encode and decode correctly."""
        pcm = np.array([-1000, -5000, -16000], dtype=np.int16)
        encoded = self.codec.encode(pcm)
        decoded = self.codec.decode(encoded)
        for i in range(3):
            assert abs(int(pcm[i]) - int(decoded[i])) < 500
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sip_codecs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipecat.transports.sip.codecs'`

**Step 3: Write implementation**

Create `src/pipecat/transports/sip/codecs.py`:

```python
"""G.711 mu-law (PCMU) codec using NumPy lookup tables.

Provides fast encode/decode via pre-built 65536-entry (encode) and 256-entry
(decode) lookup tables. Used by the RTP session to convert between PCM16 and
G.711 wire format.
"""

from __future__ import annotations

import numpy as np

# mu-law constants
_BIAS = 0x84
_CLIP = 32635
_EXP_LUT = np.array([0, 132, 396, 924, 1980, 4092, 8316, 16764], dtype=np.int16)


def _build_encode_lut() -> np.ndarray:
    """Build 65536-entry encode LUT: int16 sample -> uint8 mu-law byte."""
    lut = np.zeros(65536, dtype=np.uint8)
    for i in range(65536):
        sample = i if i < 32768 else i - 65536
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample
        sample = min(sample, _CLIP)
        sample += _BIAS
        exponent = 7
        exp_mask = 0x4000
        for _ in range(8):
            if sample & exp_mask:
                break
            exponent -= 1
            exp_mask >>= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        lut[i & 0xFFFF] = byte
    return lut


def _build_decode_lut() -> np.ndarray:
    """Build 256-entry decode LUT: uint8 mu-law byte -> int16 sample."""
    lut = np.zeros(256, dtype=np.int16)
    for i in range(256):
        b = ~i & 0xFF
        sign = b & 0x80
        exponent = (b >> 4) & 0x07
        mantissa = b & 0x0F
        sample = int(_EXP_LUT[exponent]) + (mantissa << (exponent + 3))
        if sign:
            sample = -sample
        lut[i] = np.int16(max(-32768, min(32767, sample)))
    return lut


class G711Codec:
    """G.711 mu-law codec with lazy-initialized LUT singleton."""

    _instance: G711Codec | None = None

    def __init__(self):
        self._encode_lut = _build_encode_lut()
        self._decode_lut = _build_decode_lut()

    @classmethod
    def get_instance(cls) -> G711Codec:
        """Get or create the singleton codec instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, pcm: np.ndarray) -> np.ndarray:
        """Encode int16 PCM samples to mu-law bytes.

        Args:
            pcm: Array of int16 PCM samples.

        Returns:
            Array of uint8 mu-law encoded bytes.
        """
        indices = pcm.view(np.uint16)
        return self._encode_lut[indices]

    def decode(self, ulaw: np.ndarray) -> np.ndarray:
        """Decode mu-law bytes to int16 PCM samples.

        Args:
            ulaw: Array of uint8 mu-law encoded bytes.

        Returns:
            Array of int16 PCM samples.
        """
        return self._decode_lut[ulaw]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sip_codecs.py -v
```

Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add src/pipecat/transports/sip/codecs.py tests/test_sip_codecs.py
git commit -m "feat(sip): add G.711 mu-law codec with LUT tables"
```

---

### Task 3: Resampling

**Files:**
- Modify: `src/pipecat/transports/sip/codecs.py` (add resample functions)
- Create: `tests/test_sip_resample.py`

**Step 1: Write the failing tests**

```python
"""Tests for integer-ratio resampling."""

import numpy as np
import pytest

from pipecat.transports.sip.codecs import resample_down, resample_up


class TestResampleUp:
    def test_8k_to_16k_length(self):
        """160 samples at 8kHz becomes 320 at 16kHz."""
        samples = np.zeros(160, dtype=np.int16)
        result = resample_up(samples, factor=2)
        assert len(result) == 320
        assert result.dtype == np.int16

    def test_empty_input(self):
        """Empty array returns empty array."""
        result = resample_up(np.array([], dtype=np.int16), factor=2)
        assert len(result) == 0

    def test_preserves_dc(self):
        """Constant signal stays constant after upsampling."""
        samples = np.full(160, 1000, dtype=np.int16)
        result = resample_up(samples, factor=2)
        assert np.all(result == 1000)

    def test_sine_wave_roundtrip(self):
        """Upsample then downsample of a sine wave preserves shape."""
        t = np.arange(160, dtype=np.float64)
        sine = (10000 * np.sin(2 * np.pi * t / 160)).astype(np.int16)
        up = resample_up(sine, factor=2)
        down = resample_down(up, factor=2)
        # Allow some error from filter + interpolation
        max_err = np.max(np.abs(sine.astype(np.int32) - down[:160].astype(np.int32)))
        assert max_err < 2000

    def test_generic_factor(self):
        """Non-standard factor works."""
        samples = np.arange(80, dtype=np.int16)
        result = resample_up(samples, factor=3)
        assert len(result) == 240


class TestResampleDown:
    def test_16k_to_8k_length(self):
        """320 samples at 16kHz becomes 160 at 8kHz."""
        samples = np.zeros(320, dtype=np.int16)
        result = resample_down(samples, factor=2)
        assert len(result) == 160
        assert result.dtype == np.int16

    def test_24k_to_8k_length(self):
        """480 samples at 24kHz becomes 160 at 8kHz."""
        samples = np.zeros(480, dtype=np.int16)
        result = resample_down(samples, factor=3)
        assert len(result) == 160

    def test_empty_input(self):
        """Empty array returns empty array."""
        result = resample_down(np.array([], dtype=np.int16), factor=2)
        assert len(result) == 0

    def test_preserves_dc(self):
        """Constant signal stays constant after downsampling."""
        samples = np.full(320, 5000, dtype=np.int16)
        result = resample_down(samples, factor=2)
        assert np.allclose(result, 5000, atol=1)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sip_resample.py -v
```

Expected: FAIL with `ImportError: cannot import name 'resample_down' from 'pipecat.transports.sip.codecs'`

**Step 3: Add resample functions to `codecs.py`**

Append to `src/pipecat/transports/sip/codecs.py`:

```python
# Pre-computed index arrays for 2x upsampling (160 -> 320 samples).
_UP2_X_OLD = np.arange(160, dtype=np.float64)
_UP2_X_NEW = np.arange(320, dtype=np.float64) / 2


def resample_up(samples: np.ndarray, factor: int) -> np.ndarray:
    """Upsample by integer factor using linear interpolation.

    Args:
        samples: Input int16 PCM samples.
        factor: Upsampling factor (e.g. 2 for 8kHz -> 16kHz).

    Returns:
        Upsampled int16 PCM array.
    """
    if len(samples) == 0:
        return np.array([], dtype=np.int16)
    # Fast path for 2x (8kHz -> 16kHz) with pre-computed indices
    if factor == 2 and len(samples) == 160:
        result = np.interp(_UP2_X_NEW, _UP2_X_OLD, samples.astype(np.float64))
        return np.clip(result, -32768, 32767).astype(np.int16)
    indices = np.arange(len(samples) * factor, dtype=np.float64) / factor
    result = np.interp(indices, np.arange(len(samples)), samples.astype(np.float64))
    return np.clip(result, -32768, 32767).astype(np.int16)


def resample_down(samples: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by integer factor with moving-average anti-alias filter.

    Args:
        samples: Input int16 PCM samples.
        factor: Downsampling factor (e.g. 2 for 16kHz -> 8kHz).

    Returns:
        Downsampled int16 PCM array.
    """
    if len(samples) == 0:
        return np.array([], dtype=np.int16)
    pad = factor - 1
    sig = samples.astype(np.float32)
    padded = np.pad(sig, (pad, pad), mode="edge")
    kernel = np.ones(factor, dtype=np.float32) / factor
    filtered = np.convolve(padded, kernel, mode="same")
    filtered = filtered[pad : pad + len(sig)]
    decimated = filtered[::factor]
    return np.clip(decimated, -32768, 32767).astype(np.int16)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sip_resample.py -v
```

Expected: All 9 tests PASS.

**Step 5: Commit**

```bash
git add src/pipecat/transports/sip/codecs.py tests/test_sip_resample.py
git commit -m "feat(sip): add integer-ratio resampling functions"
```

---

### Task 4: SDP Parsing and Generation

**Files:**
- Create: `src/pipecat/transports/sip/sdp.py`
- Create: `tests/test_sip_sdp.py`

**Step 1: Write the failing tests**

```python
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
        assert "PCMU" in codecs

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
        assert "m=audio 10000 RTP/AVP 0 8" in sdp
        assert "a=rtpmap:0 PCMU/8000" in sdp
        assert "a=rtpmap:8 PCMA/8000" in sdp
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
        assert 0 in codecs  # PCMU
        assert 8 in codecs  # PCMA
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sip_sdp.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
"""SDP (Session Description Protocol) parsing and generation for SIP transport."""

from __future__ import annotations

import re
from typing import Dict, Tuple

# Codec name -> RTP payload type
CODEC_PT: Dict[str, int] = {"PCMU": 0, "PCMA": 8}
# RTP payload type -> codec name
PT_CODEC: Dict[int, str] = {v: k for k, v in CODEC_PT.items()}


def parse_sdp(sdp: str) -> Tuple[str, int, Dict[int, str]]:
    """Parse SDP to extract connection IP, audio port, and codec map.

    Args:
        sdp: Raw SDP text.

    Returns:
        Tuple of (ip, port, codecs) where codecs maps payload type to codec name.
    """
    ip = ""
    port = 0
    codecs: Dict[int, str] = {}

    for line in sdp.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if line.startswith("c="):
            # c=IN IP4 192.168.1.10
            parts = line.split()
            if len(parts) >= 3:
                ip = parts[-1]
        elif line.startswith("m=audio"):
            # m=audio 20000 RTP/AVP 0 8
            parts = line.split()
            if len(parts) >= 4:
                port = int(parts[1])
                for pt_str in parts[3:]:
                    try:
                        pt = int(pt_str)
                        if pt in PT_CODEC:
                            codecs[pt] = PT_CODEC[pt]
                    except ValueError:
                        pass
        elif line.startswith("a=rtpmap:"):
            # a=rtpmap:0 PCMU/8000
            m = re.match(r"a=rtpmap:(\d+)\s+(\w+)/(\d+)", line)
            if m:
                pt = int(m.group(1))
                name = m.group(2)
                codecs[pt] = name

    return ip, port, codecs


def generate_sdp(*, local_ip: str, local_port: int, session_id: int) -> str:
    """Generate SDP answer for a 200 OK response.

    Args:
        local_ip: Local IP address for media.
        local_port: Local RTP port.
        session_id: Session identifier for o= line.

    Returns:
        SDP string with CRLF line endings.
    """
    return (
        f"v=0\r\n"
        f"o=pipecat {session_id} {session_id} IN IP4 {local_ip}\r\n"
        f"s=Pipecat SIP Transport\r\n"
        f"c=IN IP4 {local_ip}\r\n"
        f"t=0 0\r\n"
        f"m=audio {local_port} RTP/AVP 0 8\r\n"
        f"a=rtpmap:0 PCMU/8000\r\n"
        f"a=rtpmap:8 PCMA/8000\r\n"
        f"a=sendrecv\r\n"
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sip_sdp.py -v
```

Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add src/pipecat/transports/sip/sdp.py tests/test_sip_sdp.py
git commit -m "feat(sip): add SDP parsing and generation"
```

---

### Task 5: RTP Header Utilities

**Files:**
- Create: `src/pipecat/transports/sip/rtp.py`
- Create: `tests/test_sip_rtp.py`

**Step 1: Write the failing tests**

```python
"""Tests for RTP header pack/unpack and DTMF parsing."""

import struct

import numpy as np
import pytest

from pipecat.transports.sip.rtp import pack_rtp_header, unpack_rtp_header


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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sip_rtp.py -v
```

Expected: FAIL

**Step 3: Write implementation**

Create `src/pipecat/transports/sip/rtp.py` with header utilities and RTPSession class:

```python
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


def pack_rtp_header(
    *, seq: int, timestamp: int, ssrc: int, payload_type: int = 0
) -> bytes:
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
        self._last_dtmf_ts = -1  # For dedup

    async def start(self, remote_addr: Tuple[str, int]):
        """Bind UDP socket and prepare for send/receive.

        Args:
            remote_addr: Remote RTP endpoint (ip, port).
        """
        self._remote_addr = remote_addr
        self._running = True

        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _RTPProtocol(self),
            local_addr=("0.0.0.0", self.local_port),
        )
        actual_addr = self._transport.get_extra_info("sockname")
        if actual_addr:
            self.local_port = actual_addr[1]

        logger.info(
            "RTP session started on port %d -> %s", self.local_port, remote_addr
        )

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

        # Drop oldest if queue full
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
            elif qsize >= self._prebuffer_frames or (
                qsize > 0 and loop.time() >= prebuf_deadline
            ):
                playing = True
                pcm_bytes = self.tx_queue.get_nowait()
            else:
                if qsize == 0:
                    prebuf_deadline = (
                        loop.time() + self._prebuffer_frames * FRAME_DURATION
                    )
                pcm_bytes = SILENCE_PCM16

            pcm16 = np.frombuffer(pcm_bytes[:PCM16_FRAME_SIZE], dtype=np.int16)
            ulaw = self._codec.encode(pcm16)

            header = pack_rtp_header(
                seq=self._seq, timestamp=self._timestamp, ssrc=self._ssrc
            )
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
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sip_rtp.py -v
```

Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/pipecat/transports/sip/rtp.py tests/test_sip_rtp.py
git commit -m "feat(sip): add RTP header utilities and session"
```

---

### Task 6: DTMF Detection Tests

**Files:**
- Modify: `tests/test_sip_rtp.py` (add DTMF tests)

**Step 1: Add DTMF tests to the existing test file**

Append to `tests/test_sip_rtp.py`:

```python
from pipecat.transports.sip.rtp import parse_dtmf_event


class TestDTMFParsing:
    def test_parse_digit_0(self):
        """Event 0 = digit '0'."""
        # event=0, end=1, volume=10, duration=800
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
```

**Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_sip_rtp.py -v
```

Expected: All 13 tests PASS (7 header + 6 DTMF).

**Step 3: Commit**

```bash
git add tests/test_sip_rtp.py
git commit -m "test(sip): add RFC 2833 DTMF parsing tests"
```

---

### Task 7: SIP Message Parsing and Response Building

**Files:**
- Create: `src/pipecat/transports/sip/signaling.py`
- Create: `tests/test_sip_signaling.py`

**Step 1: Write the failing tests**

```python
"""Tests for SIP message parsing and response building."""

import pytest

from pipecat.transports.sip.signaling import SIPMessage, SIPMethod, build_200_ok, build_200_ok_bye, build_100_trying, build_bye


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
            "REGISTER sip:reg@10.0.0.1 SIP/2.0\r\n"
            "Call-ID: reg-001\r\n"
            "CSeq: 1 REGISTER\r\n"
            "\r\n"
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
        response = build_200_ok(
            invite=msg, local_ip="192.168.1.1", local_port=10000, session_id=42
        )
        text = response.decode()
        assert text.startswith("SIP/2.0 200 OK")
        assert "Content-Type: application/sdp" in text
        assert "m=audio 10000" in text
        assert "tag=bot-42" in text

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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sip_signaling.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
"""SIP message parsing and response building.

Handles the minimum SIP signaling needed for a UAS: parsing incoming
INVITE/BYE/ACK requests and building 100 Trying, 200 OK, and BYE responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from pipecat.transports.sip.sdp import generate_sdp


class SIPMethod(Enum):
    """Supported SIP request methods."""

    INVITE = "INVITE"
    ACK = "ACK"
    BYE = "BYE"
    CANCEL = "CANCEL"
    OPTIONS = "OPTIONS"


@dataclass
class SIPMessage:
    """Parsed SIP message (request or response).

    Parameters:
        method: SIP method for requests (None for responses).
        request_uri: Request URI (None for responses).
        status_code: Status code for responses (None for requests).
        headers: Dict of header name -> value.
        body: Message body (SDP, etc.) or None.
    """

    method: Optional[SIPMethod]
    request_uri: Optional[str]
    status_code: Optional[int]
    headers: Dict[str, str]
    body: Optional[str]

    @property
    def call_id(self) -> str:
        return self.headers.get("Call-ID", "")

    @property
    def from_header(self) -> str:
        return self.headers.get("From", "")

    @property
    def to_header(self) -> str:
        return self.headers.get("To", "")

    @property
    def via(self) -> str:
        return self.headers.get("Via", "")

    @property
    def cseq(self) -> str:
        return self.headers.get("CSeq", "")

    @classmethod
    def parse(cls, data: bytes) -> SIPMessage:
        """Parse raw bytes into a SIPMessage.

        Args:
            data: Raw SIP message bytes.

        Returns:
            Parsed SIPMessage.
        """
        text = data.decode("utf-8", errors="replace")
        head, _, body = text.partition("\r\n\r\n")
        lines = head.split("\r\n")
        first_line = lines[0]

        method = None
        request_uri = None
        status_code = None

        if first_line.startswith("SIP/"):
            parts = first_line.split(" ", 2)
            status_code = int(parts[1])
        else:
            parts = first_line.split(" ", 2)
            try:
                method = SIPMethod(parts[0])
            except ValueError:
                method = None
            request_uri = parts[1] if len(parts) > 1 else None

        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip()] = value.strip()

        return cls(
            method=method,
            request_uri=request_uri,
            status_code=status_code,
            headers=headers,
            body=body.strip() if body.strip() else None,
        )


def build_100_trying(*, invite: SIPMessage) -> bytes:
    """Build a 100 Trying response to an INVITE.

    Args:
        invite: The parsed INVITE message.

    Returns:
        Encoded SIP response bytes.
    """
    response = (
        f"SIP/2.0 100 Trying\r\n"
        f"Via: {invite.via}\r\n"
        f"From: {invite.from_header}\r\n"
        f"To: {invite.to_header}\r\n"
        f"Call-ID: {invite.call_id}\r\n"
        f"CSeq: {invite.cseq}\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return response.encode("utf-8")


def build_200_ok(
    *, invite: SIPMessage, local_ip: str, local_port: int, session_id: int
) -> bytes:
    """Build a 200 OK response to an INVITE with SDP answer.

    Args:
        invite: The parsed INVITE message.
        local_ip: Local IP for SDP.
        local_port: Local RTP port for SDP.
        session_id: Session ID for SDP and To tag.

    Returns:
        Encoded SIP response bytes with SDP body.
    """
    sdp = generate_sdp(local_ip=local_ip, local_port=local_port, session_id=session_id)
    sdp_bytes = sdp.encode("utf-8")
    response = (
        f"SIP/2.0 200 OK\r\n"
        f"Via: {invite.via}\r\n"
        f"From: {invite.from_header}\r\n"
        f"To: {invite.to_header};tag=bot-{session_id}\r\n"
        f"Call-ID: {invite.call_id}\r\n"
        f"CSeq: {invite.cseq}\r\n"
        f"Contact: <sip:pipecat@{local_ip}>\r\n"
        f"Content-Type: application/sdp\r\n"
        f"Content-Length: {len(sdp_bytes)}\r\n"
        f"\r\n"
    )
    return response.encode("utf-8") + sdp_bytes


def build_200_ok_bye(*, bye: SIPMessage) -> bytes:
    """Build a 200 OK response to a BYE request.

    Args:
        bye: The parsed BYE message.

    Returns:
        Encoded SIP response bytes.
    """
    response = (
        f"SIP/2.0 200 OK\r\n"
        f"Via: {bye.via}\r\n"
        f"From: {bye.from_header}\r\n"
        f"To: {bye.to_header}\r\n"
        f"Call-ID: {bye.call_id}\r\n"
        f"CSeq: {bye.cseq}\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return response.encode("utf-8")


def _extract_uri(header: str) -> str:
    """Extract SIP URI from a header like '<sip:user@host>;tag=...'."""
    m = re.search(r"<(sip:[^>]+)>", header)
    return m.group(1) if m else "sip:unknown@0.0.0.0"


def _append_tag(header: str, tag: str) -> str:
    """Append ;tag=... to a SIP header if not already present."""
    if "tag=" in header:
        return header
    return f"{header};tag={tag}"


def build_bye(
    *,
    call_id: str,
    from_header: str,
    to_header: str,
    local_tag: str,
    local_ip: str,
    local_sip_port: int,
) -> bytes:
    """Build a UAS-initiated BYE request.

    In a UAS-initiated BYE, From/To are swapped relative to the original INVITE:
    From = us (original To + our tag), To = remote (original From with their tag).

    Args:
        call_id: Call-ID from the original INVITE.
        from_header: Original INVITE From header (remote party, has their tag).
        to_header: Original INVITE To header (us, may not have tag).
        local_tag: Our tag from the 200 OK.
        local_ip: Our IP for the Via header.
        local_sip_port: Our SIP port for the Via header.

    Returns:
        Encoded BYE request bytes.
    """
    remote_uri = _extract_uri(from_header)
    local_from = _append_tag(to_header, local_tag)
    request = (
        f"BYE {remote_uri} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{local_sip_port};branch=z9hG4bK{call_id[:8]}\r\n"
        f"From: {local_from}\r\n"
        f"To: {from_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 BYE\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return request.encode("utf-8")
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sip_signaling.py -v
```

Expected: All 10 tests PASS.

**Step 5: Commit**

```bash
git add src/pipecat/transports/sip/signaling.py tests/test_sip_signaling.py
git commit -m "feat(sip): add SIP message parsing and response building"
```

---

### Task 8: Transport Classes

**Files:**
- Create: `src/pipecat/transports/sip/transport.py`

This is the core integration task. It creates `SIPInputTransport`, `SIPOutputTransport`, `SIPCallTransport`, and `SIPServerTransport`.

**Step 1: Write the transport module**

```python
"""SIP/RTP transport for Pipecat.

Provides SIPServerTransport (SIP listener + port manager) that produces
SIPCallTransport instances (BaseTransport) per incoming call. Each call
gets its own input/output processors for pipeline integration.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport

from pipecat.transports.sip.codecs import G711Codec, resample_down, resample_up
from pipecat.transports.sip.params import SIPParams
from pipecat.transports.sip.rtp import RTPSession
from pipecat.transports.sip.sdp import parse_sdp
from pipecat.transports.sip.signaling import (
    SIPMessage,
    SIPMethod,
    build_100_trying,
    build_200_ok,
    build_200_ok_bye,
    build_bye,
)

logger = logging.getLogger(__name__)


@dataclass
class SIPSession:
    """Per-call SIP session state.

    Parameters:
        call_id: SIP Call-ID header value.
        local_tag: Our tag from the 200 OK response.
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

    def __post_init__(self):
        self.rtp_session = RTPSession(local_port=self.local_rtp_port)
        self.stopped_event = asyncio.Event()
        self._stopped = False
        self._bye_sent = False

    def set_sip_transport(
        self, transport: asyncio.DatagramTransport, addr: Tuple[str, int]
    ):
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


class SIPInputTransport(BaseInputTransport):
    """Pulls PCM16 from RTP rx_queue, resamples 8k->16k, pushes to pipeline.

    Args:
        session: The SIPSession for this call.
        params: Transport parameters.
    """

    def __init__(self, session: SIPSession, params: SIPParams, **kwargs):
        super().__init__(params=params, **kwargs)
        self._session = session
        self._rx_task: Optional[asyncio.Task] = None
        self._dtmf_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._rx_task = self.create_task(self._rx_loop(), "sip_rx_loop")
        if self._params.dtmf_enabled:
            self._dtmf_task = self.create_task(self._dtmf_loop(), "sip_dtmf_loop")

    async def stop(self, frame: EndFrame):
        if self._rx_task:
            await self.cancel_task(self._rx_task)
        if self._dtmf_task:
            await self.cancel_task(self._dtmf_task)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
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

    async def _dtmf_loop(self):
        """Pull DTMF digits from RTP session and push as transport messages."""
        while True:
            try:
                digit = await self._session.rtp_session.dtmf_queue.get()
                frame = InputTransportMessageFrame(
                    message={"type": "dtmf", "digit": digit}
                )
                await self.push_frame(frame)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("SIP DTMF error: %s", e)


class SIPOutputTransport(BaseOutputTransport):
    """Receives pipeline audio, resamples to 8k, encodes G.711, sends via RTP.

    Args:
        session: The SIPSession for this call.
        params: Transport parameters.
    """

    def __init__(self, session: SIPSession, params: SIPParams, **kwargs):
        super().__init__(params=params, **kwargs)
        self._session = session
        self._codec = G711Codec.get_instance()
        self._rtp_buffer = bytearray()

    async def start(self, frame: StartFrame):
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
            # Generic resample for other rates
            logger.warning("Unexpected sample rate %d, skipping", sample_rate)
            return True

        self._rtp_buffer.extend(pcm_8k.tobytes())

        # Slice into 320-byte (160 sample) aligned frames
        while len(self._rtp_buffer) >= 320:
            chunk = bytes(self._rtp_buffer[:320])
            del self._rtp_buffer[:320]
            await self._session.rtp_session.tx_queue.put(chunk)

        return True


class SIPCallTransport(BaseTransport):
    """Per-call transport wrapping SIP session with input/output processors.

    Created by SIPServerTransport for each incoming call. Provides standard
    BaseTransport interface with input()/output() for pipeline integration.

    Args:
        session: The SIPSession for this call.
        params: Transport parameters.
    """

    def __init__(self, session: SIPSession, params: SIPParams, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._params = params
        self._input: Optional[SIPInputTransport] = None
        self._output: Optional[SIPOutputTransport] = None

    @property
    def session(self) -> SIPSession:
        """Access the SIP session state (call_id, headers, etc.)."""
        return self._session

    def input(self) -> SIPInputTransport:
        if not self._input:
            self._input = SIPInputTransport(
                self._session, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> SIPOutputTransport:
        if not self._output:
            self._output = SIPOutputTransport(
                self._session, self._params, name=self._output_name
            )
        return self._output

    async def hangup(self):
        """Initiate a UAS-side hangup (send SIP BYE, stop RTP)."""
        await self._session.stop()


class _SIPServerProtocol(asyncio.DatagramProtocol):
    """Asyncio UDP protocol for SIP signaling."""

    def __init__(self, server: SIPServerTransport):
        self._server = server

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        self._server._handle_message(data, addr)

    def error_received(self, exc: Exception):
        logger.error("SIP protocol error: %s", exc)


class SIPServerTransport:
    """SIP UAS server that listens for incoming calls.

    Manages a UDP SIP listener and RTP port pool. Each incoming INVITE
    creates a SIPCallTransport and fires the on_call_started event.

    Events:
        on_call_started(server, call_transport): Fired when a new call is ready.
        on_call_ended(server, call_transport): Fired when a call ends.
        on_call_failed(server, call_transport, error): Fired on call error.

    Args:
        params: SIP transport parameters.

    Example::

        server = SIPServerTransport(params=SIPParams())

        @server.event_handler("on_call_started")
        async def on_call(server, call_transport):
            pipeline = Pipeline([
                call_transport.input(), stt, llm, tts, call_transport.output()
            ])
            task = PipelineTask(pipeline)
            await runner.run(task)

        await server.start()
    """

    def __init__(self, params: SIPParams):
        self._params = params
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._active_calls: Dict[str, Tuple[SIPSession, SIPCallTransport]] = {}
        self._pending_acks: Dict[str, asyncio.Task] = {}
        self._used_ports: Set[int] = set()
        self._running = False
        self._local_port = 0
        self._event_handlers: Dict[str, Any] = {}

    def event_handler(self, event_name: str):
        """Decorator to register an event handler.

        Args:
            event_name: Event name (on_call_started, on_call_ended, on_call_failed).
        """
        def decorator(func):
            self._event_handlers[event_name] = func
            return func
        return decorator

    async def _call_event_handler(self, event_name: str, *args):
        """Call a registered event handler."""
        handler = self._event_handlers.get(event_name)
        if handler:
            try:
                result = handler(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Event handler %s error: %s", event_name, e)

    async def start(self):
        """Start the SIP UDP listener."""
        self._running = True
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _SIPServerProtocol(self),
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

    async def stop(self):
        """Stop the SIP server and all active calls."""
        self._running = False
        # Cancel pending ACK timers
        for task in self._pending_acks.values():
            task.cancel()
        self._pending_acks.clear()
        # Stop all active calls
        for session, call_transport in list(self._active_calls.values()):
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
        # Random first
        for _ in range(100):
            port = random.randint(lo, hi)
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        # Linear fallback
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

        # Send 100 Trying immediately
        trying = build_100_trying(invite=msg)
        self._transport.sendto(trying, addr)

        # Parse remote SDP
        remote_ip, remote_port, codecs = parse_sdp(msg.body or "")
        if not remote_ip or not remote_port:
            logger.error("Invalid SDP in INVITE for call %s", msg.call_id)
            return

        # Negotiate codec
        negotiated_codec = "PCMU"
        for pref in self._params.codec_preferences:
            for pt, name in codecs.items():
                if name == pref:
                    negotiated_codec = name
                    break
            else:
                continue
            break

        # Allocate RTP port
        try:
            local_rtp_port = self._allocate_rtp_port()
        except RuntimeError:
            logger.error("No RTP ports available for call %s", msg.call_id)
            return

        # Determine local IP
        local_ip = self._transport.get_extra_info("sockname")[0]
        if local_ip == "0.0.0.0":
            local_ip = "127.0.0.1"

        session_id = random.randint(1, 0xFFFFFFFF)
        local_tag = f"bot-{session_id}"

        # Create session
        session = SIPSession(
            call_id=msg.call_id,
            local_tag=local_tag,
            from_header=msg.from_header,
            to_header=msg.to_header,
            via_header=msg.via,
            cseq=msg.cseq,
            remote_rtp_addr=(remote_ip, remote_port),
            local_rtp_port=local_rtp_port,
            local_ip=local_ip,
            local_sip_port=self._local_port,
            codec=negotiated_codec,
        )
        session.set_sip_transport(self._transport, addr)

        # Send 200 OK with SDP
        ok = build_200_ok(
            invite=msg,
            local_ip=local_ip,
            local_port=local_rtp_port,
            session_id=session_id,
        )
        self._transport.sendto(ok, addr)

        # Create transport
        call_transport = SIPCallTransport(session=session, params=self._params)

        # Store and wait for ACK
        self._active_calls[msg.call_id] = (session, call_transport)

        # Start ACK timeout
        ack_task = asyncio.get_event_loop().create_task(
            self._ack_timeout(msg.call_id)
        )
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

        # Cancel ACK timeout
        ack_task = self._pending_acks.pop(call_id, None)
        if ack_task:
            ack_task.cancel()

        entry = self._active_calls.get(call_id)
        if not entry:
            return
        session, call_transport = entry

        # Start RTP and fire event
        asyncio.get_event_loop().create_task(
            self._start_call(session, call_transport)
        )

    async def _start_call(self, session: SIPSession, call_transport: SIPCallTransport):
        """Start RTP session and fire on_call_started event."""
        try:
            await session.start_rtp()
            # Start RTP send loop in background
            asyncio.get_event_loop().create_task(self._run_rtp(session))
            await self._call_event_handler(
                "on_call_started", self, call_transport
            )
        except Exception as e:
            logger.error("Call start error: %s", e)
            await self._call_event_handler(
                "on_call_failed", self, call_transport, e
            )
            await self._cleanup_call(session.call_id)

    async def _run_rtp(self, session: SIPSession):
        """Run RTP send loop, cleanup on completion."""
        try:
            await session.rtp_session.run()
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

    def _handle_bye(self, msg: SIPMessage, addr: Tuple[str, int]):
        """Handle BYE: respond 200 OK, teardown call."""
        ok = build_200_ok_bye(bye=msg)
        self._transport.sendto(ok, addr)

        entry = self._active_calls.get(msg.call_id)
        if entry:
            session, call_transport = entry
            asyncio.get_event_loop().create_task(
                self._on_bye_received(session, call_transport)
            )
        else:
            logger.debug("BYE for unknown call %s", msg.call_id)

    async def _on_bye_received(
        self, session: SIPSession, call_transport: SIPCallTransport
    ):
        """Handle BYE received: stop session, fire event, cleanup."""
        await session.stop()
        await self._call_event_handler("on_call_ended", self, call_transport)
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
```

**Step 2: Run lint check**

```bash
uv run ruff check src/pipecat/transports/sip/transport.py
```

**Step 3: Commit**

```bash
git add src/pipecat/transports/sip/transport.py
git commit -m "feat(sip): add SIP transport classes (server, call, input, output)"
```

---

### Task 9: Module Exports

**Files:**
- Modify: `src/pipecat/transports/sip/__init__.py`

**Step 1: Update `__init__.py` with exports**

```python
"""SIP/RTP transport for Pipecat.

Provides SIPServerTransport for accepting incoming SIP calls and routing
them through Pipecat pipelines with G.711 audio over RTP.
"""

from pipecat.transports.sip.params import SIPParams
from pipecat.transports.sip.transport import (
    SIPCallTransport,
    SIPServerTransport,
    SIPSession,
)

__all__ = [
    "SIPCallTransport",
    "SIPParams",
    "SIPServerTransport",
    "SIPSession",
]
```

**Step 2: Verify imports work**

```bash
uv run python -c "from pipecat.transports.sip import SIPServerTransport, SIPCallTransport, SIPParams, SIPSession; print('Imports OK')"
```

**Step 3: Commit**

```bash
git add src/pipecat/transports/sip/__init__.py
git commit -m "feat(sip): add module exports"
```

---

### Task 10: Example

**Files:**
- Create: `examples/foundational/55zzo-sip-transport.py`

**Step 1: Write example**

```python
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SIP Transport Example - Echo Bot

Accepts incoming SIP calls and echoes received audio back to the caller.
Useful for testing SIP connectivity and verifying audio round-trip.

Requirements:
    pip install "pipecat-ai[sip]"

Testing with pjsua (CLI SIP client):
    pjsua --null-audio sip:bot@<HOST>:5060

Testing with Linphone:
    1. Open Linphone
    2. Call sip:bot@<HOST>:5060

Environment variables:
    SIP_HOST: Bind address (default: 0.0.0.0)
    SIP_PORT: SIP port (default: 5060)
"""

import asyncio
import logging
import os
import sys

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.sip import SIPCallTransport, SIPParams, SIPServerTransport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    sip_host = os.getenv("SIP_HOST", "0.0.0.0")
    sip_port = int(os.getenv("SIP_PORT", "5060"))

    params = SIPParams(
        sip_listen_host=sip_host,
        sip_listen_port=sip_port,
        audio_in_enabled=True,
        audio_out_enabled=True,
    )

    server = SIPServerTransport(params=params)

    @server.event_handler("on_call_started")
    async def on_call(server, call_transport: SIPCallTransport):
        logger.info("Call started: %s", call_transport.session.call_id)

        # Echo pipeline: input -> output (echoes audio back)
        pipeline = Pipeline([call_transport.input(), call_transport.output()])

        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        await runner.run(task)

        logger.info("Call ended: %s", call_transport.session.call_id)

    @server.event_handler("on_call_ended")
    async def on_call_ended(server, call_transport: SIPCallTransport):
        logger.info("Call ended (BYE): %s", call_transport.session.call_id)

    await server.start()
    logger.info("SIP echo bot listening on %s:%d", sip_host, sip_port)

    # Run until interrupted
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add examples/foundational/55zzo-sip-transport.py
git commit -m "feat(sip): add SIP echo bot example"
```

---

### Task 11: Run Full Test Suite and Lint

**Step 1: Run all SIP tests**

```bash
uv run pytest tests/test_sip_codecs.py tests/test_sip_resample.py tests/test_sip_sdp.py tests/test_sip_rtp.py tests/test_sip_signaling.py -v
```

Expected: All tests PASS.

**Step 2: Run ruff lint**

```bash
uv run ruff check src/pipecat/transports/sip/
```

Fix any issues found.

**Step 3: Run ruff format check**

```bash
uv run ruff format --check src/pipecat/transports/sip/
```

Fix any formatting issues.

**Step 4: Run the full pipecat test suite to check for regressions**

```bash
uv run pytest tests/ -x --timeout=60
```

Expected: No regressions.

**Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix(sip): address lint and formatting issues"
```

---

### Task 12: Add Towncrier Changelog Fragment

**Files:**
- Create: `changes/XXXX.feature.md` (use next available number)

**Step 1: Check existing changelog fragments**

```bash
ls changes/
```

**Step 2: Create changelog fragment**

```markdown
Add SIP/RTP transport (UAS) for accepting incoming SIP calls. Supports G.711 PCMU/PCMA codecs, 20ms RTP packetization, RFC 2833 DTMF detection, and per-call pipeline integration via `SIPServerTransport` and `SIPCallTransport`.
```

**Step 3: Commit**

```bash
git add changes/
git commit -m "docs(sip): add changelog fragment for SIP transport"
```

---

### Task 13: Final Verification and PR Prep

**Step 1: Verify all tests pass**

```bash
uv run pytest tests/test_sip_*.py -v
```

**Step 2: Verify imports**

```bash
uv run python -c "
from pipecat.transports.sip import SIPServerTransport, SIPCallTransport, SIPParams, SIPSession
from pipecat.transports.sip.codecs import G711Codec, resample_up, resample_down
from pipecat.transports.sip.sdp import parse_sdp, generate_sdp
from pipecat.transports.sip.rtp import RTPSession, pack_rtp_header, unpack_rtp_header, parse_dtmf_event
from pipecat.transports.sip.signaling import SIPMessage, SIPMethod, build_100_trying, build_200_ok, build_200_ok_bye, build_bye
print('All imports OK')
"
```

**Step 3: Review git log**

```bash
git log --oneline main..HEAD
```

**Step 4: Create PR**

```bash
gh pr create --title "feat(transports): add SIP/RTP transport (UAS)" --body "$(cat <<'EOF'
## Summary

Adds a SIP UAS transport to Pipecat enabling PBX/SBC SIP calls to enter pipelines directly.

- `SIPServerTransport` — UDP SIP listener with per-call session management and RTP port pool
- `SIPCallTransport` — per-call `BaseTransport` with `input()`/`output()` processors
- G.711 PCMU/PCMA codecs via numpy LUT tables
- 20ms RTP packetization with drift-corrected timing and pre-buffering
- RFC 2833 DTMF digit detection
- Integer-ratio resampling (8kHz ↔ 16kHz)

## Architecture

```
SIPServerTransport (SIP listener)
  └─ per INVITE → SIPCallTransport (BaseTransport)
       ├─ input()  → SIPInputTransport (BaseInputTransport)
       └─ output() → SIPOutputTransport (BaseOutputTransport)
            └─ RTPSession (UDP send/recv, 20ms timing)
```

## Supported Codecs

| Codec | PT | Rate |
|-------|-----|------|
| PCMU  | 0   | 8000 |
| PCMA  | 8   | 8000 |

## Local Verification

```bash
# Run tests
uv run pytest tests/test_sip_*.py -v

# Run echo bot example
uv run python examples/foundational/55zzo-sip-transport.py

# Test with pjsua
pjsua --null-audio sip:bot@localhost:5060
```

## Test Instructions

```bash
uv run pytest tests/test_sip_codecs.py tests/test_sip_resample.py tests/test_sip_sdp.py tests/test_sip_rtp.py tests/test_sip_signaling.py -v
```

## Future Roadmap

- SRTP / TLS transport security
- SIP CANCEL, REGISTER, REFER methods
- Opus codec support
- Full jitter buffer with packet reordering
- Echo cancellation (AEC) integration
- SIP digest authentication

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
            parts = line.split()
            if len(parts) >= 3:
                ip = parts[-1]
        elif line.startswith("m=audio"):
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
        f"m=audio {local_port} RTP/AVP 0\r\n"
        f"a=rtpmap:0 PCMU/8000\r\n"
        f"a=sendrecv\r\n"
    )

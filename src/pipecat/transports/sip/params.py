#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FreeSWITCH SIP transport parameters."""

from __future__ import annotations

from typing import List, Tuple

from pipecat.transports.base_transport import TransportParams


class FreeSwitchSIPParams(TransportParams):
    """Parameters for FreeSWITCH SIP/RTP transport.

    Extends TransportParams with SIP-specific configuration for signaling,
    RTP media, and codec settings. Scoped for LAN use with FreeSWITCH
    (no NAT/STUN, no REGISTER, dial-in only).

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
    codec_preferences: List[str] = ["PCMU"]
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

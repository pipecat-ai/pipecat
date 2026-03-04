#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FreeSWITCH SIP/RTP transport for Pipecat.

Provides FreeSwitchSIPServerTransport for accepting incoming SIP calls and
routing them through Pipecat pipelines with G.711 audio over RTP. Scoped for
LAN use with FreeSWITCH (no NAT/STUN, no REGISTER, dial-in only).
"""

from pipecat.transports.sip.params import FreeSwitchSIPParams
from pipecat.transports.sip.transport import (
    FreeSwitchSIPCallTransport,
    FreeSwitchSIPServerTransport,
    FreeSwitchSIPSession,
)

__all__ = [
    "FreeSwitchSIPCallTransport",
    "FreeSwitchSIPParams",
    "FreeSwitchSIPServerTransport",
    "FreeSwitchSIPSession",
]

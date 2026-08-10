#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent Client Protocol support.

The Agent Client Protocol (https://agentclientprotocol.com) is what code editors
speak to coding agents. :class:`~pipecat.services.acp.service.ACPService` runs an
agent as a subprocess and bridges it to a Pipecat pipeline, so a bot can drive
one the way an editor would.

The service is protocol plumbing only. It emits ACP frames and no speakable
text; see :mod:`pipecat.services.acp.renderers` for the seam where a voice
client decides what the agent's output should sound like.
"""

from pipecat.services.acp.aggregator import ACPUserAggregator
from pipecat.services.acp.client import ACPClient, ACPError
from pipecat.services.acp.permissions import ACPAutoPermission
from pipecat.services.acp.renderers import describe
from pipecat.services.acp.service import ACPService

__all__ = [
    "ACPAutoPermission",
    "ACPClient",
    "ACPError",
    "ACPService",
    "ACPUserAggregator",
    "describe",
]

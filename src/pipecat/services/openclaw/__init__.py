#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenClaw Gateway support.

The Gateway is the websocket an OpenClaw agent publishes for other programs to
drive it. :class:`~pipecat.services.openclaw.client.OpenClawGatewayClient`
starts runs on it, streams their output, redirects work already in flight, and
stops it. :class:`~pipecat.services.openclaw.processor.OpenClawGatewayProcessor`
is the pipeline half, translating between that traffic and frames.

The split leaves the decision of what an agent's output should sound like to a
service wrapping the processor.
"""

from pipecat.services.openclaw.client import (
    DEFAULT_GATEWAY_URL,
    DEFAULT_SESSION_KEY,
    OpenClawError,
    OpenClawEvent,
    OpenClawGatewayClient,
    OpenClawResult,
    OpenClawRun,
    collect_result,
)
from pipecat.services.openclaw.frames import (
    OpenClawAbortFrame,
    OpenClawRunCancelledFrame,
    OpenClawRunCompletedFrame,
    OpenClawRunFailedFrame,
    OpenClawRunStartedFrame,
    OpenClawSendFrame,
    OpenClawSteerFrame,
    OpenClawTextFrame,
)
from pipecat.services.openclaw.processor import OpenClawGatewayProcessor

__all__ = [
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_SESSION_KEY",
    "OpenClawAbortFrame",
    "OpenClawError",
    "OpenClawEvent",
    "OpenClawGatewayClient",
    "OpenClawGatewayProcessor",
    "OpenClawResult",
    "OpenClawRun",
    "OpenClawRunCancelledFrame",
    "OpenClawRunCompletedFrame",
    "OpenClawRunFailedFrame",
    "OpenClawRunStartedFrame",
    "OpenClawSendFrame",
    "OpenClawSteerFrame",
    "OpenClawTextFrame",
    "collect_result",
]

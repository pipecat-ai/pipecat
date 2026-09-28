#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenClaw Gateway support.

The Gateway is the websocket an OpenClaw agent publishes for other programs to
drive it. :class:`~pipecat.services.openclaw.client.OpenClawGatewayClient`
starts runs on it, streams their output, redirects work already in flight, and
stops it. :class:`~pipecat.services.openclaw.gateway.OpenClawGatewayService`
is the pipeline half, translating between that traffic and frames.

The split leaves the decision of what an agent's output should sound like to a
service wrapping it.
"""

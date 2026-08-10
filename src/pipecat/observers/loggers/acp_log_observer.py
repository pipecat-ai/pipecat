#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ACP logging observer for Pipecat."""

from loguru import logger

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.services.acp.renderers import describe
from pipecat.services.acp.service import ACPService


class ACPLogObserver(BaseObserver):
    """Observer to log an ACP agent's activity to the console.

    Logs one line per ACP frame: session and turn boundaries, agent messages
    and reasoning, tool calls and their updates, plans, and the agent's requests
    to the client. Non-ACP frames are ignored.

    Useful while deciding what a renderer should say. It shows the full stream a
    real agent produces without any of it reaching a TTS service.

    Example::

        worker = PipelineWorker(pipeline, observers=[ACPLogObserver()])
    """

    async def on_push_frame(self, data: FramePushed):
        """Log an ACP frame as it is pushed.

        Args:
            data: The frame push event.
        """
        # Every processor the frame passes through pushes it again. Anchoring on
        # the service logs each frame once, on the hop where it enters or leaves.
        if not isinstance(data.source, ACPService) and not isinstance(data.destination, ACPService):
            return

        if summary := describe(data.frame):
            logger.info(f"ACP | {summary}")

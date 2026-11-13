#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""In-process worker bus backed by asyncio queues."""

from loguru import logger

from pipecat.bus.bus import WorkerBus
from pipecat.bus.messages import BusMessage


class AsyncQueueBus(WorkerBus):
    """In-process bus that delivers messages via priority queues."""

    async def publish(self, message: BusMessage) -> None:
        """Deliver a message to all local subscriber queues.

        Args:
            message: The bus message to deliver.
        """
        logger.trace(f"{self}: sending {message}")
        self.on_message_received(message)

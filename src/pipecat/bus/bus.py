#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract worker bus for inter-worker pub/sub messaging.

Provides the abstract `WorkerBus` base class. Concrete implementations
(e.g. `AsyncQueueBus`) live in separate modules.
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import cast

from loguru import logger

from pipecat.bus.messages import BusLocalMessage, BusMessage, BusSystemMessage
from pipecat.bus.queue import BusMessageQueue
from pipecat.bus.subscriber import BusSubscriber
from pipecat.utils.base_object import BaseObject


@dataclass
class BusSubscription:
    """A single subscriber's state on the bus.

    Parameters:
        subscriber: The subscriber receiving messages.
        queue: Priority queue for incoming messages.
        data_queue: Secondary queue for data messages dispatched by
            the router worker.
        router_task: Task that reads from the priority queue, handles
            system messages inline, and routes data messages to the
            data queue.
        data_task: Task that processes data messages sequentially from
            the data queue.
    """

    subscriber: BusSubscriber
    queue: BusMessageQueue = field(default_factory=BusMessageQueue, repr=False)
    data_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    router_task: asyncio.Task | None = field(default=None, repr=False)
    data_task: asyncio.Task | None = field(default=None, repr=False)


class WorkerBus(BaseObject):
    """Abstract base for inter-worker and runner-worker communication.

    Provides pub/sub messaging where each subscriber receives messages
    independently through its own priority queue. System messages
    (e.g. cancel) are delivered before normal data messages.

    Subclasses implement ``publish()`` for the specific transport.
    ``send()`` handles local-only messages automatically. For network
    buses, override ``start()``/``stop()`` to manage connections and
    call ``on_message_received()`` when messages arrive from the
    network.
    """

    def __init__(self, **kwargs):
        """Initialize the WorkerBus.

        Args:
            **kwargs: Additional arguments passed to `BaseObject`.
        """
        super().__init__(**kwargs)
        self._subscriptions: dict[str, BusSubscription] = {}
        self._running = False

    async def start(self):
        """Start dispatch tasks for all registered subscribers."""
        if self._running:
            return
        self._running = True
        for sub in self._subscriptions.values():
            self._start_dispatch_task(sub)
        # Schedule tasks right away.
        await asyncio.sleep(0)

    async def stop(self):
        """Stop all dispatch tasks."""
        if not self._running:
            return
        self._running = False
        for sub in self._subscriptions.values():
            if sub.router_task:
                await self.cancel_task(sub.router_task)
                sub.router_task = None
            if sub.data_task:
                await self.cancel_task(sub.data_task)
                sub.data_task = None

    async def subscribe(self, subscriber: BusSubscriber) -> None:
        """Register a subscriber to receive messages from the bus.

        Idempotent: re-subscribing an already-registered subscriber is
        a no-op.

        Args:
            subscriber: The `BusSubscriber` to register.
        """
        if subscriber.name in self._subscriptions:
            return
        sub = BusSubscription(subscriber=subscriber)
        if self._running:
            self._start_dispatch_task(sub)
            # Schedule worker right away.
            await asyncio.sleep(0)
        self._subscriptions[subscriber.name] = sub

    async def unsubscribe(self, subscriber: BusSubscriber) -> None:
        """Remove a subscriber and cancel its dispatch tasks.

        Args:
            subscriber: The `BusSubscriber` to remove.
        """
        sub = self._subscriptions.pop(subscriber.name, None)
        if sub:
            if sub.router_task:
                await self.cancel_task(sub.router_task)
            if sub.data_task:
                await self.cancel_task(sub.data_task)

    async def send(self, message: BusMessage) -> None:
        """Send a message through the bus.

        Local-only messages are delivered directly to subscribers.
        All other messages are passed to ``publish()`` for transport.

        Args:
            message: The bus message to send.
        """
        if isinstance(message, BusLocalMessage):
            self.on_message_received(message)
            return
        await self.publish(message)

    @abstractmethod
    async def publish(self, message: BusMessage) -> None:
        """Publish a message to the transport.

        Subclasses implement this for the specific transport. Called
        by ``send()`` after filtering local-only messages.

        Args:
            message: The bus message to publish.
        """
        pass

    def on_message_received(self, message: BusMessage) -> None:
        """Deliver a message to all local subscribers via their priority queues.

        Called by bus implementations when a message arrives (either from
        a local ``send()`` or from a network transport).
        """
        for sub in self._subscriptions.values():
            sub.queue.put_nowait(message)

    def _start_dispatch_task(self, sub: BusSubscription) -> None:
        """Start the router and data dispatch tasks for a subscriber."""
        sub.router_task = self.create_task(self._router_task(sub), f"bus_router_{sub.subscriber}")
        sub.data_task = self.create_task(
            self._data_dispatch_task(sub), f"bus_data_{sub.subscriber}"
        )

    async def _router_task(self, sub: BusSubscription):
        """Route system messages inline, data messages to the data queue."""
        while True:
            message = await sub.queue.get()
            if isinstance(message, BusSystemMessage):
                # A subscriber exception must not tear down the dispatch task,
                # or the subscriber would silently stop receiving all future
                # messages (including cancel/cleanup). Log and keep going.
                try:
                    await sub.subscriber.on_bus_message(cast(BusMessage, message))
                except Exception:
                    logger.exception(
                        f"{self}: subscriber '{sub.subscriber.name}' raised handling "
                        f"system message {message}; keeping dispatch alive"
                    )
            else:
                sub.data_queue.put_nowait(message)

    async def _data_dispatch_task(self, sub: BusSubscription):
        """Process data messages sequentially from the data queue."""
        while True:
            message = await sub.data_queue.get()
            # A subscriber exception must not tear down the dispatch task,
            # or the subscriber would silently stop receiving all future
            # messages (including cancel/cleanup). Log and keep going.
            try:
                await sub.subscriber.on_bus_message(message)
            except Exception:
                logger.exception(
                    f"{self}: subscriber '{sub.subscriber.name}' raised handling "
                    f"data message {message}; keeping dispatch alive"
                )

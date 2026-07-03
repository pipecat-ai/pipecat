#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""PGMQ (PostgreSQL Message Queue) worker bus for distributed workers."""

import asyncio
import json

from loguru import logger

from pipecat.bus.bus import WorkerBus
from pipecat.bus.messages import BusMessage
from pipecat.bus.network.pgmq_backends import (
    DirectPgmqBackend,
    PgmqBackend,
)
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer

try:
    from pgmq.async_queue import PGMQueue
except ModuleNotFoundError as e:  # pragma: no cover - exercised only when extra is missing
    logger.error(f"Exception: {e}")
    logger.error('In order to use PgmqBus, you need to `uv add "pipecat-ai[pgmq]"`.')
    raise ImportError(f"Missing module: {e}") from e


class PgmqBus(WorkerBus):
    """Distributed worker bus backed by PGMQ (PostgreSQL Message Queue).

    Pub/sub fan-out is implemented on top of PGMQ's point-to-point queue
    semantics by giving each :class:`PgmqBus` instance its own queue and
    broadcasting on publish. A reader long-polls the instance's queue and
    dispatches received messages to local subscribers.

    ``BusLocalMessage`` messages bypass the network entirely and are
    delivered directly to local subscribers.

    Two backends are supported (see :mod:`pipecat.bus.network.pgmq_backends`):

    - :class:`DirectPgmqBackend` — calls ``pgmq.*`` directly and discovers
      peers by queue-name prefix. Suitable when bus peers trust each other.
    - :class:`IsolatedPgmqBackend` — calls SECURITY DEFINER Postgres
      wrappers over an asyncpg pool. Suitable when peers should be isolated
      and the channel name is the bus capability.

    Construct with either ``pgmq=PGMQueue`` (uses :class:`DirectPgmqBackend`)
    or ``backend=PgmqBackend`` (any backend). The two are mutually exclusive.

    Requires the ``pgmq`` extra. Install with
    ``uv add "pipecat-ai[pgmq]"``.

    Example::

        from pgmq.async_queue import PGMQueue

        pgmq = PGMQueue(
            host="...",
            port="5432",
            database="postgres",
            username="postgres",
            password="...",
            pool_size=4,
        )
        await pgmq.init()
        bus = PgmqBus(pgmq=pgmq, channel="pipecat_acme")

    Notes:
        Prefer a session-mode pooler when available. Transaction-mode
        pooling works for direct ``pgmq.*`` calls but is fragile around the
        long-poll inside the SECURITY DEFINER ``bus_subscribe`` wrapper used
        by :class:`IsolatedPgmqBackend`.
        The underlying connection pool must allow at least two concurrent
        connections (one for the reader's long-poll, one for publishes).
    """

    def __init__(
        self,
        *,
        pgmq: PGMQueue | None = None,
        backend: PgmqBackend | None = None,
        serializer: MessageSerializer | None = None,
        channel: str = "pipecat_bus",
        visibility_timeout: int = 30,
        batch_size: int = 10,
        poll_interval_ms: int = 100,
        max_poll_seconds: int = 5,
        **kwargs,
    ):
        """Initialize the PgmqBus.

        Args:
            pgmq: An initialized ``PGMQueue`` client. Selects
                :class:`DirectPgmqBackend`. Mutually exclusive with
                ``backend``.
            backend: A :class:`PgmqBackend` instance (e.g.
                :class:`IsolatedPgmqBackend`, or a custom backend).
                Mutually exclusive with ``pgmq``.
            serializer: The :class:`MessageSerializer` for encoding/decoding
                messages. Defaults to :class:`JSONMessageSerializer`.
            channel: Channel name. With :class:`DirectPgmqBackend` this is
                sanitized into a queue-name prefix. With
                :class:`IsolatedPgmqBackend` it is the bus capability passed
                to every wrapper call.
            visibility_timeout: Seconds a read message stays invisible
                before redelivery. Defaults to 30.
            batch_size: Maximum messages to fetch per read. Defaults to 10.
            poll_interval_ms: Long-poll check interval in milliseconds.
                Defaults to 100. (Backend may ignore if it doesn't expose
                this knob.)
            max_poll_seconds: Maximum seconds the reader blocks per poll
                cycle. Defaults to 5.
            **kwargs: Additional arguments passed to :class:`WorkerBus`.
        """
        super().__init__(**kwargs)
        if pgmq is not None and backend is not None:
            raise ValueError("PgmqBus accepts pgmq= or backend=, not both")
        if backend is not None:
            self._backend: PgmqBackend = backend
        elif pgmq is not None:
            self._backend = DirectPgmqBackend(pgmq)
        else:
            raise ValueError("PgmqBus requires pgmq= or backend=")

        self._serializer = serializer or JSONMessageSerializer()
        self._channel = channel
        self._visibility_timeout = visibility_timeout
        self._batch_size = batch_size
        self._poll_interval_ms = poll_interval_ms
        self._max_poll_seconds = max_poll_seconds
        self._queue_name: str | None = None
        self._reader_task: asyncio.Task | None = None

    async def start(self):
        """Join the channel via the backend and start the reader task."""
        await super().start()
        self._queue_name = await self._backend.join(self._channel)
        logger.debug(f"{self}: joined channel via backend; queue='{self._queue_name}'")
        self._reader_task = self.create_task(self._reader_loop())
        await asyncio.sleep(0)

    async def stop(self):
        """Stop the reader task and leave the channel."""
        await super().stop()
        if self._reader_task:
            await self.cancel_task(self._reader_task)
            self._reader_task = None
        if self._queue_name:
            try:
                await self._backend.leave(self._queue_name, channel=self._channel)
            except Exception:
                logger.exception(f"{self}: backend leave failed for queue '{self._queue_name}'")
            self._queue_name = None

    async def publish(self, message: BusMessage) -> None:
        """Broadcast a message to every peer on this channel.

        The backend handles peer discovery and fan-out; per-peer failures
        are the backend's responsibility to absorb so the publish does not
        raise.
        """
        payload_bytes = self._serializer.serialize(message)
        try:
            payload = json.loads(payload_bytes)
        except Exception:
            logger.exception(f"{self}: failed to encode payload for pgmq")
            return

        if self._queue_name is None:
            logger.warning(f"{self}: publish called before start(); dropping message")
            return

        try:
            await self._backend.publish(self._channel, self._queue_name, payload)
        except Exception:
            logger.exception(f"{self}: backend publish failed")

    async def _reader_loop(self) -> None:
        while True:
            if self._queue_name is None:
                # Defensive: start() always sets this before launching the
                # task, but cover the race where stop() clears it mid-loop.
                return
            try:
                messages = await self._backend.read(
                    self._queue_name,
                    channel=self._channel,
                    vt=self._visibility_timeout,
                    qty=self._batch_size,
                    max_poll_seconds=self._max_poll_seconds,
                    poll_interval_ms=self._poll_interval_ms,
                )
            except Exception:
                logger.exception(f"{self}: backend read failed; backing off")
                await asyncio.sleep(1)
                continue

            for msg in messages or []:
                try:
                    raw = json.dumps(msg.message).encode("utf-8")
                    bus_message = self._serializer.deserialize(raw)
                    if bus_message:
                        self.on_message_received(bus_message)
                except Exception:
                    logger.exception(f"{self}: failed to deserialize message {msg.msg_id}")
                finally:
                    try:
                        await self._backend.archive(
                            self._queue_name,
                            channel=self._channel,
                            msg_id=msg.msg_id,
                        )
                    except Exception:
                        logger.exception(f"{self}: failed to archive message {msg.msg_id}")

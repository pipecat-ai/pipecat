#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""PGMQ (PostgreSQL Message Queue) agent bus for distributed agents."""

import asyncio
import json
import re
import time
import uuid

from loguru import logger

from pipecat.bus.bus import TaskBus
from pipecat.bus.messages import BusMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer

try:
    from pgmq.async_queue import PGMQueue
except ModuleNotFoundError as e:  # pragma: no cover - exercised only when extra is missing
    logger.error(f"Exception: {e}")
    logger.error("In order to use PgmqBus, you need to `pip install pipecat-ai-subagents[pgmq]`.")
    raise Exception(f"Missing module: {e}")


_INVALID_CHANNEL_CHARS = re.compile(r"[^A-Za-z0-9_]")
_MAX_CHANNEL_LEN = 30
_PEER_LIST_TTL_S = 1.0


def _sanitize_channel(channel: str) -> str:
    """Coerce an arbitrary channel string into a Postgres-safe identifier prefix.

    PGMQ queue names must match ``^[a-zA-Z_][a-zA-Z0-9_]*$`` and are bounded by
    Postgres identifier limits (effective ~48 chars after PGMQ's internal
    ``q_``/``a_`` table prefixes). This helper replaces invalid characters with
    underscores, ensures the prefix doesn't start with a digit, and truncates
    to leave headroom for an instance suffix.

    Args:
        channel: User-supplied channel name (may contain colons, slashes, etc).

    Returns:
        A sanitized prefix safe for use in PGMQ queue names.
    """
    safe = _INVALID_CHANNEL_CHARS.sub("_", channel)
    if safe and safe[0].isdigit():
        safe = f"q_{safe}"
    return safe[:_MAX_CHANNEL_LEN] or "pipecat_bus"


class PgmqBus(TaskBus):
    """Distributed agent bus backed by PGMQ (PostgreSQL Message Queue).

    Implements pub/sub fan-out on top of PGMQ's point-to-point queue semantics
    by giving each ``PgmqBus`` instance its own queue and broadcasting on
    publish. The reader long-polls its own queue and dispatches received
    messages to local subscribers.

    ``BusLocalMessage`` messages bypass PGMQ and are delivered directly to
    local subscribers.

    Requires the ``pgmq`` and ``asyncpg`` packages. Install with
    ``pip install pipecat-ai-subagents[pgmq]``.

    The provided ``PGMQueue`` must already have its connection pool
    initialized via ``await pgmq.init()`` before being passed. The adapter
    does not own the client's lifetime and will not close it on stop.

    Notes:
        Prefer the session-mode pooler (e.g. port 5432 in Supabase) when
        one is available. Transaction-mode pooling (e.g. port 6543 in
        Supabase) works in practice with this adapter because each PGMQ
        call is a single SQL statement, but it logs benign "resetting
        connection with an active transaction" warnings and is more
        fragile around prepared statements.
        The underlying pool must allow at least 2 concurrent connections
        (one for the reader's long-poll, one for publishes); 4+ recommended
        under load.

    Example::

        from pgmq.async_queue import PGMQueue

        pgmq = PGMQueue(
            host="aws-0-us-east-1.pooler.supabase.com",
            port="5432",
            database="postgres",
            username="postgres.<project-ref>",
            password="...",
            pool_size=4,
        )
        await pgmq.init()
        bus = PgmqBus(pgmq=pgmq, channel="pipecat_acme")
    """

    def __init__(
        self,
        *,
        pgmq: PGMQueue,
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
            pgmq: An initialized ``PGMQueue`` client. Call ``await pgmq.init()``
                before passing.
            serializer: The `MessageSerializer` for encoding/decoding messages.
                Defaults to `JSONMessageSerializer`.
            channel: Channel prefix for queue names. Sanitized to alphanumeric
                + underscore. Defaults to ``"pipecat_bus"``.
            visibility_timeout: Seconds a read message stays invisible before
                redelivery. Defaults to 30.
            batch_size: Maximum messages to fetch per read. Defaults to 10.
            poll_interval_ms: Long-poll check interval in milliseconds.
                Defaults to 100.
            max_poll_seconds: Maximum seconds the reader blocks per poll cycle.
                Defaults to 5.
            **kwargs: Additional arguments passed to `TaskBus`.
        """
        super().__init__(**kwargs)
        self._pgmq = pgmq
        self._serializer = serializer or JSONMessageSerializer()
        self._safe_channel = _sanitize_channel(channel)
        self._visibility_timeout = visibility_timeout
        self._batch_size = batch_size
        self._poll_interval_ms = poll_interval_ms
        self._max_poll_seconds = max_poll_seconds
        self._queue_name: str | None = None
        self._reader_task: asyncio.Task | None = None
        self._peer_cache: list[str] = []
        self._peer_cache_at: float = 0.0

    async def start(self):
        """Create this instance's queue and start the reader task."""
        await super().start()
        self._queue_name = f"{self._safe_channel}_{uuid.uuid4().hex[:12]}"
        await self._pgmq.create_queue(self._queue_name)
        logger.debug(f"{self}: created pgmq queue '{self._queue_name}'")
        self._reader_task = self.create_task(self._reader_loop(), f"{self}::pgmq_reader")
        await asyncio.sleep(0)

    async def stop(self):
        """Stop the reader task and drop this instance's queue."""
        await super().stop()
        if self._reader_task:
            await self.cancel_task(self._reader_task)
            self._reader_task = None
        if self._queue_name:
            try:
                await self._pgmq.drop_queue(self._queue_name)
            except Exception:
                logger.exception(f"{self}: failed to drop queue '{self._queue_name}'")
            self._queue_name = None

    async def publish(self, message: BusMessage) -> None:
        """Broadcast a message to every peer queue sharing the channel prefix.

        Args:
            message: The bus message to publish.
        """
        payload_bytes = self._serializer.serialize(message)
        try:
            payload = json.loads(payload_bytes)
        except Exception:
            logger.exception(f"{self}: failed to encode payload for pgmq")
            return

        peers = await self._peer_queues()
        logger.trace(f"{self}: publishing to {len(peers)} peer queue(s)")
        for queue_name in peers:
            try:
                await self._pgmq.send(queue_name, payload)
            except Exception:
                logger.warning(
                    f"{self}: send to peer queue '{queue_name}' failed; invalidating cache"
                )
                self._peer_cache_at = 0.0

    async def _peer_queues(self) -> list[str]:
        now = time.monotonic()
        if self._peer_cache and now - self._peer_cache_at < _PEER_LIST_TTL_S:
            return self._peer_cache
        names = await self._pgmq.list_queues()
        prefix = f"{self._safe_channel}_"
        self._peer_cache = [name for name in names if name.startswith(prefix)]
        self._peer_cache_at = now
        return self._peer_cache

    async def _reader_loop(self) -> None:
        while True:
            try:
                messages = await self._pgmq.read_with_poll(
                    queue=self._queue_name,
                    vt=self._visibility_timeout,
                    qty=self._batch_size,
                    max_poll_seconds=self._max_poll_seconds,
                    poll_interval_ms=self._poll_interval_ms,
                )
            except Exception:
                logger.exception(f"{self}: pgmq read failed; backing off")
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
                        await self._pgmq.delete(self._queue_name, msg.msg_id)
                    except Exception:
                        logger.exception(f"{self}: failed to delete message {msg.msg_id}")

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Backends for :class:`pipecat.bus.network.pgmq.PgmqBus`.

A backend owns the wire-side of bus operations: allocating a queue when a
bus instance joins a channel, fanning a published message out to channel
peers, long-polling for incoming messages, archiving them, and dropping
the queue when the bus stops. :class:`PgmqBus` is the orchestrator on top
of this surface.

Two backends ship in this module:

- :class:`DirectPgmqBackend` calls :class:`pgmq.async_queue.PGMQueue`
  directly. Peer discovery is via ``list_queues()`` filtered by a channel
  prefix. The channel name is encoded in the queue name and is therefore
  enumerable by any role that can read ``pg_class``. Choose this backend
  when bus peers trust each other (single-tenant deployments, internal
  services).
- :class:`IsolatedPgmqBackend` calls a small set of SECURITY DEFINER
  Postgres wrappers (``public.bus_join``, ``bus_publish``,
  ``bus_subscribe``, ``bus_archive``, ``bus_leave``) over an
  :mod:`asyncpg` pool. Queue names are server-allocated opaque UUIDs and
  a server-side peer registry replaces ``list_queues`` discovery. Choose
  this backend when bus peers should be isolated from each other and the
  channel name itself is the bus capability.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger

try:
    from pgmq.async_queue import PGMQueue
except ModuleNotFoundError as e:  # pragma: no cover - exercised only when extra is missing
    logger.error(f"Exception: {e}")
    logger.error('In order to use PgmqBus backends, you need to `uv add "pipecat-ai[pgmq]"`.')
    raise ImportError(f"Missing module: {e}") from e


_INVALID_CHANNEL_CHARS = re.compile(r"[^A-Za-z0-9_]")
_MAX_CHANNEL_LEN = 30
_PEER_LIST_TTL_S = 1.0


def _sanitize_channel(channel: str) -> str:
    """Coerce an arbitrary channel string into a Postgres-safe identifier prefix.

    Only used by :class:`DirectPgmqBackend`, which builds queue names as
    ``{channel}_{uuid}``. :class:`IsolatedPgmqBackend` does not embed the
    channel in the queue name, so this helper does not apply there.

    Args:
        channel: User-supplied channel name (may contain colons, slashes, etc).

    Returns:
        A sanitized prefix safe for use as a PGMQ queue-name prefix.
    """
    safe = _INVALID_CHANNEL_CHARS.sub("_", channel)
    if safe and safe[0].isdigit():
        safe = f"q_{safe}"
    return safe[:_MAX_CHANNEL_LEN] or "pipecat_bus"


@dataclass
class BackendMessage:
    """A message returned by a backend's ``read`` call.

    Backends normalize whatever shape they get off the wire into this
    minimal record so the bus orchestrator can stay backend-agnostic.
    """

    msg_id: int
    message: dict


@runtime_checkable
class PgmqBackend(Protocol):
    """Wire-side interface :class:`PgmqBus` delegates to.

    A backend instance is shared across the lifetime of a bus and may be
    shared across multiple buses (e.g. one process running both an upstream
    and downstream bus on different channels against the same database).
    """

    async def join(self, channel: str) -> str:
        """Allocate a queue for this bus and register it on ``channel``.

        Returns:
            The opaque queue name the bus should read from.
        """
        ...

    async def publish(
        self,
        channel: str,
        my_queue: str,
        payload: dict,
    ) -> None:
        """Fan ``payload`` out to every peer queue on ``channel`` except ``my_queue``."""
        ...

    async def read(
        self,
        queue: str,
        *,
        channel: str,
        vt: int,
        qty: int,
        max_poll_seconds: int,
        poll_interval_ms: int,
    ) -> list[BackendMessage]:
        """Long-poll for messages on ``queue``. Returns an empty list on timeout."""
        ...

    async def archive(
        self,
        queue: str,
        *,
        channel: str,
        msg_id: int,
    ) -> bool:
        """Acknowledge / archive a processed message."""
        ...

    async def leave(self, queue: str, *, channel: str) -> None:
        """Drop the queue and unregister it from ``channel``."""
        ...


class DirectPgmqBackend:
    """Backend that calls :class:`pgmq.async_queue.PGMQueue` directly.

    - Queue names are constructed client-side as ``{safe_channel}_{uuid12}``.
    - Peer discovery uses ``pgmq.list_queues()`` filtered by channel prefix,
      cached for :data:`_PEER_LIST_TTL_S` seconds.
    - Per-peer ``send`` failures are caught; the cache is invalidated and the
      fanout continues. The publish does not raise.

    The provided ``PGMQueue`` must already be initialized (``await pgmq.init()``).
    The backend does not own the client's lifetime.

    Use this when bus peers trust each other. The channel name appears in
    queue names visible to any role that can read ``pg_class``, so channels
    are not secret.
    """

    def __init__(self, pgmq: PGMQueue):
        """Initialize the backend with an already-initialized PGMQueue client."""
        self._pgmq = pgmq
        # channel -> (cached_at, peer_queue_names)
        self._peer_cache: dict[str, tuple[float, list[str]]] = {}

    async def join(self, channel: str) -> str:
        """Create a queue named ``{safe_channel}_{uuid12}`` for this bus."""
        queue_name = f"{_sanitize_channel(channel)}_{uuid.uuid4().hex[:12]}"
        await self._pgmq.create_queue(queue_name)
        logger.debug(f"DirectPgmqBackend: created pgmq queue '{queue_name}'")
        return queue_name

    async def publish(self, channel: str, my_queue: str, payload: dict) -> None:
        """Send ``payload`` to every cached peer queue for ``channel``."""
        peers = await self._peers(channel)
        logger.trace(f"DirectPgmqBackend: publishing to {len(peers)} peer queue(s)")
        for queue_name in peers:
            try:
                await self._pgmq.send(queue_name, payload)
            except Exception:
                logger.warning(
                    f"DirectPgmqBackend: send to peer queue '{queue_name}' failed; "
                    "invalidating cache"
                )
                self._peer_cache.pop(channel, None)

    async def read(
        self,
        queue: str,
        *,
        channel: str,
        vt: int,
        qty: int,
        max_poll_seconds: int,
        poll_interval_ms: int,
    ) -> list[BackendMessage]:
        """Long-poll ``queue`` via ``pgmq.read_with_poll``."""
        messages = await self._pgmq.read_with_poll(
            queue=queue,
            vt=vt,
            qty=qty,
            max_poll_seconds=max_poll_seconds,
            poll_interval_ms=poll_interval_ms,
        )
        return [BackendMessage(msg_id=m.msg_id, message=m.message) for m in messages or []]

    async def archive(self, queue: str, *, channel: str, msg_id: int) -> bool:
        """Archive a processed message via ``pgmq.delete``."""
        return await self._pgmq.delete(queue, msg_id)

    async def leave(self, queue: str, *, channel: str) -> None:
        """Drop the queue and invalidate the cached peer list for ``channel``."""
        await self._pgmq.drop_queue(queue)
        self._peer_cache.pop(channel, None)

    async def _peers(self, channel: str) -> list[str]:
        now = time.monotonic()
        cached = self._peer_cache.get(channel)
        if cached and now - cached[0] < _PEER_LIST_TTL_S:
            return cached[1]
        names = await self._pgmq.list_queues()
        prefix = f"{_sanitize_channel(channel)}_"
        peers = [n for n in names if n.startswith(prefix)]
        self._peer_cache[channel] = (now, peers)
        return peers


class IsolatedPgmqBackend:
    """Backend that calls SECURITY DEFINER Postgres wrappers over asyncpg.

    Use this when bus peers should be isolated from each other and the
    channel name is the bus capability. The backend never issues raw
    ``pgmq.*`` calls; every operation goes through ``public.bus_*``
    wrappers, which enforce ``(queue_name, channel)`` membership against
    a server-side peer registry table.

    Wire format (server-side SQL, defined out-of-band in the deployer's
    migrations)::

        bus_join(p_channel text) RETURNS text
        bus_publish(p_channel text, p_my_queue text, p_message jsonb) RETURNS bigint[]
        bus_subscribe(p_my_queue text, p_channel text, p_vt int,
                      p_qty int, p_max_seconds int)
            RETURNS TABLE(msg_id bigint, message jsonb)
        bus_archive(p_my_queue text, p_channel text, p_msg_id bigint) RETURNS boolean
        bus_leave(p_my_queue text, p_channel text) RETURNS void

    The ``asyncpg.Pool`` must allow at least two concurrent connections
    (one held by the reader loop's long-poll, one for publishes). Pool
    lifetime is owned by the caller — this backend does not close it.

    Notes:
        - ``bus_subscribe`` long-polls inside a SECURITY DEFINER function.
          Use a session-mode pooler; transaction-mode poolers may drop the
          connection mid-poll.
        - Payload serialization: the caller hands :class:`PgmqBus` a
          :mod:`json`-encodable ``dict``; the backend forwards it as
          ``jsonb`` via ``json.dumps`` because asyncpg does not auto-coerce
          ``dict`` to ``jsonb``.
    """

    def __init__(self, pool: Any):
        """Initialize the backend.

        Args:
            pool: An :class:`asyncpg.pool.Pool` that the bus will use for all
                wrapper calls. Typed as ``Any`` to keep :mod:`asyncpg` an
                optional import.
        """
        self._pool = pool

    async def join(self, channel: str) -> str:
        """Call ``public.bus_join(channel)`` and return the server-allocated queue name."""
        async with self._pool.acquire() as conn:
            queue_name = await conn.fetchval(
                "SELECT public.bus_join($1)",
                channel,
            )
        if not queue_name:
            raise RuntimeError("IsolatedPgmqBackend: bus_join returned no queue name")
        logger.debug(f"IsolatedPgmqBackend: joined channel; queue='{queue_name}'")
        return str(queue_name)

    async def publish(self, channel: str, my_queue: str, payload: dict) -> None:
        """Call ``public.bus_publish`` to fan ``payload`` out to channel peers."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "SELECT public.bus_publish($1, $2, $3::jsonb)",
                channel,
                my_queue,
                json.dumps(payload),
            )

    async def read(
        self,
        queue: str,
        *,
        channel: str,
        vt: int,
        qty: int,
        max_poll_seconds: int,
        poll_interval_ms: int,
    ) -> list[BackendMessage]:
        """Long-poll ``queue`` via ``public.bus_subscribe``.

        ``poll_interval_ms`` is honored server-side by ``pgmq.read_with_poll``'s
        default; the wrapper does not expose it.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT msg_id, message FROM public.bus_subscribe($1, $2, $3, $4, $5)",
                queue,
                channel,
                vt,
                qty,
                max_poll_seconds,
            )
        out: list[BackendMessage] = []
        for row in rows:
            raw = row["message"]
            message = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
            out.append(BackendMessage(msg_id=int(row["msg_id"]), message=message))
        return out

    async def archive(self, queue: str, *, channel: str, msg_id: int) -> bool:
        """Call ``public.bus_archive`` to acknowledge a processed message."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT public.bus_archive($1, $2, $3)",
                queue,
                channel,
                msg_id,
            )
        return bool(result)

    async def leave(self, queue: str, *, channel: str) -> None:
        """Call ``public.bus_leave`` to drop the queue and unregister it."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "SELECT public.bus_leave($1, $2)",
                queue,
                channel,
            )


__all__ = [
    "BackendMessage",
    "DirectPgmqBackend",
    "IsolatedPgmqBackend",
    "PgmqBackend",
]

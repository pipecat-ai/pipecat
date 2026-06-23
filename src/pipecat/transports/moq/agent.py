#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Announcement-driven MoQ voice-agent server (no ``/start`` control plane).

Where :class:`~pipecat.transports.moq.transport.MOQTransport` is a single
session that dials a relay (or serves) and waits for one fixed peer, this
module hosts *many* sessions off one connection, discovering clients by MoQ
announcement -- the same pattern as ``moq-boy``'s viewer fan-in.

One long-lived :class:`MOQAgentServer` holds a single ``moq.Client`` and a
shared ``moq.OriginProducer``. It subscribes to a client prefix and, for each
broadcast announced under it, spawns a fresh pipeline:

    {client_prefix}/{id}    <- browser publishes its mic here (unauthenticated)
    {bot_prefix}/{id}       <- the agent publishes its reply here

Mirroring ``moq-boy``'s ``anon``/``demo`` split, the client prefix is
unauthenticated (anyone can talk) while the bot prefix is where the agent
publishes; in production that's an authenticated ``demo/...`` path so only the
server can answer, but the single shared connection holds the one publish
token for the whole fleet.

The per-session media engine (Opus publish/subscribe, the ``transcript.json``
side-channel, audio pacing/PTS) is inherited unchanged from
:class:`MOQTransport`; only the connection bring-up is replaced.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from pipecat.transports.moq.transport import MOQParams, MOQTransport

try:
    import moq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the MOQ agent server, you need to `pip install pipecat-ai[moq]`.")
    raise Exception(f"Missing module: {e}")


DEFAULT_CLIENT_PREFIX = "anon/voice/client"
DEFAULT_BOT_PREFIX = "anon/voice/bot"
DEFAULT_RELAY_URL = "http://localhost:4443"


# Builds and runs one session's pipeline to completion. Called with the
# session transport and its id; should return when the session ends (the
# session fires ``on_disconnected`` once the client's mic track closes).
SessionBot = Callable[["MOQAgentSession", str], Awaitable[None]]

# Bit position of the fleet "internal relay" prefix in a 62-bit origin id (the
# top 30 bits). Mirrors moq-edge's vod::hop PREFIX_SHIFT; the actual prefix VALUE
# is never hardcoded here (see local_ingest).
_INTERNAL_PREFIX_SHIFT = 32


def local_ingest(hops: list[int]) -> bool:
    """Whether this broadcast was ingested on the edge we're connected to.

    Self-election for the per-edge fleet: a client's announcement propagates
    across the relay mesh, so every edge's worker sees it. Only the worker on the
    client's OWN edge should answer, else N edges spawn N bots for one client.

    We elect off the hop chain (oldest-first), the same list moq-edge's VOD
    recorder uses -- but from a different vantage point. The recorder runs
    in-process and a relay never stamps its own id into its `cluster.origin`, so a
    locally-ingested broadcast carries ZERO internal hops there. We reach the
    origin over the internal UDS as a downstream consumer, so the serving edge
    stamps itself LAST: a locally-ingested broadcast carries exactly ONE internal
    hop (that edge), while a peer-forwarded one carries two or more.

    The fleet prefix is read off the LAST hop (always the serving edge, which is
    internal), never hardcoded -- mirroring how the relay derives it from its own
    id, so this can't drift from the value tofu stamps.
    """
    if not hops:
        # Over the UDS the serving edge is always present, so an empty chain is
        # anomalous; don't claim it as ours.
        return False
    prefix = hops[-1] >> _INTERNAL_PREFIX_SHIFT
    internal = sum(1 for hop in hops if (hop >> _INTERNAL_PREFIX_SHIFT) == prefix)
    return internal == 1


class MOQAgentSession(MOQTransport):
    """A single voice session driven by :class:`MOQAgentServer`.

    Unlike :class:`MOQTransport`, this owns no MoQ connection: the server
    holds one shared ``moq.Client`` + ``OriginProducer`` for the whole fleet
    and hands each session the shared ``origin`` to publish its reply on, the
    ``bot_path`` to publish at, and the already-announced client mic broadcast
    to consume. Everything else is inherited.

    Args:
        params: MOQ configuration parameters (audio/transcript settings).
        origin: The server's shared ``OriginProducer`` to publish the reply on.
        bot_path: Full broadcast path for the bot's reply (e.g. ``anon/voice/bot/abc``).
        peer_path: Full broadcast path of the client's mic (e.g. ``anon/voice/client/abc``).
        peer_broadcast: The already-announced ``BroadcastConsumer`` for the mic.
    """

    def __init__(
        self,
        params: MOQParams,
        *,
        origin: moq.OriginProducer,
        bot_path: str,
        peer_path: str,
        peer_broadcast: moq.BroadcastConsumer,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize a server-driven MoQ session bound to a shared origin."""
        super().__init__(params, input_name=input_name, output_name=output_name)
        self._origin = origin
        # Override the namespace-derived defaults from MOQTransport.__init__.
        self._broadcast_path = bot_path
        self._peer_broadcast_path = peer_path
        self._peer_broadcast = peer_broadcast

    async def _run(self):
        """Publish the reply on the shared origin and forward the client's mic.

        Replaces :meth:`MOQTransport._run`: there is no connection to bring up
        (the server owns it), and the peer broadcast is already resolved, so we
        publish, signal connected, and pump audio until the mic track ends.
        """
        try:
            self._origin.publish(self._broadcast_path, self._publish_broadcast)
            logger.info(
                f"MOQ agent: publishing reply {self._broadcast_path!r} "
                f"(audio={self._params.audio_out_track!r}, "
                f"transcript={self._params.transcript_track!r})"
            )
            await self._call_event_handler("on_connected")
            # The mic broadcast is already announced, so the client is "here".
            await self._call_event_handler("on_client_connected")
            await self._forward_peer_audio(self._peer_broadcast)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"MOQ agent session error: {e}", exc_info=True)
            await self._call_event_handler("on_error", str(e), e)
        finally:
            self._audio_out = None
            await self._call_event_handler("on_client_disconnected")
            await self._call_event_handler("on_disconnected")

    async def cleanup(self):
        """Withdraw just this session's broadcast; leave the shared origin up."""
        if self._cleaned_up:
            return
        # The shared origin serves every other session, so only finish THIS
        # broadcast. That withdraws the bot's announcement and the browser sees
        # the agent leave.
        try:
            self._publish_broadcast.finish()
        except Exception as e:
            logger.debug(f"MOQ agent: broadcast.finish() raised: {e}")
        await super().cleanup()


class MOQAgentServer:
    """Discovers clients by announcement and runs one pipeline per client.

    Example::

        async def run_session_bot(transport: MOQAgentSession, client_id: str):
            pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
            worker = PipelineWorker(pipeline)
            ...  # wire on_disconnected -> worker.cancel()
            await WorkerRunner(handle_sigint=False).run()

        server = MOQAgentServer(MOQParams(...), run_session_bot, relay_url="http://localhost:4443")
        await server.run()

    Args:
        params: Per-session MOQ media parameters (shared by every session).
        run_bot: Builds and runs one session's pipeline to completion.
        relay_url: The MoQ relay to dial.
        client_prefix: Announcement prefix to discover client mics under.
        bot_prefix: Prefix to publish bot replies under.
        verify_ssl: Verify the relay's TLS certificate (off for self-signed dev relays).
        max_sessions: Concurrency cap; excess clients queue (each session is a
            full STT+LLM+TTS pipeline, so this bounds cost / rate limits).
    """

    def __init__(
        self,
        params: MOQParams,
        run_bot: SessionBot,
        *,
        relay_url: str = DEFAULT_RELAY_URL,
        client_prefix: str = DEFAULT_CLIENT_PREFIX,
        bot_prefix: str = DEFAULT_BOT_PREFIX,
        verify_ssl: bool = True,
        max_sessions: int = 8,
    ):
        """Initialize the MoQ agent server."""
        self._params = params
        self._run_bot = run_bot
        self._relay_url = relay_url
        self._client_prefix = client_prefix.rstrip("/")
        self._bot_prefix = bot_prefix.rstrip("/")
        self._verify_ssl = verify_ssl
        self._sem = asyncio.Semaphore(max_sessions)
        self._tasks: set[asyncio.Task] = set()
        self._warned_no_hops = False

    async def run(self):
        """Connect, then dispatch a session per announced client. Runs forever."""
        origin = moq.OriginProducer()
        logger.info(
            f"MOQ agent server: connecting to {self._relay_url} "
            f"(discover {self._client_prefix!r}/* -> reply {self._bot_prefix!r}/*)"
        )
        async with moq.Client(
            self._relay_url, publish=origin, subscribe=origin, tls_verify=self._verify_ssl
        ) as client:
            # announced(prefix) re-roots at the prefix, so ann.path is the
            # suffix after it (the session id), e.g. "abc" for
            # "{client_prefix}/abc" -- same as moq-boy's viewer_id. Strip
            # defensively in case a build hands back the full path.
            async with client.announced(self._client_prefix) as announced:
                async for ann in announced:
                    sid = ann.path
                    prefix = self._client_prefix + "/"
                    if sid.startswith(prefix):
                        sid = sid[len(prefix) :]
                    if not sid:
                        continue

                    # Self-election: only answer clients ingested on OUR edge, else
                    # every edge's worker answers the same client (announcements fan
                    # out across the mesh). Inert until moq-rs exposes the hop chain
                    # on the announcement; until then we serve every client (correct
                    # single-edge, duplicated on a fleet -- see _announced_hops).
                    hops = self._announced_hops(ann)
                    if hops is not None and not local_ingest(hops):
                        logger.debug(
                            f"MOQ agent server: {sid!r} ingested on another edge; skipping"
                        )
                        continue

                    logger.info(f"MOQ agent server: client {sid!r} announced")
                    task = asyncio.create_task(self._session(origin, sid, ann.broadcast))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

    def _announced_hops(self, ann) -> list[int] | None:
        """The announced broadcast's hop chain (origin ids), or None if unavailable.

        Proposed moq-rs API: ``ann.hops`` (origin ids, oldest-first), surfaced from
        the same ``broadcast.info().hops`` the Rust VOD recorder reads. Logged once
        so the self-election gap is visible.
        """
        hops = getattr(ann, "hops", None)
        if hops is None:
            if not self._warned_no_hops:
                self._warned_no_hops = True
                logger.warning(
                    "MOQ agent server: this moq-rs build doesn't expose announcement hops; "
                    "self-election is OFF -- safe on a single edge, but a multi-edge fleet "
                    "will spawn a duplicate bot per client. Expose ann.hops to enable it."
                )
            return None
        return list(hops)

    async def _session(
        self, origin: moq.OriginProducer, sid: str, broadcast: moq.BroadcastConsumer
    ):
        if self._sem.locked():
            logger.warning(f"MOQ agent server: at capacity, client {sid!r} queued")
        async with self._sem:
            session = MOQAgentSession(
                self._params,
                origin=origin,
                bot_path=f"{self._bot_prefix}/{sid}",
                peer_path=f"{self._client_prefix}/{sid}",
                peer_broadcast=broadcast,
            )
            logger.info(f"MOQ agent server: session {sid!r} starting")
            try:
                await self._run_bot(session, sid)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"MOQ agent server: session {sid!r} crashed")
            finally:
                await session.cleanup()
                logger.info(f"MOQ agent server: session {sid!r} ended")

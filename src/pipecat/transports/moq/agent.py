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
shared ``moq.OriginProducer``. It subscribes to the REQUEST prefix and, for each
broadcast announced under it, spawns a fresh pipeline:

    {request_prefix}/{id}     <- client publishes its mic (request) here
    {response_prefix}/{id}    <- the agent publishes its reply (response) here

Request and response live under SEPARATE prefixes on purpose: the server only
``announced()``s ``request/*``, so its own ``response/*`` publishes never appear
in its discovery stream (no announcement loop), and a per-client token can be
scoped tightly -- publish ``request/<id>``, subscribe ``response/<id>`` -- so a
client can't read another's request or spoof a response. The prefixes are just
strings; the deployment chooses auth + namespacing (e.g. ``demo/pipecat/request``
behind minted tokens, or open ``request`` for a public demo).

The per-session media engine (Opus publish/subscribe, the ``transcript.json.z``
JSON-stream side-channel, audio pacing/PTS) is inherited unchanged from
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


DEFAULT_REQUEST_PREFIX = "request"
DEFAULT_RESPONSE_PREFIX = "response"
DEFAULT_RELAY_URL = "http://localhost:4443"


# Builds and runs one session's pipeline to completion. Called with the
# session transport and its id; should return when the session ends (the
# session fires ``on_disconnected`` once the client's mic track closes).
SessionBot = Callable[["MOQAgentSession", str], Awaitable[None]]

# Admission policy: decides whether to answer an announced client. Receives the
# raw announcement (``.path``, ``.broadcast``, ``.hops``) and returns True to
# serve it. ``None`` answers every client -- the generic default. A deployment
# injects its own policy here (e.g. self-electing one relay edge per client by the
# hop chain) without this library knowing anything about that policy.
ServeFilter = Callable[["moq.Announcement"], bool]


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
        bot_path: Full broadcast path for the agent's response (e.g. ``response/abc``).
        peer_path: Full broadcast path of the client's request/mic (e.g. ``request/abc``).
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
        request_prefix: Announcement prefix to discover client requests (mics) under.
        response_prefix: Prefix to publish the agent's responses under. Keep it
            DISJOINT from request_prefix so the server never discovers its own
            replies (no announcement loop).
        verify_ssl: Verify the relay's TLS certificate (off for self-signed dev relays).
        max_sessions: Concurrency cap; excess clients queue (each session is a
            full STT+LLM+TTS pipeline, so this bounds cost / rate limits).
        should_serve: Optional admission policy ``(announcement) -> bool``. Return
            False to decline a client (e.g. self-election on a multi-relay fleet).
            Default answers every announced client.
    """

    def __init__(
        self,
        params: MOQParams,
        run_bot: SessionBot,
        *,
        relay_url: str = DEFAULT_RELAY_URL,
        request_prefix: str = DEFAULT_REQUEST_PREFIX,
        response_prefix: str = DEFAULT_RESPONSE_PREFIX,
        verify_ssl: bool = True,
        max_sessions: int = 8,
        should_serve: ServeFilter | None = None,
    ):
        """Initialize the MoQ agent server."""
        self._params = params
        self._run_bot = run_bot
        self._relay_url = relay_url
        self._request_prefix = request_prefix.rstrip("/")
        self._response_prefix = response_prefix.rstrip("/")
        self._verify_ssl = verify_ssl
        self._should_serve = should_serve
        self._sem = asyncio.Semaphore(max_sessions)
        self._tasks: set[asyncio.Task] = set()

    async def run(self):
        """Connect, then dispatch a session per announced client. Runs forever."""
        origin = moq.OriginProducer()
        logger.info(
            f"MOQ agent server: connecting to {self._relay_url} "
            f"(discover {self._request_prefix!r}/* -> reply {self._response_prefix!r}/*)"
        )
        async with moq.Client(
            self._relay_url, publish=origin, subscribe=origin, tls_verify=self._verify_ssl
        ) as client:
            # announced(prefix) re-roots at the prefix, so ann.path is the
            # suffix after it (the session id), e.g. "abc" for
            # "{request_prefix}/abc" -- same as moq-boy's viewer_id. Strip
            # defensively in case a build hands back the full path.
            async with client.announced(self._request_prefix) as announced:
                async for ann in announced:
                    sid = ann.path
                    prefix = self._request_prefix + "/"
                    if sid.startswith(prefix):
                        sid = sid[len(prefix) :]
                    if not sid:
                        continue

                    # Optional admission policy: a deployment can decline a client
                    # here (e.g. self-electing one relay edge per client on a fleet).
                    # Default answers every announced client.
                    if self._should_serve is not None and not self._should_serve(ann):
                        logger.debug(f"MOQ agent server: declined client {sid!r}")
                        continue

                    logger.info(f"MOQ agent server: client {sid!r} announced")
                    task = asyncio.create_task(self._session(origin, sid, ann.broadcast))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

    async def _session(
        self, origin: moq.OriginProducer, sid: str, broadcast: moq.BroadcastConsumer
    ):
        if self._sem.locked():
            logger.warning(f"MOQ agent server: at capacity, client {sid!r} queued")
        async with self._sem:
            session = MOQAgentSession(
                self._params,
                origin=origin,
                bot_path=f"{self._response_prefix}/{sid}",
                peer_path=f"{self._request_prefix}/{sid}",
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

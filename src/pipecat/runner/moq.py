#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MoQ runner helpers.

Configuration helpers used by the development runner to construct the
MoQ relay config the browser needs, whether it arrives in a ``/start``
response or, in direct mode, in the client URL.
"""

import argparse
import secrets
from typing import Any
from urllib.parse import urlencode, urlparse

from loguru import logger

DEFAULT_MOQ_SERVE_BIND = "[::]:4080"
DEFAULT_MOQ_PATH = "/moq"
DEFAULT_MOQ_NAMESPACE = "pipecat"
# Participant ids name the direction each side carries: the bot publishes
# its responses, the browser publishes its requests.
DEFAULT_MOQ_BOT_ID = "response"
DEFAULT_MOQ_CLIENT_ID = "request"

# How long a direct-mode bot holds the relay open waiting for a browser.
# The stock timeouts assume ``/start`` spawned the bot with a client already
# connecting; here the bot arrives first, so the wait is however long it
# takes someone to open the page.
DIRECT_MODE_PEER_WAIT_SECS = 3600.0

# End a direct-mode call after this long with no speech in either
# direction. Idle counts speech frames, not media — an abandoned open tab
# keeps publishing silent mic audio — so this is what stops a forgotten
# call from running its bot (and its STT vendor bill) indefinitely.
DIRECT_MODE_SESSION_IDLE_SECS = 300.0


def _new_session_namespace() -> str:
    """Mint an unguessable namespace for a single client-mode session.

    Client mode dials a shared relay, so the namespace is all that keeps
    one session's broadcasts (``<namespace>/response``) from colliding with
    another's. On an anonymous relay (e.g. ``cdn.moq.dev/anon``) there is
    no authentication either, so the namespace doubles as the session's
    only access control: it must be unguessable, not merely unique.
    Serve mode gets a fixed default instead, since a socket bound to
    localhost isn't reachable to begin with. That's an accident of local
    dev, not isolation: anything that can reach the socket can publish
    the request path. Exposing it needs real access control — a
    path-scoped moq-token — rather than an unguessable name.
    """
    return f"{DEFAULT_MOQ_NAMESPACE}-{secrets.token_hex(8)}"


def _parse_bind_port(bind: str) -> int:
    """Parse the port from a bind address like ``[::]:4080`` or ``0.0.0.0:4080``."""
    _, sep, port_str = bind.rpartition(":")
    if not sep:
        raise ValueError(f"--moq-bind must include a port (got {bind!r})")
    try:
        return int(port_str)
    except ValueError as e:
        raise ValueError(f"--moq-bind has a non-numeric port: {bind!r}") from e


def _validate_moq_args(args: argparse.Namespace) -> bool:
    """Validate MoQ CLI args, warn on conflicts, and stash derived host/port/path on ``args``.

    Returns ``True`` if the args are usable, ``False`` if validation failed.

    Populates: ``args.moq_serve`` (resolved from ``--moq-connect`` when
    left unspecified), ``args.moq_host``, ``args.moq_port``,
    ``args.moq_path``, ``args.moq_bind`` (defaulted in serve mode),
    ``args.moq_tls_host`` (the hostname presented to the browser).
    """
    # Naming a relay with --moq-connect is what selects client mode;
    # there is no default relay to fall back on. Absent that, the bot
    # serves locally, so the common `-t moq` case needs no relay at all.
    if args.moq_serve is None:
        args.moq_serve = args.moq_connect is None

    has_cert = bool(args.moq_tls_cert)
    has_key = bool(args.moq_tls_key)
    has_generate = bool(args.moq_tls_generate)

    if args.moq_serve:
        # Server mode.
        if args.moq_connect is not None:
            logger.warning(
                "--moq-connect is ignored in server mode (use --moq-bind to set the listen address)"
            )
        if args.moq_tls_insecure:
            logger.warning(
                "--moq-tls-insecure is ignored in server mode "
                "(server-side TLS is set via --moq-tls-cert or --moq-tls-generate)"
            )
        if has_cert != has_key:
            logger.error(
                "server mode requires both --moq-tls-cert AND --moq-tls-key "
                "(or use --moq-tls-generate <hostname> for a self-signed dev cert)"
            )
            return False
        if (has_cert and has_key) and has_generate:
            logger.warning(
                "--moq-tls-generate is ignored — using --moq-tls-cert/--moq-tls-key instead"
            )
        elif not (has_cert and has_key) and not has_generate:
            # No TLS config supplied at all — default to a self-signed dev
            # cert for localhost rather than requiring --moq-tls-generate.
            args.moq_tls_generate = "localhost"
            has_generate = True

        bind = args.moq_bind or DEFAULT_MOQ_SERVE_BIND
        try:
            bind_port = _parse_bind_port(bind)
        except ValueError as e:
            logger.error(str(e))
            return False

        # Hostname the browser uses to reach the bot. In dev with
        # --moq-tls-generate, that's the cert hostname. With a CA-signed
        # cert, fall back to localhost (operator can patch via env / code).
        tls_host = args.moq_tls_generate or "localhost"

        args.moq_bind = bind
        args.moq_host = tls_host
        args.moq_port = bind_port
        args.moq_path = DEFAULT_MOQ_PATH
        args.moq_tls_host = tls_host
    else:
        # Client mode, which only --moq-connect can select.
        connect = args.moq_connect
        parsed = urlparse(connect)
        if not parsed.hostname:
            logger.error(
                f"--moq-connect must be a full URL with a host "
                f"(e.g. https://relay.example.com/moq); got {connect!r}"
            )
            return False
        # Default the port from the scheme so URLs on standard HTTPS/HTTP
        # ports (e.g. https://relay.quic.video/anon) don't have to spell
        # it out explicitly.
        default_ports = {"https": 443, "http": 80}
        client_port = parsed.port or default_ports.get(parsed.scheme.lower())
        if client_port is None:
            logger.error(
                f"--moq-connect needs an explicit port; scheme "
                f"{parsed.scheme!r} has no standard default (got {connect!r})"
            )
            return False
        client_host = parsed.hostname
        client_path = parsed.path or DEFAULT_MOQ_PATH

        if has_generate:
            logger.warning("--moq-tls-generate is ignored — only used in server mode")
        if has_key and not has_cert:
            logger.error("--moq-tls-key requires --moq-tls-cert")
            return False
        if has_key and has_cert:
            logger.warning(
                "--moq-tls-key is ignored in client mode (--moq-tls-cert is used for "
                "self-signed cert fingerprint pinning only)"
            )

        args.moq_host = client_host
        args.moq_port = client_port
        args.moq_path = client_path
        args.moq_tls_host = client_host

    # An explicit --moq-namespace always wins. Otherwise serve mode gets
    # the fixed default (the bot owns its socket, so a stable, readable
    # path is free) while client mode is left unresolved so every /start
    # can mint its own — see :func:`_new_session_namespace`.
    if args.moq_namespace is None and args.moq_serve:
        args.moq_namespace = DEFAULT_MOQ_NAMESPACE

    if getattr(args, "moq_direct", False):
        if args.moq_serve:
            logger.error(
                "--moq-direct needs a relay to meet the browser on: pass "
                "--moq-connect <url>. Serve mode hands the browser its cert "
                "fingerprint through /start, which direct mode doesn't have."
            )
            return False
        # Every call already carries the browser's unguessable session id,
        # so a namespace isn't needed to keep callers apart. Default to
        # none and let the paths sit at the relay root, which keeps the
        # client URL down to the relay. Naming one scopes the bot to that
        # room instead, so it ignores traffic anywhere else on the relay.
        if args.moq_namespace is None:
            args.moq_namespace = ""

    return True


def _join_path(*parts: str) -> str:
    """Join broadcast path components, dropping empty ones.

    An unset namespace has to vanish rather than leave a leading slash,
    since moq paths are relative and trim their boundaries.
    """
    return "/".join(p for p in parts if p)


def _client_prefix(args: argparse.Namespace) -> str:
    """Return the broadcast prefix browsers announce themselves under.

    Direct mode watches this rather than one fixed path, because the id
    that separates one caller from the next is minted by the browser.
    """
    return f"{_join_path(args.moq_namespace, args.moq_client_id)}/"


def _session_paths(args: argparse.Namespace, session: str) -> tuple[str, str]:
    """Return the ``(response, request)`` broadcast paths for one session.

    Both sides of a call hang off the id the browser chose, so a bot
    serves exactly the caller that announced it and nobody else.
    """
    return (
        _join_path(args.moq_namespace, args.moq_bot_id, session),
        f"{_client_prefix(args)}{session}",
    )


class MOQDirectHost:
    """Serve a MoQ direct-mode bot per browser under a request prefix.

    The host dials a relay as a client and watches for browsers announcing
    themselves under ``request_prefix``; each browser mints its own session
    id, and the host runs one ``bot(runner_args)`` per id. Sessions can't
    collide — every call lives on its own pair of broadcast paths — so a
    single host serves any number of calls, concurrently or back to back.

    Lifecycle guards, which matter wherever instances are billed or capped
    (a deployed host with no exit would hold an agent slot forever):

    - Per-session limits belong to the ``runner_args`` the factory returns:
      ``connection_timeout`` bounds the wait for an announcing browser's
      media, and ``pipeline_idle_timeout_secs`` ends calls with no speech
      in either direction (an abandoned open tab keeps publishing silent
      mic audio, which is not activity).
    - ``host_idle_secs`` exits the host after that long with no live calls.
      ``None`` runs until cancelled — right for a development server, wrong
      for a capped cloud deployment.

    Browser departures are not announced by the relay (moq-ffi exposes no
    deactivation events), so a session ends when its transport sees the
    peer's streams close, bounded by the idle guards above.

    Example (a cloud entry point)::

        async def bot(runner_args):
            host = MOQDirectHost.from_env(run_bot)
            await host.run()
    """

    def __init__(
        self,
        bot,
        *,
        relay_url: str,
        request_prefix: str,
        runner_args_factory,
        verify_ssl: bool = True,
        host_idle_secs: float | None = None,
    ):
        """Initialize the host.

        Args:
            bot: Async callable invoked as ``bot(runner_args)`` once per
                session; the call *is* the session, ending when it returns.
            relay_url: Relay the host dials to watch for browsers.
            request_prefix: Broadcast prefix browsers announce under,
                including the trailing slash (e.g. ``pcc/request/``).
            runner_args_factory: Callable mapping a browser-minted session
                id to the ``RunnerArguments`` its bot receives — including
                the per-session broadcast paths and idle limits.
            verify_ssl: Verify the relay's TLS certificate.
            host_idle_secs: Exit after this long with no live calls;
                ``None`` runs until cancelled.
        """
        self._bot = bot
        self._relay_url = relay_url
        self._request_prefix = request_prefix
        self._runner_args_factory = runner_args_factory
        self._verify_ssl = verify_ssl
        self._host_idle_secs = host_idle_secs

    @classmethod
    def from_env(cls, bot) -> "MOQDirectHost":
        """Build a host from ``MOQ_*`` environment variables.

        For platforms that start bots without CLI arguments (e.g. Pipecat
        Cloud). Reads:

        - ``MOQ_RELAY_URL`` (required)
        - ``MOQ_NAMESPACE`` (default empty — calls at the relay root)
        - ``MOQ_BOT_ID`` / ``MOQ_CLIENT_ID`` (default ``response`` /
          ``request``)
        - ``MOQ_TLS_INSECURE`` — set to 1 to skip relay cert verification
        - ``MOQ_PEER_WAIT_SECS`` — per-session wait for the announcing
          browser's media (default 60)
        - ``MOQ_SESSION_IDLE_SECS`` — end calls with no speech for this
          long; 0 disables (default 60)
        - ``MOQ_HOST_IDLE_SECS`` — exit after this long with no calls;
          0 disables (default 60)
        """
        import os

        from pipecat.runner.types import MOQRunnerArguments

        relay_url = os.environ["MOQ_RELAY_URL"]
        namespace = os.getenv("MOQ_NAMESPACE", "")
        bot_id = os.getenv("MOQ_BOT_ID", DEFAULT_MOQ_BOT_ID)
        client_id = os.getenv("MOQ_CLIENT_ID", DEFAULT_MOQ_CLIENT_ID)
        verify_ssl = os.getenv("MOQ_TLS_INSECURE", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        )
        peer_wait = float(os.getenv("MOQ_PEER_WAIT_SECS", "60"))
        session_idle = float(os.getenv("MOQ_SESSION_IDLE_SECS", "60")) or None
        host_idle = float(os.getenv("MOQ_HOST_IDLE_SECS", "60")) or None

        parsed = urlparse(relay_url)
        hostname = parsed.hostname
        if not hostname:
            raise ValueError(f"MOQ_RELAY_URL has no hostname: {relay_url!r}")

        def factory(session: str) -> MOQRunnerArguments:
            runner_args = MOQRunnerArguments(
                host=hostname,
                port=parsed.port or 443,
                path=parsed.path or "/",
                namespace=namespace,
                participant_id=bot_id,
                peer_id=client_id,
                verify_ssl=verify_ssl,
                serve=False,
                session_id=session,
                connection_timeout=peer_wait,
                response_path=_join_path(namespace, bot_id, session),
                request_path=_join_path(namespace, client_id, session),
            )
            runner_args.handle_sigint = False  # one host, many sessions
            runner_args.pipeline_idle_timeout_secs = session_idle
            return runner_args

        return cls(
            bot,
            relay_url=relay_url,
            request_prefix=f"{_join_path(namespace, client_id)}/",
            runner_args_factory=factory,
            verify_ssl=verify_ssl,
            host_idle_secs=host_idle,
        )

    async def run(self):
        """Watch the relay and serve a bot per announced session id.

        Returns when ``host_idle_secs`` elapses with no live calls, or
        raises if the relay watch itself fails.
        """
        import asyncio
        import time

        import moq

        sessions: dict[str, asyncio.Task] = {}

        async def run_session(session: str):
            try:
                await self._bot(self._runner_args_factory(session))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.opt(exception=e).error(f"MoQ session {session!r} failed: {e}")
            finally:
                logger.info(f"MoQ session {session!r} ended")

        async def watch(origin: "moq.OriginProducer"):
            # `path` is relative to the prefix, so it is the id itself.
            async for announcement in origin.consume().announced(self._request_prefix):
                session = announcement.path
                for done in [s for s, t in sessions.items() if t.done()]:
                    del sessions[done]
                if session in sessions:
                    continue
                logger.info(f"MoQ direct: client {session!r} arrived, starting a bot")
                sessions[session] = asyncio.create_task(run_session(session))

        idle_note = (
            f" (exits after {self._host_idle_secs:.0f}s with no calls)"
            if self._host_idle_secs
            else ""
        )
        logger.info(
            f"MoQ direct host: watching {self._request_prefix!r} on {self._relay_url}{idle_note}"
        )
        origin = moq.OriginProducer()
        async with moq.Client(
            self._relay_url, tls_verify=self._verify_ssl, publish=origin, subscribe=origin
        ):
            watch_task = asyncio.create_task(watch(origin))
            last_call = time.monotonic()
            try:
                while True:
                    await asyncio.wait({watch_task}, timeout=15)
                    if watch_task.done():
                        await watch_task  # surface a relay/watch failure
                        break
                    if any(not t.done() for t in sessions.values()):
                        last_call = time.monotonic()
                    elif (
                        self._host_idle_secs is not None
                        and time.monotonic() - last_call > self._host_idle_secs
                    ):
                        logger.info(
                            f"MoQ direct host: no calls for {self._host_idle_secs:.0f}s — exiting"
                        )
                        break
            finally:
                watch_task.cancel()
                for task in sessions.values():
                    task.cancel()


def _direct_client_url(args: argparse.Namespace, runner_url: str) -> str:
    """Build the browser URL carrying the relay config as query params.

    Direct mode has no ``/start`` response to deliver that config, so
    everything the browser can't derive on its own — where to dial, which
    namespace to meet on, and which end of the path pair each side owns —
    rides in the URL instead. Points at the prebuilt UI directly because
    the root redirect drops the query.

    Values the client already defaults to are left out, so an unscoped
    run reduces to just the relay.
    """
    params = {"relay": f"https://{args.moq_host}:{args.moq_port}{args.moq_path}"}
    if args.moq_namespace:
        params["ns"] = args.moq_namespace
    if args.moq_bot_id != DEFAULT_MOQ_BOT_ID:
        params["botId"] = args.moq_bot_id
    if args.moq_client_id != DEFAULT_MOQ_CLIENT_ID:
        params["clientId"] = args.moq_client_id
    return f"{runner_url}/client/?{urlencode(params)}"


def _cert_hash_from_pem(path: str) -> str | None:
    """Compute the base64 SHA-256 of a PEM-encoded cert on disk.

    Used in client mode when ``--moq-tls-cert`` is set and we need the
    fingerprint to send to the browser for WebTransport pinning.
    """
    try:
        import base64
        import hashlib

        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        with open(path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())
        der_bytes = cert.public_bytes(serialization.Encoding.DER)
        digest = hashlib.sha256(der_bytes).digest()
        return base64.b64encode(digest).decode()
    except Exception as e:
        logger.warning(f"Could not compute cert fingerprint from {path}: {e}")
        return None


def _hex_to_b64(hex_str: str) -> str | None:
    """Convert a hex-encoded SHA-256 digest to base64.

    The bot exposes its serve-mode cert fingerprints as hex (the moq
    library's format); the browser's ``serverCertificateHashes`` expects
    base64 of the raw bytes.
    """
    try:
        import base64

        return base64.b64encode(bytes.fromhex(hex_str)).decode()
    except Exception as e:
        logger.warning(f"Could not convert cert hash {hex_str!r}: {e}")
        return None


def _build_moq_client_config(
    args: argparse.Namespace,
    namespace: str,
    cert_fingerprints: list[str] | None = None,
) -> dict[str, Any]:
    """Build the MoQ relay config the browser needs to construct a transport.

    Returned from POST /start (under the ``moq`` key) so the React UI can
    pipe it into ``MoqTransport``'s constructor without a separate fetch.

    In serve mode the bot just minted (or loaded) its own cert; we use
    the fingerprint it reported (passed via ``cert_fingerprints``).
    Otherwise we fall back to the PEM file at ``--moq-tls-cert``, which
    in client mode is only meaningful for a self-signed relay — a public
    one like ``cdn.moq.dev/anon`` is CA-signed and needs no pinning.

    Track names aren't pinned here — the bot publishes a catalog at
    runtime and the browser reads whatever it advertises (codec, sample
    rate, channel count, track name). Lets us add tracks (video,
    screen-share) without a server-side config update.
    """
    # certHash must be set ONLY for self-signed certs. It purposely
    # doesn't work with valid CA-signed TLS certificates.
    cert_hash: str | None = None
    if args.moq_serve and cert_fingerprints:
        cert_hash = _hex_to_b64(cert_fingerprints[0])
    elif getattr(args, "moq_tls_cert", None):
        cert_hash = _cert_hash_from_pem(args.moq_tls_cert)

    # WebTransport always uses HTTPS — even for self-signed dev relays,
    # the cert is pinned via `certHash` below.
    return {
        "relayUrl": f"https://{args.moq_host}:{args.moq_port}{args.moq_path}",
        "certHash": cert_hash,
        "serve": args.moq_serve,
        "namespace": namespace,
        "clientId": args.moq_client_id,
        "botId": args.moq_bot_id,
        "transcriptTrack": "transcript.json.z",
    }

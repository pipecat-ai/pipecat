#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SIP connection implementation for Pipecat.

This module provides :class:`SIPConnection`, the protocol object behind
:class:`~pipecat.transports.sip.transport.SIPTransport`. It wraps the
baresip-python binding — registration, inbound and outbound calls, DTMF,
hold and transfer, and programmatic audio/video access — and exposes them
as plain methods plus :class:`~pipecat.utils.base_object.BaseObject`
events, all in SIP vocabulary. The Daily-compatible dial-in/dial-out
vocabulary lives one layer up, in the transport.

The sharing model, from the outside in:

- **One SIP stack per process.** The first connection to
  :meth:`~SIPConnection.connect` starts it and fixes the runtime-wide
  settings; the last to :meth:`~SIPConnection.disconnect` closes it
  (reference-counted).
- **One user agent — one registration — per address-of-record.** Cached
  and shared: any number of connections on the same account reuse it, and
  the last of them to disconnect unregisters it, so a closed bot leaves no
  binding behind at the registrar.
- **N connections per address-of-record, one call per connection.** A
  connection is a call slot. Each inbound INVITE is routed to the first
  idle connection bound to the account — the claim is synchronous, so
  two INVITEs can never land on one connection — and refused with 486
  when every connection is busy (the stack's own
  ``max_concurrent_calls`` cap refuses even earlier). That cap defaults to
  2 and is runtime-wide, so holding more than two calls at once means
  passing a raised ``max_concurrent_calls`` on **every** connection.

So "one registration, N concurrent conversations" is spelled: one
process, one runtime, one user agent for the account, and N connections
— typically one per pipeline/transport.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, replace
from typing import Optional

from loguru import logger

from pipecat.transports.sip.media_encryption import SIPMediaEncryption
from pipecat.utils.base_object import BaseObject
from pipecat.utils.shared import acquires, releases

try:
    from baresip import (
        Account,
        AudioNotActive,
        AudioRestarted,
        BaresipError,
        Call,
        Config,
        Event,
        Runtime,
        StackEvent,
        UserAgent,
        VideoNotActive,
        VideoRestarted,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use the SIP transport, you need to `uv add "pipecat-ai[sip]"`.')
    raise ImportError(f"Missing module: {e}") from e


class _BaresipLogBridge(logging.Handler):
    """Routes the binding's stdlib ``baresip.*`` records into loguru.

    The binding logs through the standard library (attaching only a
    NullHandler), so without a bridge the native stack's warnings never
    reach pipecat's loguru output. Correlation fields (``sip_call_id``,
    ...) are re-bound so loguru sinks keep seeing them.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        fields = ("seq", "cmd", "event", "call", "sip_call_id", "peer")
        bound = logger.bind(**{f: getattr(record, f) for f in fields if hasattr(record, f)})
        bound.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _install_baresip_log_bridge():
    """Attach the bridge once, when the runtime first starts.

    Deliberately not at import time: an application that configures the
    ``baresip`` logger itself should not have handlers added just by
    importing this module.
    """
    baresip_logger = logging.getLogger("baresip")
    if not any(isinstance(h, _BaresipLogBridge) for h in baresip_logger.handlers):
        baresip_logger.addHandler(_BaresipLogBridge())


# The binding's log-level names mapped onto the stdlib gate the bridge
# sits behind.
_STDLIB_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# The exact close reason baresip reports on the leg it transferred away when a
# REFER (blind or with Replaces) it sent completes — mirrors baresip-python's
# ``Call._TRANSFER_SUCCESS`` (its own spelling). Distinguishes a transfer from a
# hangup on the close, so the flag that requests a transfer need not be trusted
# across the close race.
_TRANSFER_CLOSE_REASON = "Call transfered"

# How long a closing connection waits for the registrar to confirm the
# unregister before giving up on it: a dead registrar must not hold up
# shutdown, and the binding expires on its own anyway.
_UNREGISTER_TIMEOUT = 2.0


_JITTER_BUFFER_MODES = ("off", "fixed", "adaptive")

#: The receive jitter buffer a connection runs unless told otherwise: fixed,
#: 40 ms. A fixed buffer holds every packet for its minimum and returns to it
#: after any disturbance, and its minimum is the arrival excursion it absorbs
#: before discarding frames — 40 ms is over ten times the jitter measured on
#: a datacentre SIP path, at 60 ms less turn latency than the stack's own
#: 100–200 ms. The stack's setting remains available as ("fixed", (100, 200)).
DEFAULT_JITTER_BUFFER: tuple[str, tuple[int, int]] = ("fixed", (40, 60))


def _normalize_jitter_buffer(mode: str | None, ms: tuple | None) -> tuple[str | None, tuple | None]:
    """Validate the jitter-buffer arguments; a range given alone means a fixed buffer.

    Neither given selects :data:`DEFAULT_JITTER_BUFFER`.
    """
    if mode is None and ms is None:
        return DEFAULT_JITTER_BUFFER
    if mode is not None and mode not in _JITTER_BUFFER_MODES:
        raise ValueError(f"jitter_buffer_mode must be one of {_JITTER_BUFFER_MODES}, got {mode!r}")
    if ms is None:
        if mode in ("fixed", "adaptive"):
            raise ValueError(f"jitter_buffer_mode={mode!r} needs jitter_buffer_ms=(min, max)")
        return mode, None
    if mode == "off":
        raise ValueError("jitter_buffer_ms has no effect with jitter_buffer_mode='off'")
    try:
        lo, hi = (int(v) for v in ms)
    except (TypeError, ValueError):
        raise ValueError(f"jitter_buffer_ms must be a (min, max) pair in ms, got {ms!r}") from None
    if not 0 < lo <= hi:
        raise ValueError(f"jitter_buffer_ms needs 0 < min <= max, got {ms!r}")
    return mode or "fixed", (lo, hi)


def _validate_sip_port(port: int | None) -> int | None:
    """Check a ``sip_port``: an integer from 1 to 65535, or None."""
    if port is None:
        return None
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError(f"sip_port must be an integer in 1-65535, got {port!r}")
    return port


def _validate_extra_config(lines: tuple | None) -> tuple:
    """Check ``extra_config`` lines and return them as a tuple.

    Raises:
        ValueError: ``lines`` is a bare string, or an entry is not a
            non-empty single line.
    """
    if lines is None:
        return ()
    if isinstance(lines, str):
        raise ValueError("extra_config must be a tuple of lines, not a single string")
    checked = tuple(lines)
    for line in checked:
        if not isinstance(line, str) or not line.strip() or any(c in line for c in "\n\r\0"):
            raise ValueError(f"extra_config entries must be non-empty single lines, got {line!r}")
    return checked


# The stack's buffer for the headers of the message that ends a call, in
# bytes. The headers are held there as ``"Name: value\r\n"`` lines with a
# terminator, so their lines must come to less than this.
_HANGUP_HEADERS_MAX_BYTES = 2048


def _validate_headers(headers: dict | None) -> dict | None:
    """Check custom SIP headers and return a copy, or None for no headers.

    The binding makes the same checks when the headers are sent. Headers
    stored for later (see :meth:`SIPConnection.set_hangup_headers`) are
    checked here instead, so a bad one is reported where it was given and
    not while the call is being torn down.

    Raises:
        ValueError: A name that is empty or holds a colon, a space or a line
            break; a value that is None or holds a line break; or headers
            too long for the stack to carry.
    """
    if not headers:
        return None
    checked = {}
    size = 0
    for name, value in headers.items():
        if not isinstance(name, str) or not name or any(c in name for c in "\r\n: "):
            raise ValueError(f"invalid SIP header name {name!r}")
        if value is None:
            raise ValueError(f"SIP header {name}: the value is None")
        value = str(value)
        if "\r" in value or "\n" in value:
            raise ValueError(f"SIP header {name}: the value must be a single line")
        checked[name] = value
        size += len(f"{name}: {value}\r\n".encode())
    if size >= _HANGUP_HEADERS_MAX_BYTES:
        raise ValueError(
            f"SIP headers must come to less than {_HANGUP_HEADERS_MAX_BYTES} bytes, got {size}"
        )
    return checked


@dataclass(frozen=True)
class _RuntimeSettings:
    """Runtime-wide baresip settings, fixed by the first connection.

    The SIP stack is one per process, so these knobs cannot vary between
    connections; a later connection asking for different values is a
    configuration error, not a preference.
    """

    net_interface: str | None = None
    sip_port: int | None = None
    expose_headers: tuple = ()
    max_concurrent_calls: int | None = 2
    rtp_timeout: int = 0
    instance_id: str | None = None
    video_size: tuple = (640, 480)
    video_fps: float = 30.0
    video_bitrate: int = 1_000_000
    native_log_level: str = "warning"
    sip_trace: bool = False
    jitter_buffer_mode: str | None = None
    jitter_buffer_ms: tuple | None = None
    extra_config: tuple = ()

    def to_config(self) -> Config:
        """These settings as the binding's ``Config`` object.

        The object form matters: ``expose_headers`` and ``instance_id``
        are applied by ``Runtime.start()`` only when it receives a
        ``Config`` — rendered text cannot carry them.
        """
        return Config(
            net_interface=self.net_interface,
            expose_headers=self.expose_headers,
            max_concurrent_calls=self.max_concurrent_calls,
            rtp_timeout=self.rtp_timeout,
            instance_id=self.instance_id,
            video_size=self.video_size,
            video_fps=self.video_fps,
            video_bitrate=self.video_bitrate,
            native_log_level=self.native_log_level,
            sip_trace=self.sip_trace,
            extra_config_text=(
                self._sip_listen_config() + self._jitter_buffer_config() + self._extra_config_text()
            ),
        )

    def _sip_listen_config(self) -> str:
        """The SIP port as the stack's own ``sip_listen`` config line.

        ``Config`` has no typed field for it. The stack requires an address
        in the line but binds the port on each interface it uses (see
        ``net_interface``), so the address is always ``0.0.0.0``. Unset
        renders nothing, and the stack binds an ephemeral port.
        """
        if self.sip_port is None:
            return ""
        return f"sip_listen 0.0.0.0:{self.sip_port}\n"

    def _jitter_buffer_config(self) -> str:
        """The receive jitter buffer as the stack's own config lines.

        ``Config`` has no typed field for ``audio_jitter_buffer_*``, so the
        lines go through its raw-text escape hatch. A connection always
        carries a mode (the constructor resolves ``None`` to the default);
        settings built without one render nothing and leave the stack's
        compiled setting in place.
        """
        if self.jitter_buffer_mode is None:
            return ""
        lines = [f"audio_jitter_buffer_type {self.jitter_buffer_mode}"]
        if self.jitter_buffer_ms is not None:
            lo, hi = self.jitter_buffer_ms
            lines.append(f"audio_jitter_buffer_ms {lo}-{hi}")
        return "".join(f"{line}\n" for line in lines)

    def _extra_config_text(self) -> str:
        """The application's own ``extra_config`` lines, one per line.

        They come after the lines this class renders, so the text a
        connection produced before ``extra_config`` existed is unchanged at
        its start. Nothing given renders nothing.
        """
        return "".join(f"{line}\n" for line in self.extra_config)

    @property
    def stdlib_log_level(self) -> int:
        """The stdlib level for the ``baresip`` logger these settings need.

        The SIP trace is emitted at DEBUG, so enabling it must open the
        gate that far; otherwise the gate follows ``native_log_level``.
        """
        if self.sip_trace:
            return logging.DEBUG
        return _STDLIB_LOG_LEVELS.get(self.native_log_level, logging.WARNING)


class _SharedRuntime:
    """The process-wide SIP stack, shared by every connection.

    Reference-counted: the first :meth:`acquire` starts the runtime, the
    last :meth:`release` closes it. User agents are cached per
    address-of-record so several connections can share one registration,
    and inbound calls on a shared account are routed to an idle
    connection (or rejected when none is free).
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._runtime: Runtime | None = None
        self._settings: _RuntimeSettings | None = None
        self._owners = 0
        self._uas: dict = {}
        self._ua_accounts: dict = {}
        self._ua_users: dict = {}
        self._bindings: dict = {}
        self._pending_tasks: set = set()

    async def acquire(self, settings: _RuntimeSettings) -> Runtime:
        """Start (or reuse) the runtime and count an owner.

        Args:
            settings: The runtime-wide settings this owner needs.

        Returns:
            The running runtime.

        Raises:
            ValueError: The runtime is already up with different
                runtime-wide settings.
        """
        async with self._lock:
            if self._runtime is None:
                # Bridge and open the stdlib gate before the stack's first
                # line so the requested verbosity is captured from the start.
                _install_baresip_log_bridge()
                logging.getLogger("baresip").setLevel(settings.stdlib_log_level)
                runtime = Runtime()
                await runtime.start(settings.to_config())
                self._runtime = runtime
                self._settings = settings
            elif settings != self._settings:
                raise ValueError(
                    "the SIP runtime is already up with different runtime-wide "
                    f"settings ({self._settings!r}); all SIPConnections in a process "
                    "must agree on net_interface, sip_port, expose_headers, "
                    "max_concurrent_calls, rtp_timeout, instance_id, "
                    "native_log_level, sip_trace, the jitter buffer, extra_config, "
                    "and the video parameters"
                )
            self._owners += 1
            return self._runtime

    async def release(self):
        """Drop one owner; the last one closes the runtime."""
        async with self._lock:
            if self._owners == 0:
                return
            self._owners -= 1
            if self._owners > 0:
                return
            runtime = self._runtime
            self._runtime = None
            self._settings = None
            self._uas.clear()
            self._ua_accounts.clear()
            self._ua_users.clear()
            self._bindings.clear()
        if runtime is not None:
            await runtime.close()

    async def bind_ua(self, account: Account, connection: "SIPConnection") -> UserAgent:
        """Get the account's user agent and route its inbound calls.

        The first binding for an address-of-record allocates the user
        agent and installs the inbound dispatcher; later bindings share
        it. An inbound call goes to the first bound connection without an
        active call, and is rejected when every connection is busy. A
        connection with ``route_inbound=False`` (a consult/transfer leg)
        gets the shared user agent to place its outbound call but is left
        out of the inbound routing list.
        """
        aor = f"sip:{account.user}@{account.domain}"
        async with self._lock:
            connections = self._bindings.setdefault(aor, [])
            if aor in self._uas and connection._account_kwargs != self._ua_accounts.get(aor):
                raise ValueError(
                    f"a user agent for {aor} is already up with different account "
                    "parameters; connections sharing an address-of-record must agree "
                    "on transport, registrar, password, auth_user, audio_codecs, "
                    "dtmf_mode, media_encryption, and extra_params (only "
                    "reg_interval may differ)"
                )
            if aor not in self._uas:
                ua = await UserAgent.create(self._runtime, account)
                self._uas[aor] = ua
                self._ua_accounts[aor] = dict(connection._account_kwargs)
                pending = self._pending_tasks

                def route_incoming(call: Call):
                    for candidate in list(connections):
                        if candidate.take_incoming(call):
                            return
                    logger.warning(f"SIP incoming call from {call.peer}: all connections busy")
                    task = asyncio.create_task(_reject_quietly(call))
                    pending.add(task)
                    task.add_done_callback(pending.discard)

                ua.on_incoming(route_incoming)
            self._ua_users[aor] = self._ua_users.get(aor, 0) + 1
            if connection._route_inbound and connection not in connections:
                connections.append(connection)
            return self._uas[aor]

    async def unbind(self, aor: str, connection: "SIPConnection"):
        """Stop routing inbound calls to a connection.

        The account's last connection also unregisters the user agent, so
        the registrar stops forwarding calls to a bot that is gone. A
        failed or unanswered unregister is logged and otherwise ignored:
        the binding expires at the registrar on its own.
        """
        async with self._lock:
            connections = self._bindings.get(aor)
            if connections and connection in connections:
                connections.remove(connection)
            remaining = self._ua_users.get(aor, 0) - 1
            if remaining > 0:
                self._ua_users[aor] = remaining
                return
            self._ua_users.pop(aor, None)
            ua = self._uas.pop(aor, None)
            self._ua_accounts.pop(aor, None)
        if ua is None:
            return
        try:
            await ua.unregister(timeout=_UNREGISTER_TIMEOUT)
        except BaresipError as e:
            logger.debug(f"SIP unregister of {aor} on disconnect: {e}")


async def _reject_quietly(call: Call):
    try:
        await call.reject()
    except BaresipError as e:
        logger.debug(f"rejecting surplus SIP call: {e}")


_SHARED = _SharedRuntime()


class SIPConnection(BaseObject):
    """A SIP endpoint handling one call at a time.

    Wraps the baresip-python binding: registration (or registration-less
    trunk operation), one inbound or outbound call, DTMF in both
    directions, hold/resume, blind and attended transfer, and the
    programmatic audio/video taps the transport consumes. Constructor
    arguments are plain types only — the binding never appears in
    application signatures.

    Runtime-wide arguments (``net_interface``, ``sip_port``, ``expose_headers``,
    ``max_concurrent_calls``, the jitter buffer, the logging parameters,
    ``extra_config``, and the video parameters) configure the process-wide SIP stack and are
    fixed by the first connection to connect; every later connection must
    pass the same values.

    Event handlers available:

    - connected: the connection is attached to the running SIP stack
      (fires once, after the shared connect work completes).
    - disconnected: the connection detached from the stack.
    - registered: registration succeeded; receives the address-of-record.
    - incoming: an inbound call arrived and this connection took it;
      receives a payload dict (``sessionId``, ``direction``, ``sipFrom``,
      ``sipTo``, ``displayName``, ``sipHeaders``).
    - call_progress: the outbound call is ringing or in early dialog;
      receives a payload dict.
    - call_established: media is up — the call is answered and audio RTP
      flows (not merely the SIP handshake); receives a payload dict.
    - call_closed: the call ended; receives a payload dict with ``reason``
      and ``established`` (whether call_established had fired).
    - call_failed: an outbound call ended without establishing; receives
      a payload dict with ``error`` (the typed failure's name) and
      ``message``.
    - dtmf: the far end pressed a key; receives the binding's DigitEvent
      and the call's session id.
    - info: the far end sent a SIP INFO request on the call; receives a
      payload dict with ``sessionId``, ``contentType`` (empty when the
      request had none) and ``body`` (the raw bytes, exactly as sent). A
      PBX or contact-center platform uses INFO to pass an application
      event inside a call; what the body means is between the two sides.
      DTMF sent as INFO arrives as ``dtmf``, not here.
    - remote_hold: the far end put the call on hold or resumed it;
      receives a payload dict with ``on`` (bool).
    - renegotiated: an established call renegotiated (a mid-call
      re-INVITE offer arrived, or the peer answered ours); receives a
      payload dict with ``sdp`` (``"offer"`` or ``"answer"``).
    - audio_warning: the call's audio layer reported a warning; receives
      the warning, the call direction ("in"/"out"), and the session id.
    - media_restarted: a renegotiation replaced the media streams;
      receives ``"audio"`` or ``"video"``. Consumers should rebuild
      resamplers and re-read geometry.

    Example::

        connection = SIPConnection(user="1001", domain="example.com", password="...")

        @connection.event_handler("incoming")
        async def on_incoming(connection, data):
            await connection.answer()

    Behind NAT, media needs an address in the SDP that the peer can reach.
    It is set through ``extra_params`` (raw baresip account parameters).

    ``medianat=stun`` advertises the address a STUN server reports back.
    That address reaches the bot behind an endpoint-independent NAT — a
    home or office NAT, or a host with a one-to-one public address such as
    an EC2 Elastic IP. Behind an endpoint-dependent ("symmetric") NAT, an
    AWS NAT Gateway among them, it does not: the NAT allocates a fresh
    external port per destination, so the port STUN reported is valid only
    towards the STUN server, and the peer's RTP to the advertised port is
    dropped — the bot receives nothing. Two things carry media across such
    a NAT:

    - The peer latches onto the source address and port the bot's RTP
      actually arrives from, rather than trusting the SDP ("symmetric
      RTP"; the ``Symmetric RTP`` setting on a Twilio Elastic SIP Trunk).
      baresip sends from call establishment, so there is always something
      to latch onto. This is per peer: every carrier or SBC the bot talks
      to has to do it.
    - ``medianat=turn`` works whatever the peer does. baresip allocates a
      relayed address on a TURN server, advertises that in the SDP, and
      binds the peer's RTP and RTCP addresses on the relay, for the cost of
      one relay hop of latency. It takes four parameters:
      ``medianat=turn``, ``stunserver=turn:HOST:3478`` (the scheme must be
      ``turn:`` or ``turns:``; UDP unless the URI carries
      ``?transport=tcp``), and ``stunuser`` / ``stunpass``, both mandatory.

    ``medianat=ice`` is for a peer that also speaks ICE, never a PSTN SIP
    trunk::

        connection = SIPConnection(
            user="1001",
            domain="example.com",
            password="...",
            extra_params=("medianat=stun", "stunserver=stun:stun.l.google.com:19302"),
        )
    """

    def __init__(
        self,
        *,
        user: str,
        domain: str,
        password: str = "",
        auth_user: str | None = None,
        transport: str = "udp",
        registrar: str | None = None,
        reg_interval: int = 240,
        audio_codecs: tuple | None = None,
        dtmf_mode: str = "rtpevent",
        media_encryption: SIPMediaEncryption = SIPMediaEncryption.NONE,
        extra_params: tuple | None = None,
        jitter_buffer_mode: str | None = None,
        jitter_buffer_ms: tuple | None = None,
        net_interface: str | None = None,
        sip_port: int | None = None,
        expose_headers: tuple = (),
        max_concurrent_calls: int | None = 2,
        rtp_timeout: int = 0,
        instance_id: str | None = None,
        video_size: tuple = (640, 480),
        video_fps: float = 30.0,
        video_bitrate: int = 1_000_000,
        native_log_level: str = "warning",
        sip_trace: bool = False,
        extra_config: tuple | None = None,
        route_inbound: bool = True,
        **kwargs,
    ):
        """Initialize the connection.

        Args:
            user: The user part of ``sip:user@domain``.
            domain: Registration domain; may carry a port.
            password: Authentication password; may be empty.
            auth_user: Digest username when the credential store keys it
                differently from ``user`` (credential-list trunks).
            transport: SIP transport: "udp", "tcp", or "tls". Over TLS the
                server's certificate is verified: it has to chain to a
                trusted CA and name the host connected to, or registration
                and calls fail. For a server with a private or self-signed
                certificate, point the ``SIP_CA_FILE`` environment variable
                at its CA certificate; a server reached by IP address needs
                a certificate that carries that address.
            registrar: Outbound proxy to register through, when that is
                not the domain itself.
            reg_interval: Seconds between registration refreshes. The
                stack refreshes at about 90% of the interval, and over
                TCP or TLS that refresh is the only traffic on the
                connection inbound calls arrive on. The default, 240,
                refreshes every 216 s, inside the idle timeouts of the
                common cloud NAT gateways (AWS 350 s, Azure 240 s), so
                the connection is not dropped between calls. 0
                disables registration entirely (trunk mode): the
                connection dials directly and never registers. Without
                a registration binding, dial-in requires the peer or
                provider to reach this host's listening socket directly
                — see the trunk-mode notes in
                :mod:`pipecat.transports.sip.transport`.
            audio_codecs: Codec preference order by stack name; None uses
                the binding's default.
            dtmf_mode: How DTMF is sent: "rtpevent", "info", or "auto".
            media_encryption: Whether call media is encrypted, a
                :class:`~pipecat.transports.sip.media_encryption.SIPMediaEncryption`.
                ``NONE`` (the default) places unencrypted calls and accepts
                a received call either way, encrypting it when the caller
                offers SRTP. ``SRTP`` encrypts every call and refuses a
                received one that offers none. :attr:`media_encrypted`
                reports what the active call ended up with.
            extra_params: Extra baresip account parameters, each a
                ``"key=value"`` string, appended verbatim to the SIP
                address-of-record — the escape hatch for account directives
                pipecat does not model. Media-NAT traversal is the common
                case: ``("medianat=stun", "stunserver=stun:HOST:PORT")``
                behind an endpoint-independent NAT, ``medianat=turn`` behind
                a symmetric one — see the NAT notes on this class for which
                one a deployment needs (``stunserver`` has no effect without
                ``medianat``). Values are passed through unchanged; None
                (the default) adds nothing.
            jitter_buffer_mode: The receive jitter buffer: ``"off"``,
                ``"fixed"``, or ``"adaptive"``; None (the default), with
                no range, selects :data:`DEFAULT_JITTER_BUFFER` — a fixed
                40–60 ms buffer. The stack's own compiled setting, a fixed
                100–200 ms, is ``("fixed", (100, 200))``. Media waits here
                before the transport reads it, so the setting is turn
                latency. A fixed buffer holds exactly its minimum
                on every packet (the maximum is unused) and returns to it
                after any disturbance; an adaptive one starts at its
                minimum, widens once packets miss their deadline, and does
                not narrow again for the rest of the call. A packet
                arriving later than the minimum plus one packet time is
                discarded, so the minimum is the jitter tolerance. This is
                the only receive buffer in the transport's path: the
                stack's playout buffer sits downstream of the audio tap.
                Runtime-wide.
            jitter_buffer_ms: ``(min, max)`` in milliseconds for a fixed or
                adaptive buffer; given alone it selects ``"fixed"``. Not
                accepted with ``"off"``. Runtime-wide.
            net_interface: Restrict the stack to one local interface, by
                name or address. None (the default) is correct for
                production: the stack sees every interface and the OS
                routing table picks the source address per destination.
                Set it only to force a specific egress — ``"127.0.0.1"``
                for loopback targets, or one address of a multi-homed
                host. This is a restriction filter, not a listen address:
                a value like ``"0.0.0.0"`` matches no real interface and
                breaks all address selection. Runtime-wide.
            sip_port: The local SIP port, e.g. ``5060``. The stack binds it
                on each interface it uses, so ``net_interface`` decides
                the addresses. UDP and TCP listen on this port. The stack
                also opens the next port for TLS (``5060`` gives
                ``5061``), but it has no server certificate, so a TLS
                connection from a peer fails its handshake there: a fixed
                port cannot take inbound calls over TLS. None (the
                default) binds an ephemeral port, which is correct for a
                registering connection: the registration tells the server
                where the bot is. Set it for trunk mode (``reg_interval=0``)
                behind a carrier that routes INVITEs to a fixed
                ``host:port``. Interface discovery skips loopback, so a
                test with both ends on one machine also needs
                ``net_interface="127.0.0.1"``. A fixed port does not cross
                NAT: the host still needs a public address or a port
                forward. One process holds the port, so bots that share a
                host need distinct ports. Runtime-wide.
            expose_headers: SIP header names whose values ride on call
                events. Runtime-wide.
            max_concurrent_calls: Stack-wide simultaneous-call cap;
                further inbound INVITEs are refused with 486. None means
                unlimited. Runtime-wide.
            rtp_timeout: Seconds without received RTP after which a call
                is declared dead and closed (close reason
                ``"rtp stream error"``) — catches peers that vanish
                without a BYE. Direction-aware: held and send-only
                streams are not checked. 0 (the default) disables
                detection. Runtime-wide.
            instance_id: A canonical lowercase UUID identifying this
                endpoint across restarts, carried on registration
                Contacts as ``+sip.instance`` (RFC 5626): a supporting
                registrar replaces a restarted bot's stale binding
                instead of stacking a new one. Supply the same value
                every run; None sends no instance parameter.
                Runtime-wide.
            video_size: Video geometry for both directions. Runtime-wide.
            video_fps: Transmit frame pacing. Runtime-wide.
            video_bitrate: VP8 encoder target in bits/second. Runtime-wide.
            native_log_level: Lowest severity captured from the native SIP
                stack's own logging ("debug", "info", "warning", or
                "error"); records surface in pipecat's log output.
                Runtime-wide.
            sip_trace: Log every SIP message sent and received, verbatim —
                the tool for signaling and SDP/media-path debugging. The
                trace contains call metadata and digest authentication
                material, so keep it off in production. Runtime-wide.
            extra_config: Extra lines for the SIP stack's configuration,
                each a ``"key value"`` string in baresip's own config
                syntax, appended verbatim after the lines the other
                arguments produce — the escape hatch for stack settings
                pipecat does not model. The stack reads the first line for
                a setting, so a line for one that an argument here already
                sets is ignored: use the argument. One use is a development server
                whose TLS certificate cannot be verified:
                ``("sip_verify_server no",)`` turns verification off for
                every TLS connection, so never set it for production
                traffic; a private or self-signed certificate is better
                served by pointing ``SIP_CA_FILE`` at its CA. None (the
                default) adds nothing. Runtime-wide.
            route_inbound: Whether the shared inbound dispatcher routes
                incoming calls to this connection (per-connection, not
                runtime-wide). A consult/transfer leg sets ``False`` so an
                inbound INVITE is never handed to it while it sits idle
                between :meth:`connect` and :meth:`dial`; see
                :meth:`consult_connection`.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(**kwargs)
        account_args: dict = dict(
            user=user,
            domain=domain,
            password=password,
            registrar=registrar,
            reg_interval=reg_interval,
            transport=transport,
            dtmf_mode=dtmf_mode,
            auth_user=auth_user,
        )
        media_encryption = SIPMediaEncryption.parse(media_encryption)
        if audio_codecs is not None:
            account_args["audio_codecs"] = tuple(audio_codecs)
        if media_encryption is not SIPMediaEncryption.NONE:
            account_args["media_encryption"] = media_encryption.value
        if extra_params is not None:
            account_args["extra_params"] = tuple(extra_params)
        self._account = Account(**account_args)
        # The account-defining arguments, kept so consult_connection() can build a
        # sibling on the same account (reg_interval is overridden there).
        self._account_kwargs: dict = dict(
            user=user,
            domain=domain,
            password=password,
            auth_user=auth_user,
            transport=transport,
            registrar=registrar,
            audio_codecs=tuple(audio_codecs) if audio_codecs is not None else None,
            dtmf_mode=dtmf_mode,
            media_encryption=media_encryption,
            extra_params=tuple(extra_params) if extra_params is not None else None,
        )
        jitter_buffer_mode, jitter_buffer_ms = _normalize_jitter_buffer(
            jitter_buffer_mode, jitter_buffer_ms
        )
        self._settings = _RuntimeSettings(
            net_interface=net_interface,
            sip_port=_validate_sip_port(sip_port),
            rtp_timeout=rtp_timeout,
            instance_id=instance_id,
            expose_headers=tuple(expose_headers),
            max_concurrent_calls=max_concurrent_calls,
            video_size=tuple(video_size),
            video_fps=video_fps,
            video_bitrate=video_bitrate,
            native_log_level=native_log_level,
            sip_trace=sip_trace,
            jitter_buffer_mode=jitter_buffer_mode,
            jitter_buffer_ms=jitter_buffer_ms,
            extra_config=_validate_extra_config(extra_config),
        )
        # Per-connection: a consult/transfer leg opts out of inbound routing so a
        # call never lands on it while it is briefly idle between connect and dial.
        self._route_inbound = route_inbound
        self._runtime: Runtime | None = None
        self._ua: UserAgent | None = None
        self._connected = False
        self._call: Call | None = None
        self._call_incoming = False
        # Claims the connection while dial() awaits the stack, so
        # take_incoming() can't hand an inbound INVITE to it mid-dial.
        self._dialing = False
        # Set by hangup(); a close it caused is not a call failure.
        self._hangup_requested = False
        # Headers for the BYE that ends the active call, set by
        # set_hangup_headers(); per call (reset in _attach_call).
        self._hangup_headers: dict | None = None
        # "Connect" for the transport means media is up — the call is answered
        # AND audio RTP flows — not merely the SIP handshake. These track that,
        # per call (reset in _attach_call).
        self._call_established = False
        self._answered = False
        self._audio_rtp = False
        self._final_stats = None
        self._establish_task: asyncio.Task | None = None
        self._call_listener = None
        self._dtmf_listener = None
        self._info_listener = None
        self._warning_listener = None
        self._emit_tasks: set = set()

        self._register_event_handler("connected")
        self._register_event_handler("disconnected")
        self._register_event_handler("registered")
        self._register_event_handler("incoming")
        self._register_event_handler("remote_hold")
        self._register_event_handler("renegotiated")
        self._register_event_handler("call_progress")
        self._register_event_handler("call_established")
        self._register_event_handler("call_closed")
        self._register_event_handler("call_failed")
        self._register_event_handler("dtmf")
        self._register_event_handler("info")
        self._register_event_handler("audio_warning")
        self._register_event_handler("media_restarted")

    @property
    def aor(self) -> str:
        """The connection's address-of-record, ``sip:user@domain``."""
        return f"sip:{self._account.user}@{self._account.domain}"

    @property
    def is_connected(self) -> bool:
        """Whether the connection is attached to the running SIP stack."""
        return self._connected

    @property
    def has_active_call(self) -> bool:
        """Whether a call currently occupies this connection."""
        return self._call is not None

    @property
    def session_id(self) -> str | None:
        """The active call's id, or None. Stable for the call's lifetime."""
        return str(self._call.handle) if self._call is not None else None

    @property
    def final_stats(self):
        """The last ended call's media statistics, or None.

        The binding's ``CallStats`` snapshot, captured when the call
        closed. RTCP-derived fields can trail the call's end by up to one
        reporting interval.
        """
        return self._final_stats

    def configure_video(self, size: tuple, fps: float):
        """Align the connection's video geometry, before it connects.

        Called by the transport so the stack's video size and pacing
        always match the transport params. Once the shared runtime is up
        the geometry is fixed; a conflicting late configuration raises.

        Args:
            size: Video geometry as ``(width, height)``.
            fps: Transmit frame pacing in frames per second.

        Raises:
            RuntimeError: The connection is already attached to the
                stack with different video settings.
        """
        size = tuple(size)
        if size == self._settings.video_size and fps == self._settings.video_fps:
            return
        if self._connected:
            raise RuntimeError(
                "video geometry is fixed once connected: the stack runs "
                f"{self._settings.video_size}@{self._settings.video_fps}, "
                f"requested {size}@{fps}. Construct SIPConnection(video_size=..., "
                "video_fps=...) to match the transport params."
            )
        self._settings = replace(self._settings, video_size=size, video_fps=fps)

    @acquires("connection")
    async def connect(self):
        """Attach to the SIP stack and register the account.

        The first connection in the process starts the stack with this
        connection's runtime-wide settings. Registration is skipped for
        trunk-mode accounts (``reg_interval=0``); otherwise the
        ``registered`` event fires on success.

        Safe to call from both transport halves: the work runs once. A
        failed connect is terminal for the instance — later calls
        re-raise the same error rather than retrying; construct a new
        connection to retry.
        """
        self._runtime = await _SHARED.acquire(self._settings)
        # bind_ua refuses (account mismatch, UA creation failure) before counting
        # this connection as a user, so it is undone with release() alone; from
        # registration onward the binding exists and unbind() must undo it too.
        try:
            ua = await _SHARED.bind_ua(self._account, self)
        except BaseException:
            await _SHARED.release()
            self._runtime = None
            raise
        self._ua = ua
        try:
            if self._account.reg_interval != 0:
                await ua.register()
                await self._call_event_handler("registered", self.aor)
        except BaseException:
            await _SHARED.unbind(self.aor, self)
            await _SHARED.release()
            self._runtime = None
            self._ua = None
            raise
        self._connected = True
        await self._call_event_handler("connected")

    @releases("connection")
    async def disconnect(self):
        """Detach from the SIP stack, ending any active call.

        The last connection on the account unregisters it; the last
        connection in the process closes the stack. Safe to call from both
        transport halves: the work runs once, on the last release.
        """
        self._connected = False
        # A still-ringing dial's establishment watcher must not fire
        # call_failed after teardown. (On a natural close the watcher
        # finishes by itself and delivers call_failed first.)
        if self._establish_task is not None:
            self._establish_task.cancel()
            self._establish_task = None
        call = self._call
        if call is not None:
            try:
                await self._hangup_call(call)
            except ValueError as e:
                # The stack refused the hang-up headers. Teardown must still
                # finish and the far end must still be released, so the call
                # ends without them.
                logger.warning(f"{self} hang-up headers were not sent: {e}")
                try:
                    await call.hangup()
                except BaresipError as bare_error:
                    logger.debug(f"{self} hangup on disconnect: {bare_error}")
            except BaresipError as e:
                logger.debug(f"{self} hangup on disconnect: {e}")
            self._detach_call()
        await _SHARED.unbind(self.aor, self)
        await _SHARED.release()
        self._runtime = None
        self._ua = None
        await self._call_event_handler("disconnected")

    async def dial(self, uri: str, headers: dict | None = None, video: bool | str = False) -> str:
        """Start an outbound call.

        Returns as soon as the INVITE is on its way; the outcome arrives
        as events — ``call_progress``, then ``call_established``, or
        ``call_failed``.

        Args:
            uri: The SIP URI to call.
            headers: Extra headers for the INVITE.
            video: Offer video (VP8). ``True`` offers sendrecv; a
                direction string offers that direction —
                ``"inactive"`` negotiates the video stream without
                activating it, so :meth:`set_video_direction` can bring
                video up mid-call. ``False`` dials with no video stream
                at all (it can never be added later).

        Returns:
            The new call's session id.

        Raises:
            RuntimeError: Not connected, or a call is already active.
        """
        if not self._connected or self._ua is None:
            raise RuntimeError("SIPConnection is not connected")
        if self._call is not None or self._dialing:
            raise RuntimeError("SIPConnection already has an active call")
        self._dialing = True
        try:
            call = await self._ua.dial(uri, headers=headers, video=video)
            self._attach_call(call, incoming=False)
        finally:
            self._dialing = False
        self._establish_task = asyncio.create_task(self._watch_established(call))
        return str(call.handle)

    def take_incoming(self, call: Call) -> bool:
        """Claim an inbound call for this connection, if it is idle.

        Called synchronously by the shared inbound dispatcher so exactly
        one connection claims each call. Fires the ``incoming`` event;
        the handler answers or rejects.

        Returns:
            True when this connection claimed the call.
        """
        if not self._connected or self._call is not None or self._dialing:
            return False
        self._attach_call(call, incoming=True)
        self._emit("incoming", self._call_payload())
        return True

    async def answer(self, video: bool = False, headers: dict | None = None):
        """Accept the active inbound call; establishment arrives as an event.

        Args:
            video: Accept with video (VP8).
            headers: Extra headers for the 200 OK. Not supported by the
                binding yet (the stack cannot attach custom headers to
                responses); passing them raises the binding's
                ``UnsupportedFeatureError`` until that lands.
        """
        call = self._require_call()
        await call.answer(video=video, headers=headers)
        # Inbound "answered": we have sent the 200 OK (the outbound analog is
        # the CALL_ANSWERED event). Connect still waits for audio RTP.
        self._answered = True
        self._maybe_emit_connected()

    async def wait_established(self, timeout: float = 30.0):
        """Wait until the active call is SIP-established (the far end answered).

        This is the signaling answer (``CALL_ESTABLISHED``), not media-up: a
        consult leg is splice-ready the moment the far end answers, before any
        audio flows. Used to confirm a transfer target picked up before
        :meth:`attended_transfer`.

        Args:
            timeout: Seconds to wait for establishment.

        Raises:
            CallTimeout: The call did not establish within ``timeout``.
            BaresipError: The far end refused or the call otherwise failed
                (``CallBusy``, ``CallDeclined``, ``CallFailed``).
        """
        call = self._require_call()
        await call.wait_established(timeout)

    async def reject(
        self, status: int = 486, reason: str | None = None, headers: dict | None = None
    ):
        """Refuse the active inbound call before it is answered.

        The status tells the far end what to do next, and switches and
        contact-center platforms act on it: 486 (the default) says this
        endpoint is busy, 480 that it is unavailable for now, 503 that the
        call should be tried elsewhere, 603 that it is declined. With no
        arguments the call is refused with ``486 Busy Here``.

        Example::

            @connection.event_handler("incoming")
            async def on_incoming(connection, data):
                if overloaded():
                    await connection.reject(503, headers={"Retry-After": "30"})

        Args:
            status: The SIP status of the refusal, 300 to 699.
            reason: The reason phrase; None sends the standard phrase for
                the status.
            headers: Extra headers for the refusal, name to value.

        Raises:
            ValueError: A status outside 300-699, or a header name or
                value that cannot go on the wire.
        """
        call = self._require_call()
        await call.reject(status=status, reason=reason, headers=headers)

    async def hangup(self):
        """Hang up the active call; valid in any state.

        The BYE carries the headers given to :meth:`set_hangup_headers`,
        if any.
        """
        call = self._require_call()
        # Local hangup of a still-ringing dial is a stop, not a failure;
        # the establishment watcher stays quiet on the close it causes.
        self._hangup_requested = True
        await self._hangup_call(call)

    def set_hangup_headers(self, headers: dict | None):
        """Set the custom headers for the BYE that ends the active call.

        A contact-center platform reads its routing instructions from the
        BYE the bot sends: where the caller goes next, and why. The headers
        are stored and sent whenever this side ends the call, by
        :meth:`hangup` or by :meth:`disconnect` (which is how a pipeline's
        ``EndFrame`` ends it). Call it at any time once the call is
        answered; each call replaces the headers set before.

        The headers belong to the active call only and are dropped when it
        ends. When the far end hangs up first there is no BYE from this
        side, and they are not sent.

        Example::

            connection.set_hangup_headers({"X-RouteType": "queue", "X-RouteValue": "billing"})

        Args:
            headers: Header names to values, or None to send none.

        Raises:
            RuntimeError: No call is active, or it is not answered yet.
            ValueError: A header name or value that cannot go on the wire.
        """
        self._require_call()
        if not self._answered:
            raise RuntimeError("SIPConnection: hang-up headers need an answered call")
        self._hangup_headers = _validate_headers(headers)

    async def send_dtmf(self, digits: str):
        """Send DTMF digits on the active call."""
        call = self._require_call()
        await call.send_dtmf(digits)

    async def hold(self):
        """Put the active call on hold."""
        call = self._require_call()
        await call.hold()

    async def resume(self):
        """Resume the active call from hold."""
        call = self._require_call()
        await call.resume()

    async def transfer(self, uri: str, headers: dict | None = None):
        """Blind-transfer the active call (REFER).

        On success the call closes — that is the protocol, not an error.

        Args:
            uri: The SIP URI the far end should call.
            headers: Context for the party the far end calls, such as
                ``{"User-to-User": "56a390f3;encoding=hex"}`` or a case
                number in an ``X-`` header. They travel attached to
                ``uri``, escaped, and the far end sets them on the call it
                places; whether it does is the far end's choice. None
                sends the bare URI.

        Raises:
            ValueError: A header with an invalid name or value, or one
                that defines the call itself (From, To, Call-ID, Via,
                Contact and the like).
            TransferFailed: The far end refused or the transfer failed.
        """
        call = self._require_call()
        if headers:
            await call.transfer(uri, headers=headers)
        else:
            await call.transfer(uri)

    async def attended_transfer(self, consult: "SIPConnection"):
        """Splice the active call together with a consult call.

        Args:
            consult: The connection holding the consultation leg.
        """
        call = self._require_call()
        consult_call = consult._call
        if consult_call is None:
            raise RuntimeError("consult SIPConnection has no active call")
        await call.attended_transfer(consult_call)

    def consult_connection(self) -> "SIPConnection":
        """A second connection on this account for a transfer/consult leg.

        Registration-less (``reg_interval=0``): the shared runtime caches one
        UserAgent per address-of-record, so the sibling reuses this
        connection's already-registered UA to place the outbound leg — no
        second binding to clobber the first. Both calls live in the one
        runtime, which is all a Replaces splice needs (the stack resolves
        call handles globally, not per UA). It also opts out of inbound
        routing (``route_inbound=False``) so an incoming call is never handed
        to the consult leg while it is idle between connect and dial.

        The caller owns the returned connection's lifecycle: connect and dial
        it, and disconnect it when done. See
        :meth:`~pipecat.transports.sip.transport.SIPTransport.sip_attended_transfer`.
        """
        sibling = SIPConnection(reg_interval=0, route_inbound=False, **self._account_kwargs)
        # The sibling must present the SAME runtime-wide settings, or acquiring
        # the already-running runtime raises on a mismatch; share them verbatim.
        sibling._settings = self._settings
        return sibling

    def read_audio(self, max_bytes: int) -> bytes:
        """Read received PCM from the active call.

        Non-blocking. Returns up to ``max_bytes`` of received PCM —
        whatever has arrived, not all-or-nothing — and empty bytes when
        there is no call or nothing has arrived yet. A renegotiation
        surfaces as the ``media_restarted`` event and an empty read.
        """
        call = self._call
        if call is None:
            return b""
        try:
            return call.audio.read(max_bytes)
        except AudioRestarted:
            self._emit_media_restarted("audio")
            return b""
        except AudioNotActive:
            return b""

    def write_audio(self, pcm: bytes) -> int:
        """Queue PCM for transmission on the active call.

        Non-blocking. Returns the number of bytes accepted — less than
        ``len(pcm)`` when the transmit buffer is full, in which case the
        remainder was not taken and should be offered again once the
        transmit clock drains the buffer. Returns 0 with no call,
        transmit not up, or a renegotiation in progress (which also
        fires ``media_restarted``).
        """
        call = self._call
        if call is None:
            return 0
        try:
            return call.audio.write(pcm)
        except AudioRestarted:
            self._emit_media_restarted("audio")
            return 0
        except AudioNotActive:
            return 0

    def flush_tx(self):
        """Discard audio that is written but not yet transmitted.

        The binding drains the transmit buffer on its next pacing tick,
        so at most one frame (~20 ms) can still reach the wire. Quiet
        with no call or no audio — flushing nothing is not an error.
        """
        call = self._call
        if call is None:
            return
        try:
            call.audio.flush_tx()
        except AudioRestarted:
            self._emit_media_restarted("audio")
        except AudioNotActive:
            pass

    async def set_video_direction(self, direction: str):
        """Change the active call's video direction with a re-INVITE.

        ``"sendrecv"`` brings video up mid-call, ``"inactive"`` takes it
        down; the call must carry a video stream (dialed or answered
        with video). Right after establishment the stack may refuse
        with "retry shortly" while the ACK is still in flight.

        Raises:
            RuntimeError: No active call.
            Exception: The binding's error, including the
                retry-shortly case and calls without a video stream.
        """
        call = self._call
        if call is None:
            raise RuntimeError("SIPConnection has no active call")
        await call.set_video_direction(direction)

    def audio_info(self):
        """The active call's audio stream info, or None.

        The binding's ``AudioInfo``: per-direction readiness, negotiated
        sample rates, and buffer levels.
        """
        call = self._call
        if call is None:
            return None
        try:
            return call.audio.info()
        except AudioRestarted:
            self._emit_media_restarted("audio")
            return None
        except AudioNotActive:
            return None

    def read_video_frame(self):
        """The next decoded video frame from the active call, or None.

        Non-blocking; a slow reader is skipped forward to the newest
        frame. Returns None when video is not active on the call.
        """
        call = self._call
        if call is None:
            return None
        try:
            return call.video.read_frame()
        except VideoRestarted:
            self._emit_media_restarted("video")
            return None
        except VideoNotActive:
            return None

    def write_video_frame(self, i420: bytes) -> bool:
        """Queue one packed I420 frame for transmission.

        Non-blocking. Returns False when the frame was refused — no call,
        video not active, or a renegotiation gap.
        """
        call = self._call
        if call is None:
            return False
        try:
            return call.video.write_frame(i420)
        except VideoRestarted:
            self._emit_media_restarted("video")
            return False
        except VideoNotActive:
            return False

    async def request_keyframe(self):
        """Ask the far end for a keyframe; a no-op without active video."""
        call = self._call
        if call is None:
            return
        try:
            await call.video.request_keyframe()
        except (VideoNotActive, VideoRestarted):
            pass

    def _require_call(self) -> Call:
        if self._call is None:
            raise RuntimeError("SIPConnection has no active call")
        return self._call

    async def _hangup_call(self, call: Call):
        """Hang up ``call``, with the headers from :meth:`set_hangup_headers`."""
        if self._hangup_headers:
            await call.hangup(headers=self._hangup_headers)
        else:
            await call.hangup()

    @property
    def dtmf_mode(self) -> str:
        """How this connection sends DTMF: "rtpevent", "info", or "auto".

        Fixed at construction — a SIP account property, not a per-send
        choice.
        """
        return self._account.dtmf_mode

    @property
    def media_encrypted(self) -> bool:
        """Whether the active call's audio is encrypted with SRTP.

        True once the stack reports the audio stream secured, which happens
        as the call is established; False before that, on an unencrypted
        call, and without a call.
        """
        return self._call is not None and bool(self._call.media_encrypted)

    @property
    def call_direction(self) -> str | None:
        """``"in"`` or ``"out"`` for the active call, None without one."""
        if self._call is None:
            return None
        return "in" if self._call_incoming else "out"

    def _call_payload(self, **extra) -> dict:
        call = self._call
        payload: dict = {"sessionId": str(call.handle) if call else None}
        if call is not None:
            payload["sipCallId"] = call.call_id
            payload["direction"] = "in" if self._call_incoming else "out"
            if self._call_incoming:
                payload["sipFrom"] = call.peer
                payload["sipTo"] = self.aor
                payload["displayName"] = None
                payload["sipHeaders"] = dict(call.headers)
            else:
                payload["origin"] = self.aor
                payload["destination"] = call.peer
        payload.update(extra)
        return payload

    def _attach_call(self, call: Call, incoming: bool):
        self._call = call
        self._call_incoming = incoming
        self._call_established = False
        self._answered = False
        self._audio_rtp = False
        self._hangup_requested = False
        self._hangup_headers = None

        def on_event(event: StackEvent):
            self._on_call_event(call, event)

        # Capture direction and session id now: _emit runs handlers as
        # tasks, and the call may have detached by the time one executes.
        direction = "in" if incoming else "out"
        session_id = str(call.handle)

        def on_dtmf(digit_event):
            self._emit("dtmf", digit_event, session_id)

        def on_info(info_event):
            payload = {
                "sessionId": session_id,
                "contentType": info_event.content_type,
                "body": info_event.body,
            }
            self._emit("info", payload)

        def on_warning(warning):
            self._emit("audio_warning", warning, direction, session_id)

        self._call_listener = on_event
        self._dtmf_listener = on_dtmf
        self._info_listener = on_info
        self._warning_listener = on_warning
        call.on(on_event)
        call.on_dtmf(on_dtmf)
        call.on_info(on_info)
        call.on_audio_warning(on_warning)

    def _detach_call(self):
        call = self._call
        if call is None:
            return
        if self._call_listener is not None:
            call.off(self._call_listener)
        if self._dtmf_listener is not None:
            call.off_dtmf(self._dtmf_listener)
        if self._info_listener is not None:
            call.off_info(self._info_listener)
        if self._warning_listener is not None:
            call.off_audio_warning(self._warning_listener)
        self._call = None
        self._hangup_headers = None
        self._call_listener = None
        self._dtmf_listener = None
        self._info_listener = None
        self._warning_listener = None

    def _on_call_event(self, call: Call, event: StackEvent):
        """Relay one stack event for the active call; runs synchronously."""
        if self._call is not call:
            return
        if event.event is Event.CALL_RINGING or event.event is Event.CALL_PROGRESS:
            self._emit("call_progress", self._call_payload())
        elif event.event is Event.CALL_ANSWERED:
            # Outbound: the peer answered (200 OK). Inbound sets this in
            # answer() instead — CALL_ANSWERED is the outbound analog.
            logger.debug("SIP: call answered (CALL_ANSWERED)")
            self._answered = True
            self._maybe_emit_connected()
        elif event.event is Event.CALL_RTPESTAB:
            # RTP established, fired once per stream with the media name — a
            # voice bot's media path is up when audio RTP flows.
            logger.debug(f"SIP: RTP established for {event.text!r}")
            if (event.text or "") == "audio":
                self._audio_rtp = True
                self._maybe_emit_connected()
        elif event.event is Event.CALL_ESTABLISHED:
            # SIP handshake complete. Not the connect trigger — connect gates
            # on media (see _maybe_emit_connected); logged for diagnostics.
            logger.debug("SIP: SIP handshake established (CALL_ESTABLISHED)")
        elif event.event is Event.CALL_HOLD:
            self._emit("remote_hold", self._call_payload(on=True))
        elif event.event is Event.CALL_RESUME:
            self._emit("remote_hold", self._call_payload(on=False))
        elif event.event is Event.CALL_REMOTE_SDP:
            # Renegotiation traffic: an incoming re-INVITE offer or the
            # peer's answer to ours (mid-call video, direction changes).
            # Only meaningful once connected — the initial offer and
            # answer fire this too and are not "updates".
            if self._call_established:
                self._emit("renegotiated", self._call_payload(sdp=event.text or ""))
        elif event.event is Event.CALL_CLOSED:
            self._final_stats = call.final_stats
            reason = event.text or ""
            payload = self._call_payload(
                reason=reason,
                established=self._call_established,
                transferred=reason == _TRANSFER_CLOSE_REASON,
            )
            self._detach_call()
            self._emit("call_closed", payload)

    def _maybe_emit_connected(self):
        """Emit ``call_established`` once media is up, exactly once.

        "Connected" for the transport means the media path is live, not just
        that the SIP handshake finished: it fires on the later of the call being
        answered (``CALL_ANSWERED``, or our own :meth:`answer` for an inbound
        call) and the first audio RTP (``CALL_RTPESTAB``). Gating here on audio
        RTP keeps the bot's greeting from clipping into a media path that is not
        yet up, and gating on *answered* keeps an outbound bot from greeting
        over ringback / early media. Because it does not require the SIP
        handshake, a call whose ACK never arrives but whose media flows still
        signals connect — so its paired disconnect fires on close.
        """
        if self._call_established or not (self._answered and self._audio_rtp):
            return
        self._call_established = True
        logger.debug("SIP: media connected (answered + audio RTP)")
        self._emit("call_established", self._call_payload())

    async def _watch_established(self, call: Call):
        """Turn an outbound call's typed failure into the call_failed event."""
        try:
            await call.wait_established()
        except BaresipError as e:
            if self._hangup_requested:
                return
            await self._call_event_handler(
                "call_failed",
                {
                    "sessionId": str(call.handle),
                    "error": type(e).__name__,
                    "message": str(e),
                },
            )

    def _emit_media_restarted(self, kind: str):
        self._emit("media_restarted", kind)

    def _emit(self, event_name: str, *args):
        """Fire an event from synchronous code, keeping the task alive."""
        task = asyncio.create_task(self._call_event_handler(event_name, *args))
        self._emit_tasks.add(task)
        task.add_done_callback(self._emit_tasks.discard)

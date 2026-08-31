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
  and shared: any number of connections on the same account reuse it.
- **N connections per address-of-record, one call per connection.** A
  connection is a call slot. Each inbound INVITE is routed to the first
  idle connection bound to the account — the claim is synchronous, so
  two INVITEs can never land on one connection — and refused with 486
  when every connection is busy (the stack's own
  ``max_concurrent_calls`` cap refuses even earlier).

So "one registration, N concurrent conversations" is spelled: one
process, one runtime, one user agent for the account, and N connections
— typically one per pipeline/transport.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from loguru import logger

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


@dataclass(frozen=True)
class _RuntimeSettings:
    """Runtime-wide baresip settings, fixed by the first connection.

    The SIP stack is one per process, so these knobs cannot vary between
    connections; a later connection asking for different values is a
    configuration error, not a preference.
    """

    net_interface: str | None = None
    expose_headers: tuple = ()
    max_concurrent_calls: int | None = 2
    video_size: tuple = (640, 480)
    video_fps: float = 30.0
    video_bitrate: int = 1_000_000

    def render(self) -> str:
        """Render these settings into baresip runtime configuration text."""
        config = Config(
            expose_headers=self.expose_headers,
            max_concurrent_calls=self.max_concurrent_calls,
            video_size=self.video_size,
            video_fps=self.video_fps,
            video_bitrate=self.video_bitrate,
        ).render()
        if self.net_interface:
            config = f"net_interface {self.net_interface}\n" + config
        return config


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
                runtime = Runtime()
                await runtime.start(settings.render())
                self._runtime = runtime
                self._settings = settings
            elif settings != self._settings:
                raise ValueError(
                    "the SIP runtime is already up with different runtime-wide "
                    f"settings ({self._settings!r}); all SIPConnections in a process "
                    "must agree on net_interface, expose_headers, "
                    "max_concurrent_calls, and the video parameters"
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
            self._bindings.clear()
        if runtime is not None:
            await runtime.close()

    async def bind_ua(self, account: Account, connection: "SIPConnection") -> UserAgent:
        """Get the account's user agent and route its inbound calls.

        The first binding for an address-of-record allocates the user
        agent and installs the inbound dispatcher; later bindings share
        it. An inbound call goes to the first bound connection without an
        active call, and is rejected when every connection is busy.
        """
        aor = f"sip:{account.user}@{account.domain}"
        async with self._lock:
            connections = self._bindings.setdefault(aor, [])
            if aor not in self._uas:
                ua = await UserAgent.create(self._runtime, account)
                self._uas[aor] = ua
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
            if connection not in connections:
                connections.append(connection)
            return self._uas[aor]

    async def unbind(self, aor: str, connection: "SIPConnection"):
        """Stop routing inbound calls to a connection."""
        async with self._lock:
            connections = self._bindings.get(aor)
            if connections and connection in connections:
                connections.remove(connection)


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

    Runtime-wide arguments (``net_interface``, ``expose_headers``,
    ``max_concurrent_calls``, and the video parameters) configure the
    process-wide SIP stack and are fixed by the first connection to
    connect; every later connection must pass the same values.

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
    - call_established: media is up; receives a payload dict.
    - call_closed: the call ended; receives a payload dict with
      ``reason`` and ``established``.
    - call_failed: an outbound call ended without establishing; receives
      a payload dict with ``error`` (the typed failure's name) and
      ``message``.
    - dtmf: the far end pressed a key; receives the binding's DigitEvent.
    - remote_hold: the far end put the call on hold or resumed it;
      receives a payload dict with ``on`` (bool).
    - audio_warning: the call's audio layer reported a warning.
    - media_restarted: a renegotiation replaced the media streams;
      receives ``"audio"`` or ``"video"``. Consumers should rebuild
      resamplers and re-read geometry.

    Example::

        connection = SIPConnection(user="1001", domain="example.com", password="...")

        @connection.event_handler("incoming")
        async def on_incoming(connection, data):
            await connection.answer()
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
        reg_interval: int = 600,
        audio_codecs: tuple | None = None,
        dtmf_mode: str = "rtpevent",
        net_interface: str | None = None,
        expose_headers: tuple = (),
        max_concurrent_calls: int | None = 2,
        video_size: tuple = (640, 480),
        video_fps: float = 30.0,
        video_bitrate: int = 1_000_000,
        **kwargs,
    ):
        """Initialize the connection.

        Args:
            user: The user part of ``sip:user@domain``.
            domain: Registration domain; may carry a port.
            password: Authentication password; may be empty.
            auth_user: Digest username when the credential store keys it
                differently from ``user`` (credential-list trunks).
            transport: SIP transport: "udp", "tcp", or "tls".
            registrar: Outbound proxy to register through, when that is
                not the domain itself.
            reg_interval: Seconds between registration refreshes. 0
                disables registration entirely (trunk mode): the
                connection dials directly and never registers. Without
                a registration binding, dial-in requires the peer or
                provider to reach this host's listening socket directly
                — see the trunk-mode notes in
                :mod:`pipecat.transports.sip.transport`.
            audio_codecs: Codec preference order by stack name; None uses
                the binding's default.
            dtmf_mode: How DTMF is sent: "rtpevent", "info", or "auto".
            net_interface: Restrict the stack to one local interface, by
                name or address. None (the default) is correct for
                production: the stack sees every interface and the OS
                routing table picks the source address per destination.
                Set it only to force a specific egress — ``"127.0.0.1"``
                for loopback targets, or one address of a multi-homed
                host. This is a restriction filter, not a listen address:
                a value like ``"0.0.0.0"`` matches no real interface and
                breaks all address selection. Runtime-wide.
            expose_headers: SIP header names whose values ride on call
                events. Runtime-wide.
            max_concurrent_calls: Stack-wide simultaneous-call cap;
                further inbound INVITEs are refused with 486. None means
                unlimited. Runtime-wide.
            video_size: Video geometry for both directions. Runtime-wide.
            video_fps: Transmit frame pacing. Runtime-wide.
            video_bitrate: VP8 encoder target in bits/second. Runtime-wide.
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
        if audio_codecs is not None:
            account_args["audio_codecs"] = tuple(audio_codecs)
        self._account = Account(**account_args)
        self._settings = _RuntimeSettings(
            net_interface=net_interface,
            expose_headers=tuple(expose_headers),
            max_concurrent_calls=max_concurrent_calls,
            video_size=tuple(video_size),
            video_fps=video_fps,
            video_bitrate=video_bitrate,
        )
        self._runtime: Runtime | None = None
        self._ua: UserAgent | None = None
        self._connected = False
        self._call: Call | None = None
        self._call_incoming = False
        self._call_established = False
        self._final_stats = None
        self._establish_task: asyncio.Task | None = None
        self._call_listener = None
        self._dtmf_listener = None
        self._warning_listener = None
        self._emit_tasks: set = set()

        self._register_event_handler("connected")
        self._register_event_handler("disconnected")
        self._register_event_handler("registered")
        self._register_event_handler("incoming")
        self._register_event_handler("remote_hold")
        self._register_event_handler("call_progress")
        self._register_event_handler("call_established")
        self._register_event_handler("call_closed")
        self._register_event_handler("call_failed")
        self._register_event_handler("dtmf")
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
        try:
            ua = await _SHARED.bind_ua(self._account, self)
            self._ua = ua
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

        The last connection in the process closes the stack. Safe to call
        from both transport halves: the work runs once, on the last
        release.
        """
        self._connected = False
        call = self._call
        if call is not None:
            try:
                await call.hangup()
            except BaresipError as e:
                logger.debug(f"{self} hangup on disconnect: {e}")
            self._detach_call()
        await _SHARED.unbind(self.aor, self)
        await _SHARED.release()
        self._runtime = None
        self._ua = None
        await self._call_event_handler("disconnected")

    async def dial(self, uri: str, headers: dict | None = None, video: bool = False) -> str:
        """Start an outbound call.

        Returns as soon as the INVITE is on its way; the outcome arrives
        as events — ``call_progress``, then ``call_established``, or
        ``call_failed``.

        Args:
            uri: The SIP URI to call.
            headers: Extra headers for the INVITE.
            video: Offer video (VP8).

        Returns:
            The new call's session id.

        Raises:
            RuntimeError: Not connected, or a call is already active.
        """
        if not self._connected or self._ua is None:
            raise RuntimeError("SIPConnection is not connected")
        if self._call is not None:
            raise RuntimeError("SIPConnection already has an active call")
        call = await self._ua.dial(uri, headers=headers, video=video)
        self._attach_call(call, incoming=False)
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
        if not self._connected or self._call is not None:
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

    async def reject(self):
        """Decline the active inbound call with 486 Busy Here."""
        call = self._require_call()
        await call.reject()

    async def hangup(self):
        """Hang up the active call; valid in any state."""
        call = self._require_call()
        await call.hangup()

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

    async def transfer(self, uri: str):
        """Blind-transfer the active call (REFER).

        On success the call closes — that is the protocol, not an error.

        Raises:
            TransferFailed: The far end refused or the transfer failed.
        """
        call = self._require_call()
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

    def read_audio(self, max_bytes: int) -> bytes:
        """Read received PCM from the active call.

        Non-blocking. Returns empty bytes when there is no call, no audio
        yet, or fewer bytes than requested have arrived. A renegotiation
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

    @property
    def dtmf_mode(self) -> str:
        """How this connection sends DTMF: "rtpevent", "info", or "auto".

        Fixed at construction — a SIP account property, not a per-send
        choice.
        """
        return self._account.dtmf_mode

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

        def on_event(event: StackEvent):
            self._on_call_event(call, event)

        def on_dtmf(digit_event):
            self._emit("dtmf", digit_event)

        def on_warning(warning):
            self._emit("audio_warning", warning)

        self._call_listener = on_event
        self._dtmf_listener = on_dtmf
        self._warning_listener = on_warning
        call.on(on_event)
        call.on_dtmf(on_dtmf)
        call.on_audio_warning(on_warning)

    def _detach_call(self):
        call = self._call
        if call is None:
            return
        if self._call_listener is not None:
            call.off(self._call_listener)
        if self._dtmf_listener is not None:
            call.off_dtmf(self._dtmf_listener)
        if self._warning_listener is not None:
            call.off_audio_warning(self._warning_listener)
        self._call = None
        self._call_listener = None
        self._dtmf_listener = None
        self._warning_listener = None

    def _on_call_event(self, call: Call, event: StackEvent):
        """Relay one stack event for the active call; runs synchronously."""
        if self._call is not call:
            return
        if event.event is Event.CALL_RINGING or event.event is Event.CALL_PROGRESS:
            self._emit("call_progress", self._call_payload())
        elif event.event is Event.CALL_HOLD:
            self._emit("remote_hold", self._call_payload(on=True))
        elif event.event is Event.CALL_RESUME:
            self._emit("remote_hold", self._call_payload(on=False))
        elif event.event is Event.CALL_ESTABLISHED:
            self._call_established = True
            self._emit("call_established", self._call_payload())
        elif event.event is Event.CALL_CLOSED:
            self._final_stats = call.final_stats
            payload = self._call_payload(
                reason=event.text or "", established=self._call_established
            )
            self._detach_call()
            self._emit("call_closed", payload)

    async def _watch_established(self, call: Call):
        """Turn an outbound call's typed failure into the call_failed event."""
        try:
            await call.wait_established()
        except BaresipError as e:
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

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) transport implementation for Pipecat.

Uses the upstream ``moq`` Python library
(`moq-rs <https://pypi.org/project/moq-rs/>`_) for the QUIC connection,
MOQ session, announcement discovery, subscription routing, group/frame
framing, codec-specific catalog management, and Opus encode/decode +
resampling for raw audio tracks. This module just wires it into the
pipecat Frame pipeline.

Each participant publishes under a per-participant broadcast path
``<namespace>/<participant_id>`` (e.g. ``pipecat/bot0``); the bot
subscribes to the peer at ``<namespace>/<peer_id>``. Audio rides on a
single Opus track; RTVI JSON rides on a fixed-name ``transcript.json.z``
track carried by moq's JSON stream helper (``publish_json_stream`` /
``subscribe_json_stream``). The stream is an ordered, lossless append-log
of records — every message is delivered in order, unlike the JSON
*snapshot* helper (``publish_json`` / ``subscribe_json``) which collapses
to the latest value and would drop RTVI events a slow consumer fell
behind on. Compression is enabled (hence the ``.z`` suffix). The
transcript is a side-channel, like moq-boy's ``status``/``command``
tracks: it deliberately bypasses the catalog (which only describes media
renditions), so the browser reads it by the well-known name rather than
by catalog discovery.

Both directions carry the same shape: the bot publishes bot-side RTVI
events on its own ``transcript`` track and subscribes to the client's
``transcript`` track for client-side traffic (``client-ready`` for
protocol negotiation, typed text input, function-call results, etc).
This makes MoQ a full bidirectional RTVI transport, on par with the
Daily and WebSocket transports.

Two modes:

- **Server mode** (``serve=True``, currently the only supported mode):
  the bot binds its own UDP socket via ``moq.Server`` and accepts the
  browser's direct connection. Removes the need for a separate
  ``moq-relay`` process for local dev. The self-signed cert fingerprints
  are exposed via :attr:`MOQTransport.cert_fingerprints` so a browser
  can pin them.
- **Client mode** (default): MoQ client mode is not yet supported. The
  bot would dial a relay at ``relay_url`` (or the constructor's
  ``host``/``port``/``path``); transport-level wiring exists but the
  runner blocks this mode at arg-parse time until the cert-fingerprint
  plumbing to the browser is finished and we've validated the flow
  against an external relay. See
  :func:`pipecat.runner.moq._validate_moq_args` for the guard.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    import moq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use MOQ transport, you need to `pip install pipecat-ai[moq]`.")
    raise Exception(f"Missing module: {e}")


def _is_normal_close(exc: BaseException) -> bool:
    """Return True for the MoQ session-closed error we see when the peer hangs up.

    moq-rs surfaces a normal WebTransport close (code=0) as a
    ``MoqError.Protocol`` whose message contains
    ``"webtransport error: closed"``. That's not a failure — it's the
    expected end of every session, both when the browser disconnects and
    when the bot itself tears down its own transport. We log it at info
    (and skip the ``on_error`` handler) instead of ERROR + traceback.
    """
    if not isinstance(exc, moq.MoqError):
        return False
    msg = str(exc)
    return "webtransport error: closed" in msg or "session error" in msg and "closed" in msg


_moq_task_filter_installed = False


def _install_moq_task_exception_filter() -> None:
    """Chain a loop-level exception filter that swallows normal-close leaks.

    ``moq.Server`` spawns per-session tasks internally and doesn't await
    them; when the peer WebTransport closes those tasks exit with the
    normal-close ``MoqError.Protocol`` and, on garbage collection, asyncio
    prints a scary "Task exception was never retrieved" traceback. GC can
    fire well after our ``_run`` returns, so a scoped install/restore
    around ``_run`` misses it. Install once, permanently, on first
    ``_run`` — the filter only drops ``_is_normal_close`` errors and
    delegates everything else to the previously-installed handler, so
    leaving it in place is safe and idempotent across transport instances.
    """
    global _moq_task_filter_installed
    if _moq_task_filter_installed:
        return
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def _filter(loop_, context):
        exc = context.get("exception")
        if exc is not None and _is_normal_close(exc):
            logger.debug(f"MOQ: swallowed unretrieved-task exception from a normal close: {exc}")
            return
        if previous is not None:
            previous(loop_, context)
        else:
            loop_.default_exception_handler(context)

    loop.set_exception_handler(_filter)
    _moq_task_filter_installed = True


DEFAULT_NAMESPACE = "pipecat"
DEFAULT_PARTICIPANT_ID = "bot0"
DEFAULT_PEER_ID = "client0"
DEFAULT_AUDIO_OUT_TRACK = "bot-audio"
# Fixed-name JSON side-channel track (cf. ``catalog.json``). Carries RTVI
# events as a lossless, ordered append-log via moq's JSON stream helper —
# not the JSON *snapshot* helper, which collapses to the latest value and
# would drop events a slow consumer fell behind on. Not a catalog
# rendition, so the browser subscribes by this well-known name. The ``.z``
# suffix marks that the stream is compressed on the wire.
DEFAULT_TRANSCRIPT_TRACK = "transcript.json.z"

# Pin the Opus wire rate to its highest supported internal rate (Opus
# supports {8, 12, 16, 24, 48} kHz). Chrome's WebCodecs Opus decoder
# always emits AudioData at 48kHz regardless of the
# AudioDecoderConfig.sampleRate hint, so encoding at anything else
# creates a wire/playback rate mismatch on the browser side:
# @moq/watch's ring buffer is sized off the catalog rate, and if that
# disagrees with the AudioData rate the bot voice plays back
# octave-low at half speed. moq-rs handles the resampling from the
# pipeline rate up to this for us.
OPUS_SAMPLE_RATE = 48000


def _downmix_s16_to_mono(pcm: bytes, channels: int) -> bytes:
    """Downmix interleaved S16 PCM to mono by averaging channels.

    Used to compensate for the @moq/publish browser encoder publishing
    stereo even when the source mic is mono (the encoder defaults to
    the MediaStreamAudioSourceNode's channelCount=2 when the
    MediaStreamTrack doesn't expose channelCount in its settings).
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    if channels <= 1 or samples.size % channels != 0:
        return pcm
    frames = samples.size // channels
    summed = samples.reshape(frames, channels).astype(np.int32).sum(axis=1)
    out = np.clip(summed // channels, -32768, 32767).astype(np.int16)
    return out.tobytes()


class MOQTrackType(IntEnum):
    """Types of media tracks the transport surfaces in event callbacks."""

    AUDIO = 0x01
    VIDEO = 0x02
    DATA = 0x03


@dataclass
class MOQTrack:
    """Identifies a MOQ track for event callbacks.

    Parameters:
        broadcast_path: The full broadcast path (e.g. ``pipecat/bot0``).
        name: The track name (e.g. ``bot-audio``).
        track_type: The track media type.
    """

    broadcast_path: str
    name: str
    track_type: MOQTrackType = MOQTrackType.DATA

    @property
    def full_name(self) -> str:
        """Get the full track identifier."""
        return f"{self.broadcast_path}/{self.name}"


class MOQParams(TransportParams):
    """Configuration parameters for MOQ transport.

    Parameters:
        relay_url: Full relay URL (e.g. ``https://relay.example.com:4080/moq``).
            If unset, the transport composes one from the constructor's
            ``host``/``port``/``path``. Ignored in serve mode.
        namespace: Top-level namespace shared by all participants.
        participant_id: This bot's id; the bot publishes under
            ``<namespace>/<participant_id>``.
        peer_id: The id of the peer (browser/client) the bot subscribes
            to: ``<namespace>/<peer_id>``.
        audio_out_track: Name of the bot's outgoing audio track.
        transcript_track: Name of the bot's outgoing transcript track. A
            fixed-name JSON stream track carrying RTVI messages as a
            lossless, ordered append-log (moq's ``publish_json_stream`` /
            ``subscribe_json_stream``, compression on), discovered by
            convention rather than via the catalog.
        verify_ssl: Verify the relay's TLS certificate. Client mode only.
        connection_timeout: Seconds to wait for the peer broadcast to be
            announced before giving up.
        serve: When ``True``, the bot binds its own UDP socket and accepts
            incoming MOQ sessions instead of dialing a relay.
        serve_bind: Address to bind in serve mode (e.g. ``"[::]:4080"``).
            Defaults to ``"[::]:<port>"`` based on the constructor's port.
        serve_tls_host: Hostname to use in the generated self-signed
            certificate when ``serve_tls_cert``/``serve_tls_key`` aren't
            provided. The browser pins this cert via its SHA-256
            fingerprint, so the value just needs to match what the
            browser sees in the URL (typically ``"localhost"``).
        serve_tls_cert: Path to a PEM-encoded TLS certificate chain.
            If unset alongside ``serve_tls_key``, a self-signed cert is
            generated on startup.
        serve_tls_key: Path to the PEM-encoded private key matching
            ``serve_tls_cert``.
        audio_out_sample_rate: Sample rate the bot publishes its audio
            at (Hz). Pipecat hands us audio at whatever rate the TTS
            produces; the library resamples to the nearest Opus-supported
            rate before encoding.
        audio_in_sample_rate: Sample rate the pipeline expects to receive
            user audio at (Hz). Decoded Opus is resampled to this rate
            before being pushed downstream.
        audio_in_max_latency_ms: How long :func:`subscribe_audio` will
            wait for a late frame before skipping ahead. Lower = more
            interactive (fewer fills, more drops on bad networks);
            higher = smoother audio with more glass-to-glass delay.
        audio_out_frame_ms: Opus frame duration for the bot's audio
            output. Must be 2, 5, 10, 20, 40, or 60. 20 ms is the
            real-time default.
        audio_out_max_buffer_ms: How far ahead of real-time the bot is
            allowed to write audio. The bot writes TTS faster than
            real-time with future-dated timestamps so the browser player
            (``@moq/watch``) can buffer and play at the encoded pace; this
            paces ``publish_audio`` so the in-flight buffer never grows
            past this many milliseconds. Keep it a little under the
            player's buffer ceiling (``MoqTransportOptions.audioBufferMaxMs``,
            30s) so the producer self-limits below the player's drop
            ceiling and the player never has to drop. On interruption the
            pacing clock is re-anchored (see :meth:`reset_audio_pacing`) so
            the next utterance isn't delayed by the previous buffer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    relay_url: str | None = None
    namespace: str = DEFAULT_NAMESPACE
    participant_id: str = DEFAULT_PARTICIPANT_ID
    peer_id: str = DEFAULT_PEER_ID
    audio_out_track: str = DEFAULT_AUDIO_OUT_TRACK
    transcript_track: str = DEFAULT_TRANSCRIPT_TRACK
    verify_ssl: bool = True
    connection_timeout: float = 30.0
    serve: bool = False
    serve_bind: str | None = None
    serve_tls_host: str = "localhost"
    serve_tls_cert: str | None = None
    serve_tls_key: str | None = None
    audio_out_sample_rate: int = 24000
    audio_in_sample_rate: int = 16000
    audio_in_max_latency_ms: int = 500
    audio_out_frame_ms: int = 20
    audio_out_max_buffer_ms: int = 25000


class MOQCallbacks(BaseModel):
    """Callback handlers bridging :class:`MOQTransportClient` to :class:`MOQTransport`.

    Decouples the moq-rs session/connection logic from pipecat's Frame
    pipeline: :class:`MOQTransportClient` only ever calls these, it never
    reaches into :class:`MOQInputTransport`/:class:`MOQOutputTransport` or
    pipecat frame types directly.

    Parameters:
        on_connected: Called when the MOQ session (client or server) is established.
        on_disconnected: Called when the session ends.
        on_client_connected: Called when the peer's broadcast is announced.
        on_client_disconnected: Called when the peer's broadcast goes away.
        on_track_subscribed: Called when a remote track subscription succeeds.
        on_error: Called when the underlying transport errors.
        on_audio_received: Called with decoded mono PCM audio from the peer's audio track.
        on_message_received: Called with each RTVI message from the peer's transcript stream.
    """

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_client_connected: Callable[[], Awaitable[None]]
    on_client_disconnected: Callable[[], Awaitable[None]]
    on_track_subscribed: Callable[[MOQTrack], Awaitable[None]]
    on_error: Callable[[str, Exception], Awaitable[None]]
    on_audio_received: Callable[[bytes, int], Awaitable[None]]
    on_message_received: Callable[[dict], Awaitable[None]]


class MOQTransportClient:
    """Owns the moq-rs QUIC session/connection for a single MOQ transport.

    Mirrors the ``<Provider>TransportClient`` pattern used by the other
    transports (e.g. ``DailyTransportClient``, ``SmallWebRTCClient``): it
    owns dialing/serving, the publish broadcast, and forwarding the
    peer's audio/transcript, and talks back to :class:`MOQTransport` only
    through ``callbacks`` — it has no knowledge of pipecat Frames or the
    input/output transports.
    """

    def __init__(
        self,
        params: MOQParams,
        url: str,
        serve_bind: str,
        callbacks: MOQCallbacks,
    ):
        """Initialize the MOQ transport client.

        The publish broadcast and its transcript track are created here,
        synchronously, so the output processor can begin writing into
        them before the async session bring-up in :meth:`_run` finishes
        — otherwise the bot's first ~hundreds of ms of audio gets
        silently dropped while we're still dialing the relay.

        Args:
            params: MOQ configuration parameters.
            url: Full relay URL to dial in client mode.
            serve_bind: Address to bind in serve mode.
            callbacks: Event/data callbacks back to the owning transport.
        """
        self._params = params
        self._url = url
        self._serve_bind = serve_bind
        self._callbacks = callbacks

        self._connection_task: asyncio.Task | None = None
        # Shared task manager, wired in by the input/output processors'
        # setup() forwarding. Owns the serve-loop task in _run().
        self._task_manager: BaseTaskManager | None = None

        # Owned for the lifetime of this client. Created synchronously so
        # MOQOutputTransport.start() can publish_audio + write transcript
        # frames without racing with _run()'s async bring-up.
        self._publish_broadcast: moq.BroadcastProducer = moq.BroadcastProducer()
        # Lossless, ordered JSON append-log for RTVI (compression on). The
        # stream helper delivers every appended record in order — the
        # snapshot helper (publish_json) would collapse to the latest value
        # and drop events a slow consumer fell behind on.
        self._transcript_out: moq.JsonStreamProducer = self._publish_broadcast.publish_json_stream(
            params.transcript_track, compression=True
        )
        # Audio track is opened lazily once the pipeline's output sample
        # rate is known (in :meth:`open_audio_track`). We stash the rate
        # because ``publish_audio`` uses it to convert byte length to
        # wall-clock duration for pacing.
        self._audio_out: moq.AudioProducer | None = None
        self._audio_out_sample_rate: int | None = None
        # Wall-clock target for the next publish_audio write. ``None``
        # means "reset" — the next write anchors to ``time.monotonic()``.
        self._publish_audio_clock: float | None = None
        # Monotonic presentation timestamp (microseconds) for the next
        # audio chunk, advanced by each chunk's duration. The browser
        # player uses these future-dated timestamps to buffer and pace
        # playback. Not reset on interruption — the player re-anchors on
        # its own (reset() on user-started-speaking).
        self._publish_pts_us: int = 0

        # Track consumers we created so disconnect() can cancel them.
        # Each entry has a sync .cancel() method that terminates any
        # pending await on the consumer.
        self._active_consumers: list = []

        self._cert_fingerprints: list[str] = []

        self._broadcast_path = f"{params.namespace}/{params.participant_id}"
        self._peer_broadcast_path = f"{params.namespace}/{params.peer_id}"
        self._cleaned_up = False
        # Number of input/output transports currently holding the session
        # open. Both call :meth:`connect` on start and :meth:`stop`/
        # :meth:`cancel` on shutdown; the session is only actually torn
        # down once every holder has released it (see :meth:`_release`),
        # mirroring DailyTransportClient's join/leave counter.
        self._holders = 0

    async def setup(self, setup: FrameProcessorSetup):
        """Capture the task manager from the input/output processors.

        Both processors forward their setup here; we only need to store
        the reference once since they share the same task manager.
        """
        if self._task_manager is None:
            self._task_manager = setup.task_manager

    def connect(self):
        """Register a holder and start the MOQ session on the first call.

        Called by both :meth:`MOQInputTransport.start` and
        :meth:`MOQOutputTransport.start`, after ``setup()`` has wired in
        the task manager. Only the first call actually dials/serves;
        later calls just record another holder, balanced against
        :meth:`stop`/:meth:`cancel` releasing it.
        """
        self._holders += 1
        if self._connection_task is not None:
            return
        assert self._task_manager is not None, (
            "MOQTransportClient.setup() must run before connect(); "
            "input/output processors forward setup on the pipeline's first frame."
        )
        self._connection_task = self._task_manager.create_task(self._run(), "moq_run")

    # ------------------------------------------------------------------
    # Publishing helpers — called from the output transport.
    # ------------------------------------------------------------------

    def open_audio_track(self, sample_rate: int):
        """Open the bot's audio track via ``publish_audio``.

        Called by :class:`MOQOutputTransport.start` once it knows the
        pipeline's output sample rate. No-op if the track is already
        open or audio output is disabled.
        """
        if self._audio_out is not None or not self._params.audio_out_enabled:
            return

        self._audio_out_sample_rate = sample_rate
        self._audio_out = self._publish_broadcast.publish_audio(
            self._params.audio_out_track,
            moq.AudioEncoderInput(
                format=moq.AudioFormat.S16,
                sample_rate=sample_rate,
                channels=1,
            ),
            moq.AudioEncoderOutput(
                codec=moq.AudioCodec.OPUS,
                sample_rate=OPUS_SAMPLE_RATE,
                channels=None,
                bitrate=None,
                frame_duration_ms=self._params.audio_out_frame_ms,
            ),
        )
        logger.debug(
            f"MOQ: publishing audio as Opus "
            f"(pipeline rate={sample_rate}Hz, opus rate={OPUS_SAMPLE_RATE}Hz, "
            f"frame={self._params.audio_out_frame_ms}ms, "
            f"track={self._params.audio_out_track!r})"
        )

    def reset_audio_pacing(self):
        """Re-anchor the publish_audio pacing clock to wall-clock now.

        Called from :class:`MOQOutputTransport.process_frame` on
        InterruptionFrame so the next utterance plays immediately
        instead of waiting for the (now-cancelled) previous one to
        finish in pacing time.

        Part of the publish_audio pacing workaround; goes away once
        moq-rs exposes a flush primitive on ``AudioProducer``. See
        :meth:`publish_audio`.
        """
        self._publish_audio_clock = None

    async def publish_audio(self, audio: bytes):
        """Push a PCM chunk to the bot's audio track with real PTS, paced to a cap.

        The library does the Opus encode + resample inside the FFI, so we
        just write S16 PCM bytes.

        Each chunk is stamped with a monotonic presentation timestamp so
        the browser player (``@moq/watch``) can buffer the future-dated
        frames and play them at the encoded pace. ``AudioProducer.write()``
        is fire-and-forget, so to bound how far ahead the bot runs we pace
        the writes against a virtual clock: each call advances the clock by
        the chunk's audio duration, and we sleep until wall-clock is within
        ``audio_out_max_buffer_ms`` (25s) of it. That keeps the in-flight
        buffer a little under the player's drop ceiling
        (``MoqTransportOptions.audioBufferMaxMs``, 30s), so the producer
        self-limits below the consumer's cap. Interruptions are flushed on
        the browser side (``reset()`` on ``user-started-speaking``); the
        pacing clock is re-anchored here (see :meth:`reset_audio_pacing`)
        so the next utterance isn't delayed by the previous buffer.
        """
        if self._audio_out is None or not self._audio_out_sample_rate:
            return

        now = time.monotonic()
        if self._publish_audio_clock is None or self._publish_audio_clock < now:
            self._publish_audio_clock = now

        # Sleep so we're never more than `audio_out_max_buffer_ms` ahead
        # of wall-clock. Pacing happens BEFORE the write so the moq
        # library never sees more than that much audio queued at once.
        budget = self._params.audio_out_max_buffer_ms / 1000.0
        wait = self._publish_audio_clock - now - budget
        if wait > 0:
            await asyncio.sleep(wait)

        # Advance the virtual clock and the presentation timestamp by the
        # duration of this chunk. S16 = 2 bytes/sample, mono.
        duration_s = len(audio) / (self._audio_out_sample_rate * 2)
        self._publish_audio_clock += duration_s

        self._audio_out.write(moq.AudioFrame(timestamp_us=self._publish_pts_us, data=audio))
        self._publish_pts_us += int(duration_s * 1_000_000)

    async def wait_for_audio_drain(self, jitter_buffer_margin_s: float = 0.3) -> None:
        """Block until the pacing clock catches up to wall-clock, then a bit more.

        ``publish_audio`` paces each write against ``_publish_audio_clock``,
        which tracks the presentation-time deadline of the last chunk
        written to the moq audio stream. When wall-clock reaches that
        value, the last audio chunk has (approximately) been consumed by
        the browser's player. Add a small margin for the client-side
        jitter buffer to drain past the final sample before we signal
        session-ending and tear the transport down.

        No-op when no audio has been published yet or the clock has
        already caught up. Capped at ``audio_out_max_buffer_ms`` as a
        safety net so a bogus clock value can't stall shutdown.
        """
        if self._publish_audio_clock is None:
            return
        pending = self._publish_audio_clock - time.monotonic()
        if pending <= 0:
            return
        cap = self._params.audio_out_max_buffer_ms / 1000.0
        drain = min(pending, cap) + jitter_buffer_margin_s
        logger.debug(f"MOQ: draining {drain:.2f}s of buffered outbound audio before shutdown")
        await asyncio.sleep(drain)

    def publish_transcript(self, message):
        """Append an RTVI message to the transcript JSON stream.

        ``message`` is a JSON-serializable value (the RTVI message dict);
        the stream helper serializes and frames it, appending one record
        to the ordered log so no message is dropped.
        """
        self._transcript_out.append(message)

    @property
    def cert_fingerprints(self) -> list[str]:
        """SHA-256 fingerprints (hex) of the serving cert (server mode only).

        Populated once the bot has bound; empty in client mode or
        before :meth:`_run` has reached the listen step. Useful for
        telling a browser client which self-signed cert to pin.
        """
        return list(self._cert_fingerprints)

    # ------------------------------------------------------------------
    # Connection lifecycle.
    # ------------------------------------------------------------------

    async def _run(self):
        """Drive the MOQ session for the bot's lifetime.

        Owns the lifetime of either ``moq.Client`` (client mode) or
        ``moq.Server`` (serve mode). Returns once the session closes or
        :meth:`disconnect` is called.

        Both modes share a single :class:`moq.OriginProducer`. The bot
        publishes its broadcast through the origin and consumes the
        peer's broadcast through the same origin — only the transport
        bring-up differs.
        """
        # Install the loop-level filter that swallows the normal-close
        # unretrieved-task warning from moq.Server's internal tasks.
        # Idempotent + permanent for the loop's lifetime — see the
        # helper's docstring for why we can't scope it to _run.
        _install_moq_task_exception_filter()

        origin = moq.OriginProducer()

        if self._params.serve:
            ctx_label = f"serving {self._serve_bind} as {self._broadcast_path}"
        else:
            ctx_label = f"connecting to {self._url} as {self._broadcast_path}"
        logger.debug(f"MOQ: {ctx_label}")

        try:
            async with self._make_transport(origin) as transport:
                if self._params.serve:
                    server = cast(moq.Server, transport)
                    self._cert_fingerprints = server.cert_fingerprints()
                    logger.debug(
                        f"MOQ: bound on {server.local_addr} "
                        f"(cert sha256: {self._cert_fingerprints})"
                    )

                origin.publish(self._broadcast_path, self._publish_broadcast)
                logger.debug(
                    f"MOQ: published broadcast {self._broadcast_path!r} "
                    f"(transcript: {self._params.transcript_track!r}, "
                    f"audio: {self._params.audio_out_track!r})"
                )
                await self._callbacks.on_connected()

                # In serve mode, drive the accept loop in the background.
                # serve() holds each session task until the session closes,
                # so memory doesn't grow with past connections.
                serve_task: asyncio.Task | None = None
                if self._params.serve:
                    assert self._task_manager is not None, (
                        "MOQTransportClient.setup() must run before _run(); "
                        "input/output processors forward setup on the pipeline's first frame."
                    )
                    serve_task = self._task_manager.create_task(
                        cast(moq.Server, transport).serve(),
                        f"{self}::moq_serve",
                    )

                try:
                    # Drive the subscribe side. The output side keeps
                    # writing frames into self._audio_out /
                    # self._transcript_out from whichever task is running
                    # the pipeline.
                    await self._consume_peer(origin)
                finally:
                    if serve_task is not None and self._task_manager is not None:
                        await self._task_manager.cancel_task(serve_task)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_normal_close(e):
                # Browser closed the WebTransport session (or the bot did,
                # via disconnect()). Not an error — code=0 is a clean
                # close, expected at end-of-call.
                logger.debug(f"MOQ transport closed: {e}")
            else:
                logger.error(f"MOQ transport error: {e}", exc_info=True)
                await self._callbacks.on_error(str(e), e)
        finally:
            self._audio_out = None
            self._cert_fingerprints = []
            await self._callbacks.on_disconnected()

    def _make_transport(self, origin: "moq.OriginProducer"):
        """Return the async-context-manager that owns the MOQ session.

        Client mode dials the relay; serve mode binds a local socket and
        accepts incoming sessions. Both wire ``origin`` for publish and
        subscribe so the rest of ``_run`` is shape-identical.
        """
        if self._params.serve:
            tls_kwargs: dict = {}
            if self._params.serve_tls_cert and self._params.serve_tls_key:
                tls_kwargs["tls_cert"] = [self._params.serve_tls_cert]
                tls_kwargs["tls_key"] = [self._params.serve_tls_key]
            else:
                tls_kwargs["tls_generate"] = [self._params.serve_tls_host]
            return moq.Server(
                self._serve_bind,
                publish=origin,
                subscribe=origin,
                **tls_kwargs,
            )

        return moq.Client(
            self._url,
            tls_verify=self._params.verify_ssl,
            publish=origin,
            subscribe=origin,
        )

    async def _consume_peer(self, origin: "moq.OriginProducer"):
        """Wait for the peer broadcast and forward its audio track.

        Cancellation flows through the moq consumers themselves: every
        consumer we open is tracked in ``self._active_consumers`` so
        ``disconnect()`` can call ``.cancel()`` on it, which makes the
        relevant ``await`` return ``None`` cleanly. No bespoke race-with-
        an-event pattern needed.
        """
        consumer = origin.consume()

        logger.debug(f"MOQ: waiting for peer broadcast {self._peer_broadcast_path!r}")
        announced = self._track(consumer.announced_broadcast(self._peer_broadcast_path))
        try:
            peer_broadcast = await asyncio.wait_for(
                announced.available(), timeout=self._params.connection_timeout
            )
        except TimeoutError:
            logger.warning(
                f"MOQ: peer broadcast {self._peer_broadcast_path!r} did not appear "
                f"within {self._params.connection_timeout}s"
            )
            return
        except (asyncio.CancelledError, StopAsyncIteration):
            return

        logger.debug(f"MOQ: peer broadcast {self._peer_broadcast_path!r} available")
        await self._callbacks.on_client_connected()

        # Run the peer audio pump and the peer transcript (RTVI) pump
        # side by side. Both catch CancelledError internally; on
        # disconnect, ``consumer.cancel()`` on their moq consumers makes
        # each loop exit cleanly. ``return_exceptions=True`` so that if
        # one raises, we still wait for the other to finish instead of
        # leaving it running as an orphaned, uncancelled task — then we
        # re-raise the first real exception ourselves so callers see the
        # same failure they would have without ``return_exceptions``.
        try:
            results = await asyncio.gather(
                self._forward_peer_audio(peer_broadcast),
                self._forward_peer_transcript(peer_broadcast),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result
        finally:
            await self._callbacks.on_client_disconnected()

    async def _forward_peer_audio(self, peer_broadcast: "moq.BroadcastConsumer"):
        """Read the catalog, subscribe to the first audio track, pump PCM."""
        if not self._params.audio_in_enabled:
            return
        catalog_sub = self._track(peer_broadcast.subscribe_catalog())

        # @moq/publish.Broadcast publishes an initial catalog before the
        # mic permission resolves, so the first frame typically has no
        # audio. Wait for an update that does — the catalog subscription
        # is an async iterator that emits each refresh — capped so we
        # don't hang forever if the peer never adds audio at all.
        catalog = None
        try:
            async with asyncio.timeout(self._params.connection_timeout):
                async for next_catalog in catalog_sub:
                    if next_catalog.audio:
                        catalog = next_catalog
                        break
                    logger.debug(
                        f"MOQ: peer catalog has no audio yet "
                        f"(video={list(next_catalog.video)}); waiting for refresh"
                    )
        except (asyncio.CancelledError, StopAsyncIteration):
            return
        except TimeoutError:
            logger.warning(
                f"MOQ: peer broadcast never advertised audio within "
                f"{self._params.connection_timeout}s"
            )
            return

        if catalog is None or not catalog.audio:
            logger.warning("MOQ: peer closed before publishing audio in catalog")
            return

        track_name, audio = next(iter(catalog.audio.items()))
        if audio.codec != "opus":
            logger.warning(
                f"MOQ: peer audio codec {audio.codec!r} != opus; subscribe may fail. Skipping."
            )
            return

        target_rate = self._params.audio_in_sample_rate
        # The moq-rs Opus decoder (as of 0.2.17) won't remap channels.
        # `@moq/publish`'s browser-side encoder tends to publish stereo
        # even when the source MediaStreamTrack reports mono, because
        # `MediaStreamAudioSourceNode.channelCount` defaults to 2 and
        # the encoder falls back to that when `track.getSettings()`
        # omits `channelCount` (common on macOS). We decode at the
        # source channel count and downmix in Python below.
        source_channels = audio.channel_count
        logger.debug(
            f"MOQ: subscribing to peer audio {track_name!r} "
            f"(source rate={audio.sample_rate}Hz, channels={source_channels}, "
            f"output rate={target_rate}Hz, "
            f"max_latency_ms={self._params.audio_in_max_latency_ms})"
        )
        await self._callbacks.on_track_subscribed(
            MOQTrack(
                broadcast_path=self._peer_broadcast_path,
                name=track_name,
                track_type=MOQTrackType.AUDIO,
            ),
        )

        consumer = self._track(
            peer_broadcast.subscribe_audio(
                track_name,
                audio,
                moq.AudioDecoderOutput(
                    format=moq.AudioFormat.S16,
                    sample_rate=target_rate,
                    channels=source_channels,
                    latency_max_ms=self._params.audio_in_max_latency_ms,
                ),
            )
        )
        try:
            async for frame in consumer:
                if frame.data:
                    pcm = frame.data
                    if source_channels > 1:
                        pcm = _downmix_s16_to_mono(pcm, source_channels)
                    await self._callbacks.on_audio_received(pcm, target_rate)
        except asyncio.CancelledError:
            pass

    async def _forward_peer_transcript(self, peer_broadcast: "moq.BroadcastConsumer"):
        """Subscribe to the peer's transcript stream and forward RTVI messages inbound.

        Symmetric with the bot's own ``transcript`` stream (see
        :meth:`publish_transcript`): the client appends each RTVI message
        as a record on the JSON stream; the ``subscribe_json_stream``
        consumer yields them, already parsed, in order — every message,
        losslessly. Delivered via :attr:`MOQCallbacks.on_message_received`,
        which :class:`MOQTransport` wires to push an
        :class:`InputTransportMessageFrame` into the pipeline, handled by
        :class:`RTVIProcessor` the same way as any other transport. This
        is what lets ``client-ready`` (for protocol version negotiation),
        typed text input, function-call results, and any other
        client→server RTVI traffic reach the bot over MoQ.
        """
        track_name = self._params.transcript_track
        logger.debug(f"MOQ: subscribing to peer transcript {track_name!r}")
        consumer = self._track(peer_broadcast.subscribe_json_stream(track_name, compression=True))
        try:
            async for message in consumer:
                if not isinstance(message, dict):
                    logger.warning(
                        f"MOQ: expected RTVI object on transcript stream, got {type(message).__name__}"
                    )
                    continue
                await self._callbacks.on_message_received(message)
        except asyncio.CancelledError:
            pass

    def _track(self, consumer):
        """Remember a consumer so ``disconnect()`` can cancel it.

        Each moq consumer (announced-broadcast, catalog, track, audio,
        media) exposes a synchronous ``.cancel()`` that terminates any
        in-flight ``await``. Tracking them lets us tear down a session
        without racing on an external cancel event.
        """
        self._active_consumers.append(consumer)
        return consumer

    async def disconnect(self):
        """Disconnect from the MOQ relay.

        Sends an intra-transport ``session-ending`` notification on the
        transcript stream before tearing anything down. This gives the
        browser-side MoQ transport a chance to disable auto-reconnect,
        drain its jitter buffer, and close the audio decoder cleanly —
        without that heads-up, WebTransport just vanishes underneath
        Chrome's WebCodecs decoder and can crash the renderer ("Aw snap").
        """
        # Best-effort: publish the shutdown marker while the transcript
        # stream is still alive, then yield briefly so the frame has time
        # to reach the wire before we cancel _run. 250ms is well under
        # any interactive perception threshold.
        try:
            if hasattr(self, "_transcript_out") and self._transcript_out is not None:
                self._transcript_out.append({"label": "moq-transport", "type": "session-ending"})
                await asyncio.sleep(0.25)
        except Exception as e:
            logger.debug(f"MOQ: could not send session-ending notification: {e}")

        # Cancel any open consumers so their async iterations terminate.
        for c in self._active_consumers:
            try:
                c.cancel()
            except Exception as e:
                logger.debug(f"MOQ: consumer.cancel() raised: {e}")
        self._active_consumers.clear()

        if self._connection_task is not None and self._task_manager is not None:
            await self._task_manager.cancel_task(self._connection_task)
            self._connection_task = None

    async def cleanup(self):
        """Unconditionally tear down the underlying MOQ connection.

        Called as a last-resort safety net from
        :class:`MOQInputTransport.cleanup`/:class:`MOQOutputTransport.cleanup`
        (the FrameProcessor lifecycle hook, always invoked regardless of
        whether ``stop()``/``cancel()`` already released the session), so
        we guard against repeating the work. Prefer :meth:`stop`/
        :meth:`cancel` to end the session during normal shutdown — those
        wait for every holder to release before tearing anything down.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        await self.disconnect()

    async def _release(self, *, drain: bool):
        """Release one holder's claim on the session, draining first if asked.

        The session is only actually disconnected once every holder
        (input and output) has released it — see :attr:`_holders`.
        """
        if drain:
            await self.wait_for_audio_drain()
        self._holders -= 1
        if self._holders > 0:
            return
        await self.cleanup()

    async def stop(self):
        """Release this holder's claim on the session, draining buffered audio first.

        Called by both :meth:`MOQInputTransport.stop` and
        :meth:`MOQOutputTransport.stop`. Safe regardless of which one
        runs first: the session only actually disconnects once both
        have released their claim, and by the time the last one does,
        every audio frame written by the output side has already been
        drained. Draining twice (once per caller) is harmless — the
        second call just observes the clock has already caught up.
        """
        await self._release(drain=True)

    async def cancel(self):
        """Release this holder's claim on the session immediately, without draining.

        Called by both :meth:`MOQInputTransport.cancel` and
        :meth:`MOQOutputTransport.cancel`: cancellation is a hard stop,
        so there's nothing to drain, and the session disconnects as soon
        as both have released their claim.
        """
        await self._release(drain=False)


class MOQInputTransport(BaseInputTransport):
    """MOQ input transport: subscribes to the peer's audio track."""

    def __init__(
        self,
        client: MOQTransportClient,
        params: MOQParams,
        **kwargs,
    ):
        """Initialize the MOQ input transport.

        Args:
            client: MOQTransportClient instance managing the MoQ session.
            params: MOQ transport configuration parameters.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        """Forward setup to the shared MOQTransportClient so it can create tasks.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def start(self, frame: StartFrame):
        """Auto-connect to the MOQ relay when the pipeline starts.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True

        self._client.connect()
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the MOQ input transport.

        Releases this transport's claim on the shared session (see
        :meth:`MOQTransportClient.stop`). ``EndFrame`` reaches this
        (upstream) input transport before it reaches
        :class:`MOQOutputTransport` downstream, so this call alone
        doesn't end the session — the client waits for the output
        transport to release its own claim (after draining its buffered
        audio) before actually disconnecting.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        """Cancel the MOQ input transport.

        Releases this transport's claim on the shared session. Unlike
        ``stop()``, cancellation is immediate — there's no audio to
        drain.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.cancel()

    async def cleanup(self):
        """Cleanup resources."""
        await super().cleanup()
        await self._client.cleanup()

    async def push_received_audio(self, audio: bytes, sample_rate: int):
        """Push a received audio frame downstream.

        Args:
            audio: Raw mono 16-bit PCM audio bytes.
            sample_rate: Sample rate of ``audio``, in Hz.
        """
        await self.push_audio_frame(
            InputAudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=1)
        )

    async def push_received_message(self, message: dict):
        """Push a received RTVI message to :class:`RTVIProcessor` upstream.

        Called by :class:`MOQTransport` for every record delivered on the
        peer's transcript stream. ``RTVIProcessor`` sits upstream of this
        input transport (it's prepended by :class:`PipelineWorker` when
        ``enable_rtvi=True``), so pushing upstream is the direct route.

        Args:
            message: The parsed RTVI message.
        """
        await self.push_frame(
            InputTransportMessageFrame(message=message),
            FrameDirection.UPSTREAM,
        )


class MOQOutputTransport(BaseOutputTransport):
    """MOQ output transport: writes audio + transcript frames to MOQ tracks."""

    def __init__(
        self,
        client: MOQTransportClient,
        params: MOQParams,
        **kwargs,
    ):
        """Initialize the MOQ output transport.

        Args:
            client: MOQTransportClient instance managing the MoQ session.
            params: MOQ transport configuration parameters.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        """Forward setup to the shared MOQTransportClient so it can create tasks.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def start(self, frame: StartFrame):
        """Start the MOQ output transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True
        self._client.connect()
        # Open the publish_audio track now that we know the pipeline's
        # output sample rate. The producer side of the broadcast was
        # created in MOQTransportClient.__init__, so this call is
        # synchronous — no race with _run()'s async bring-up to lose
        # initial audio frames.
        self._client.open_audio_track(self.sample_rate)
        logger.debug(
            f"MOQ output: sample_rate={self.sample_rate}, chunk_size={self.audio_chunk_size}"
        )
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the MOQ output transport.

        On graceful end (an ``EndFrame`` — pushed by ``EndWorkerFrame``
        from an ``end_call`` tool, for example) let the moq audio buffer
        drain fully before releasing this transport's claim on the
        shared session (see :meth:`MOQTransportClient.stop`). Without
        this the bot's final TTS utterance gets clipped: base_output's
        stop() unblocks as soon as every audio frame has been *written*
        to moq's pacing queue, but moq buffers up to
        ``audio_out_max_buffer_ms`` (25s) of paced audio that still has
        to reach the browser and drain the client jitter buffer before
        it actually plays.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        """Cancel the MOQ output transport.

        Releases this transport's claim on the shared session. Unlike
        ``stop()``, cancellation is immediate — there's no audio to
        drain.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.cancel()

    async def cleanup(self):
        """Cleanup resources."""
        await super().cleanup()
        await self._client.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Reset the publish_audio pacing clock on interruption.

        The pacing in ``write_audio_frame`` keeps moq's in-flight buffer
        bounded so InterruptionFrame can actually stop playback in the
        browser — but the pacing clock needs to be re-anchored to ``now``
        so the next utterance plays immediately instead of catching up.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        if isinstance(frame, InterruptionFrame):
            self._client.reset_audio_pacing()
        await super().process_frame(frame, direction)

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Publish a transport message (RTVI JSON) on the transcript stream.

        Args:
            frame: The transport message frame to send.
        """
        try:
            self._client.publish_transcript(frame.message)
        except Exception as e:
            logger.warning(f"Failed to publish transport message: {e}")

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the bot's audio track.

        Args:
            frame: The output audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        try:
            await self._client.publish_audio(frame.audio)
            return True
        except Exception as e:
            logger.error(f"Error writing audio frame: {e}")
            return False


class MOQTransport(BaseTransport):
    """MOQ transport that connects to a MOQ relay via the ``moq`` library.

    Example::

        transport = MOQTransport(
            params=MOQParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                namespace="my-room",
            ),
            host="localhost",
            port=4080,
        )

        @transport.event_handler("on_connected")
        async def on_connected(transport):
            print("Connected to MOQ relay")

    Event handlers available:

    - ``on_connected`` — connection to relay established
    - ``on_disconnected`` — connection lost / closed
    - ``on_client_connected`` — peer broadcast announced (client joined)
    - ``on_client_disconnected`` — peer broadcast went away
    - ``on_track_subscribed`` — remote track subscription succeeded
    - ``on_error`` — error in the underlying transport
    """

    def __init__(
        self,
        params: MOQParams,
        host: str = "localhost",
        port: int = 4080,
        path: str = "/moq",
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the MOQ transport.

        Args:
            params: MOQ configuration parameters.
            host: Relay host address.
            port: Relay port number.
            path: MOQ endpoint path on the relay.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        callbacks = MOQCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_track_subscribed=self._on_track_subscribed,
            on_error=self._on_error,
            on_audio_received=self._on_audio_received,
            on_message_received=self._on_message_received,
        )
        self._client = MOQTransportClient(
            params=params,
            url=params.relay_url or f"https://{host}:{port}{path}",
            serve_bind=params.serve_bind or f"[::]:{port}",
            callbacks=callbacks,
        )

        self._input: MOQInputTransport | None = None
        self._output: MOQOutputTransport | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_track_subscribed")
        self._register_event_handler("on_error")

    def input(self) -> MOQInputTransport:
        """Get the input transport for receiving media."""
        if not self._input:
            self._input = MOQInputTransport(self._client, self._params, name=self._input_name)
        return self._input

    def output(self) -> MOQOutputTransport:
        """Get the output transport for sending media."""
        if not self._output:
            self._output = MOQOutputTransport(self._client, self._params, name=self._output_name)
        return self._output

    @property
    def cert_fingerprints(self) -> list[str]:
        """SHA-256 fingerprints (hex) of the serving cert (server mode only).

        Populated once the bot has bound; empty in client mode or before
        the session bring-up has reached the listen step. Useful for
        telling a browser client which self-signed cert to pin.
        """
        return self._client.cert_fingerprints

    async def disconnect(self):
        """Disconnect from the MOQ relay.

        See :meth:`MOQTransportClient.disconnect` for details.
        """
        await self._client.disconnect()

    # ------------------------------------------------------------------
    # MOQTransportClient callbacks.
    # ------------------------------------------------------------------

    async def _on_connected(self):
        """Handle the MOQ session (client or server) being established."""
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self):
        """Handle the MOQ session ending."""
        await self._call_event_handler("on_disconnected")

    async def _on_client_connected(self):
        """Handle the peer's broadcast being announced."""
        await self._call_event_handler("on_client_connected")

    async def _on_client_disconnected(self):
        """Handle the peer's broadcast going away."""
        await self._call_event_handler("on_client_disconnected")

    async def _on_track_subscribed(self, track: MOQTrack):
        """Handle a remote track subscription succeeding.

        Args:
            track: The subscribed track's identity.
        """
        await self._call_event_handler("on_track_subscribed", track)

    async def _on_error(self, message: str, exception: Exception):
        """Handle an error from the underlying MOQ transport.

        Args:
            message: Human-readable description of the error.
            exception: The underlying exception.
        """
        await self._call_event_handler("on_error", message, exception)

    async def _on_audio_received(self, audio: bytes, sample_rate: int):
        """Handle decoded PCM audio received from the peer's audio track.

        Args:
            audio: Raw mono 16-bit PCM audio bytes.
            sample_rate: Sample rate of ``audio``, in Hz.
        """
        if self._input is not None:
            await self._input.push_received_audio(audio, sample_rate)

    async def _on_message_received(self, message: dict):
        """Handle an RTVI message received from the peer's transcript stream.

        Args:
            message: The parsed RTVI message.
        """
        if self._input is not None:
            await self._input.push_received_message(message)

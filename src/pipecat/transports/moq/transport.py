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
single Opus track; transcript (RTVI JSON) rides on a raw byte track in
the same broadcast.

Two modes:

- **Client mode** (default): the bot dials a relay at ``relay_url`` (or
  the constructor's ``host``/``port``/``path``).
- **Server mode** (``serve=True``): the bot binds its own UDP socket via
  ``moq.Server`` and accepts the browser's direct connection. Removes
  the need for a separate ``moq-relay`` process for local dev. The
  self-signed cert fingerprints are exposed via
  :attr:`MOQTransport.cert_fingerprints` so a browser can pin them.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from loguru import logger
from pydantic import ConfigDict

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import moq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use MOQ transport, you need to `pip install pipecat-ai[moq]`.")
    raise Exception(f"Missing module: {e}")


DEFAULT_NAMESPACE = "pipecat"
DEFAULT_PARTICIPANT_ID = "bot0"
DEFAULT_PEER_ID = "client0"
DEFAULT_AUDIO_OUT_TRACK = "bot-audio"
DEFAULT_TRANSCRIPT_TRACK = "transcript"


def _downmix_s16_to_mono(pcm: bytes, channels: int) -> bytes:
    """Downmix interleaved S16 PCM to mono by averaging channels.

    Used to compensate for the @moq/publish browser encoder publishing
    stereo even when the source mic is mono (the encoder defaults to
    the MediaStreamAudioSourceNode's channelCount=2 when the
    MediaStreamTrack doesn't expose channelCount in its settings).
    """
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    if channels <= 1 or len(samples) % channels != 0:
        return pcm
    frames = len(samples) // channels
    out = array.array("h", [0]) * frames
    for i in range(frames):
        base = i * channels
        acc = 0
        for c in range(channels):
            acc += samples[base + c]
        out[i] = max(-32768, min(32767, acc // channels))
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
        transcript_track: Name of the bot's outgoing transcript track.
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
        audio_out_max_buffer_ms: Upper bound on how much audio we let
            the moq library hold in flight at once. The transport
            paces ``publish_audio`` writes against wall-clock so the
            unwritten + encoder + wire backlog never grows past this
            many milliseconds. Lower = faster response to interrupts,
            but more sensitive to scheduler jitter. Higher = smoother
            audio under load, but a user interrupt won't actually
            stop playback until this much already-buffered audio has
            drained on the browser side.

            WORKAROUND: moq-rs (>=0.2.17) doesn't expose a flush /
            cancel primitive on :class:`moq.AudioProducer`, so the
            only way to bound interruption latency is to keep the
            in-flight buffer small. Once an upstream flush API lands,
            this pacing can go away and the parameter can be removed.
            See https://github.com/moq-dev/moq/issues/1614.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    relay_url: Optional[str] = None
    namespace: str = DEFAULT_NAMESPACE
    participant_id: str = DEFAULT_PARTICIPANT_ID
    peer_id: str = DEFAULT_PEER_ID
    audio_out_track: str = DEFAULT_AUDIO_OUT_TRACK
    transcript_track: str = DEFAULT_TRANSCRIPT_TRACK
    verify_ssl: bool = True
    connection_timeout: float = 30.0
    serve: bool = False
    serve_bind: Optional[str] = None
    serve_tls_host: str = "localhost"
    serve_tls_cert: Optional[str] = None
    serve_tls_key: Optional[str] = None
    audio_out_sample_rate: int = 24000
    audio_in_sample_rate: int = 16000
    audio_in_max_latency_ms: int = 500
    audio_out_frame_ms: int = 20
    audio_out_max_buffer_ms: int = 20


class MOQInputTransport(BaseInputTransport):
    """MOQ input transport: subscribes to the peer's audio track."""

    def __init__(
        self,
        transport: "MOQTransport",
        params: MOQParams,
        **kwargs,
    ):
        """Initialize the MOQ input transport."""
        super().__init__(params, **kwargs)
        self._moq_transport = transport
        self._params = params
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Auto-connect to the MOQ relay when the pipeline starts."""
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True

        self._moq_transport._connection_task = self.create_task(
            self._moq_transport._run(), "moq_run"
        )
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the MOQ input transport."""
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the MOQ input transport."""
        await super().cancel(frame)

    async def cleanup(self):
        """Cleanup resources."""
        await super().cleanup()
        await self._moq_transport.cleanup()

    async def push_received_audio(self, audio: bytes, sample_rate: int):
        """Push a received audio frame downstream."""
        await self.push_audio_frame(
            InputAudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=1)
        )


class MOQOutputTransport(BaseOutputTransport):
    """MOQ output transport: writes audio + transcript frames to MOQ tracks."""

    def __init__(
        self,
        transport: "MOQTransport",
        params: MOQParams,
        **kwargs,
    ):
        """Initialize the MOQ output transport."""
        super().__init__(params, **kwargs)
        self._moq_transport = transport
        self._params = params
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the MOQ output transport."""
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True
        # Open the publish_audio track now that we know the pipeline's
        # output sample rate. The producer side of the broadcast was
        # created in MOQTransport.__init__, so this call is synchronous
        # — no race with _run() to lose initial audio frames.
        self._moq_transport.open_audio_track(self.sample_rate)
        logger.info(
            f"MOQ output: sample_rate={self.sample_rate}, chunk_size={self.audio_chunk_size}"
        )
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the MOQ output transport."""
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the MOQ output transport."""
        await super().cancel(frame)

    async def cleanup(self):
        """Cleanup resources."""
        await super().cleanup()
        await self._moq_transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Reset the publish_audio pacing clock on interruption.

        The pacing in ``write_audio_frame`` keeps moq's in-flight buffer
        bounded so InterruptionFrame can actually stop playback in the
        browser — but the pacing clock needs to be re-anchored to ``now``
        so the next utterance plays immediately instead of catching up.
        """
        if isinstance(frame, InterruptionFrame):
            self._moq_transport.reset_audio_pacing()
        await super().process_frame(frame, direction)

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Publish a transport message (RTVI JSON) on the transcript track."""
        payload = frame.message
        if not isinstance(payload, (bytes, bytearray)):
            payload = json.dumps(payload).encode("utf-8")
        try:
            self._moq_transport.publish_transcript(payload)
        except Exception as e:
            logger.warning(f"Failed to publish transport message: {e}")

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the bot's audio track."""
        try:
            await self._moq_transport.publish_audio(frame.audio)
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
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the MOQ transport.

        The publish broadcast and its transcript track are created here,
        synchronously, so the output processor can begin writing into
        them before the async session bring-up in :meth:`_run` finishes
        — otherwise the bot's first ~hundreds of ms of audio gets
        silently dropped while we're still dialing the relay.

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
        self._url = params.relay_url or f"https://{host}:{port}{path}"
        self._serve_bind = params.serve_bind or f"[::]:{port}"

        self._input: Optional[MOQInputTransport] = None
        self._output: Optional[MOQOutputTransport] = None
        self._connection_task: Optional[asyncio.Task] = None

        # Owned for the lifetime of this transport. Created synchronously
        # so MOQOutputTransport.start() can publish_audio + write
        # transcript frames without racing with _run()'s async bring-up.
        self._publish_broadcast: moq.BroadcastProducer = moq.BroadcastProducer()
        self._transcript_out: moq.TrackProducer = self._publish_broadcast.publish_track(
            params.transcript_track
        )
        # Audio track is opened lazily once the pipeline's output sample
        # rate is known (in :meth:`open_audio_track`). We stash the rate
        # because ``publish_audio`` uses it to convert byte length to
        # wall-clock duration for pacing.
        self._audio_out: Optional[moq.AudioProducer] = None
        self._audio_out_sample_rate: Optional[int] = None
        # Wall-clock target for the next publish_audio write. ``None``
        # means "reset" — the next write anchors to ``time.monotonic()``.
        self._publish_audio_clock: Optional[float] = None

        # Track consumers we created so disconnect() can cancel them.
        # Each entry has a sync .cancel() method that terminates any
        # pending await on the consumer.
        self._active_consumers: list = []

        self._cert_fingerprints: list[str] = []

        self._broadcast_path = f"{params.namespace}/{params.participant_id}"
        self._peer_broadcast_path = f"{params.namespace}/{params.peer_id}"
        self._cleaned_up = False

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_track_subscribed")
        self._register_event_handler("on_error")

    def input(self) -> MOQInputTransport:
        """Get the input transport for receiving media."""
        if not self._input:
            self._input = MOQInputTransport(self, self._params, name=self._input_name)
        return self._input

    def output(self) -> MOQOutputTransport:
        """Get the output transport for sending media."""
        if not self._output:
            self._output = MOQOutputTransport(self, self._params, name=self._output_name)
        return self._output

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
                sample_rate=None,  # let the library pick an Opus-supported rate
                channels=None,
                bitrate=None,
                frame_duration_ms=self._params.audio_out_frame_ms,
            ),
        )
        logger.info(
            f"MOQ: publishing audio as Opus "
            f"(pipeline rate={sample_rate}Hz, frame={self._params.audio_out_frame_ms}ms, "
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
        """Push a PCM chunk to the bot's audio track, paced at audio rate.

        The library does the Opus encode + resample inside the FFI, so we
        just write S16 PCM bytes.

        WORKAROUND for moq-rs (>=0.2.17): ``AudioProducer`` is fire-and-
        forget. ``write()`` accepts bytes as fast as the caller produces
        them and queues everything through the encoder, the WebTransport
        send window, the wire, and the browser-side jitter buffer. The
        sum of those buffers is what defeats user interruption — pipecat
        drains its own audio queue on InterruptionFrame, but the bytes
        that have already been ``write()``-en keep playing.

        Until ``AudioProducer`` exposes a flush / cancel primitive
        (tracking issue: https://github.com/moq-dev/moq/issues/1614),
        we bound the in-flight buffer by pacing the writes
        against a virtual clock: each call advances the clock by the
        chunk's audio duration, and we sleep until wall-clock is within
        ``audio_out_max_buffer_ms`` of it. moq then holds at most that
        many ms in flight at the moment of an interrupt, so the bot's
        voice cuts within roughly that latency plus the browser jitter
        buffer (``MoqTransportOptions.audioLatencyMs``).
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

        # Advance the virtual clock by the duration of this chunk.
        # S16 = 2 bytes/sample, mono.
        duration_s = len(audio) / (self._audio_out_sample_rate * 2)
        self._publish_audio_clock += duration_s

        # Timestamp is informational for downstream A/V sync; the encoder
        # paces frames itself. Use 0 since we only carry audio and the
        # browser plays it as soon as it arrives.
        self._audio_out.write(moq.AudioFrame(timestamp_us=0, data=audio))

    def publish_transcript(self, payload: bytes):
        """Write a transcript payload (RTVI JSON) to the transcript track."""
        self._transcript_out.write_frame(payload)

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
        origin = moq.OriginProducer()

        if self._params.serve:
            ctx_label = f"serving {self._serve_bind} as {self._broadcast_path}"
        else:
            ctx_label = f"connecting to {self._url} as {self._broadcast_path}"
        logger.info(f"MOQ: {ctx_label}")

        try:
            async with self._make_transport(origin) as transport:
                if self._params.serve:
                    self._cert_fingerprints = transport.cert_fingerprints()
                    logger.info(
                        f"MOQ: bound on {transport.local_addr} "
                        f"(cert sha256: {self._cert_fingerprints})"
                    )

                origin.publish(self._broadcast_path, self._publish_broadcast)
                logger.info(
                    f"MOQ: published broadcast {self._broadcast_path!r} "
                    f"(transcript: {self._params.transcript_track!r}, "
                    f"audio: {self._params.audio_out_track!r})"
                )
                await self._call_event_handler("on_connected")

                # In serve mode, drive the accept loop in the background.
                # serve() holds each session task until the session closes,
                # so memory doesn't grow with past connections.
                serve_task: Optional[asyncio.Task] = None
                if self._params.serve:
                    serve_task = asyncio.create_task(transport.serve())

                try:
                    # Drive the subscribe side. The output side keeps
                    # writing frames into self._audio_out /
                    # self._transcript_out from whichever task is running
                    # the pipeline.
                    await self._consume_peer(origin)
                finally:
                    if serve_task is not None:
                        serve_task.cancel()
                        try:
                            await serve_task
                        except asyncio.CancelledError:
                            pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"MOQ transport error: {e}", exc_info=True)
            await self._call_event_handler("on_error", str(e), e)
        finally:
            self._audio_out = None
            self._cert_fingerprints = []
            await self._call_event_handler("on_disconnected")

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
        if not self._params.audio_in_enabled:
            # Just wait until disconnect() cancels the parent task.
            await asyncio.Event().wait()
            return

        consumer = origin.consume()

        logger.info(f"MOQ: waiting for peer broadcast {self._peer_broadcast_path!r}")
        announced = self._track(consumer.announced_broadcast(self._peer_broadcast_path))
        try:
            peer_broadcast = await asyncio.wait_for(
                announced.available(), timeout=self._params.connection_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"MOQ: peer broadcast {self._peer_broadcast_path!r} did not appear "
                f"within {self._params.connection_timeout}s"
            )
            return
        except (asyncio.CancelledError, StopAsyncIteration):
            return

        logger.info(f"MOQ: peer broadcast {self._peer_broadcast_path!r} available")
        await self._call_event_handler("on_client_connected")

        try:
            await self._forward_peer_audio(peer_broadcast)
        finally:
            await self._call_event_handler("on_client_disconnected")

    async def _forward_peer_audio(self, peer_broadcast: "moq.BroadcastConsumer"):
        """Read the catalog, subscribe to the first audio track, pump PCM."""
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
        except asyncio.TimeoutError:
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
        logger.info(
            f"MOQ: subscribing to peer audio {track_name!r} "
            f"(source rate={audio.sample_rate}Hz, channels={source_channels}, "
            f"output rate={target_rate}Hz, "
            f"max_latency_ms={self._params.audio_in_max_latency_ms})"
        )
        await self._call_event_handler(
            "on_track_subscribed",
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
                if frame.data and self._input is not None:
                    pcm = frame.data
                    if source_channels > 1:
                        pcm = _downmix_s16_to_mono(pcm, source_channels)
                    await self._input.push_received_audio(pcm, target_rate)
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
        """Disconnect from the MOQ relay."""
        # Cancel any open consumers so their async iterations terminate.
        for c in self._active_consumers:
            try:
                c.cancel()
            except Exception as e:
                logger.debug(f"MOQ: consumer.cancel() raised: {e}")
        self._active_consumers.clear()

        if self._connection_task is not None:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

    async def cleanup(self):
        """Tear down the underlying MOQ connection.

        Both the input and output processors call this on shutdown, so
        we guard against repeating the work.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        await self.disconnect()

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) transport implementation for Pipecat.

This transport uses the upstream ``moq`` Python library
(`moq-rs <https://pypi.org/project/moq-rs/>`_) — Rust bindings for Media
over QUIC — instead of re-implementing the wire protocol. The library
handles the QUIC connection, MOQ session, announcement discovery,
subscription routing, group/frame framing, codec-specific media
catalog management, and the Opus encode/decode + resampling for raw
audio tracks.

Each participant publishes its tracks under a per-participant broadcast
path ``<namespace>/<participant_id>`` (e.g. ``pipecat/bot0``). The bot
subscribes to the peer's broadcast at ``<namespace>/<peer_id>``.

Audio is exchanged through ``publish_audio`` / ``subscribe_audio``: we
hand the library raw PCM and it does the Opus encode + rubato resample
on the way out, and the inverse on the way in. The
``AudioDecoderOutput.latency_max_ms`` knob bounds how long the consumer
will wait for late frames — that's the headline congestion-control SLA
versus WebRTC. Transcript messages (RTVI) ride on a raw byte track in
the same broadcast.
"""

import asyncio
import json
import time
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

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
from pipecat.transports.moq.protocol import MOQRole, MOQTrack, MOQTrackType

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


class MOQParams(TransportParams):
    """Configuration parameters for MOQ transport.

    Supports two modes:

    - **Client mode** (default): the bot dials a MOQ relay at ``relay_url``
      (or ``host``/``port``/``path`` on the constructor).
    - **Server mode** (``serve=True``): the bot is the MOQ server — it
      binds a UDP socket itself and accepts the browser's direct
      connection. Removes the need for a separate ``moq-relay`` process
      for local dev.

    Parameters:
        role: MOQ role (kept for compatibility; the underlying library
            handles pub/sub negotiation automatically).
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
        connection_timeout: Seconds to wait for the peer broadcast to
            appear before giving up.
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
            produces; the encoder resamples to the nearest Opus-supported
            rate before encoding.
        audio_in_sample_rate: Sample rate the pipeline expects to receive
            user audio at (Hz). Decoded Opus is resampled to this rate
            before being pushed downstream.
        audio_in_max_latency_ms: How long :func:`subscribe_audio` will
            wait for a late frame before skipping ahead. Lower = more
            interactive (fewer fills, more drops on bad networks);
            higher = smoother audio with more glass-to-glass delay.
        audio_out_frame_ms: Opus frame duration for the bot's audio
            output. Must be one of 2.5, 5, 10, 20, 40, 60. 20 ms is the
            real-time default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: MOQRole = MOQRole.PUBSUB
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
    audio_in_max_latency_ms: int = 100
    audio_out_frame_ms: float = 20.0


class MOQCallbacks(BaseModel):
    """Callback functions wired between :class:`MOQTransport` and its I/O processors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_client_connected: Callable[[], Awaitable[None]]
    on_client_disconnected: Callable[[], Awaitable[None]]
    on_track_published: Callable[[MOQTrack], Awaitable[None]]
    on_track_subscribed: Callable[[MOQTrack], Awaitable[None]]
    on_error: Callable[[str, Optional[Exception]], Awaitable[None]]


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
        self._send_interval = 0.0
        self._next_send_time = 0.0
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the MOQ output transport."""
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        logger.info(
            f"MOQ output: sample_rate={self.sample_rate}, "
            f"chunk_size={self.audio_chunk_size}, send_interval={self._send_interval:.4f}s"
        )
        # We need the pipeline's output sample rate to open the audio
        # track. If the connection is already up, this kicks off the
        # publish_audio track; otherwise _run() will pick it up once
        # _publish_broadcast is created.
        self._moq_transport.prepare_audio_encoder(self.sample_rate)
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
        """Process frames; reset pacing on interruption."""
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._next_send_time = 0.0

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
            self._moq_transport.publish_audio(frame.audio)
            await self._write_audio_sleep()
            return True
        except Exception as e:
            logger.error(f"Error writing audio frame: {e}")
            return False

    async def _write_audio_sleep(self):
        """Pace audio output so we don't burst all chunks at once."""
        current_time = time.monotonic()
        sleep_duration = max(0.0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


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
    - ``on_track_published`` — local track now has a subscriber
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

        # Set on connect; cleared on disconnect.
        self._publish_broadcast: Optional[moq.BroadcastProducer] = None
        self._audio_out: Optional[moq.AudioProducer] = None
        self._audio_out_rate: int = 0  # set in prepare_audio_encoder
        self._audio_out_pts_us: int = 0  # presentation timestamp, advances per write
        self._transcript_out: Optional[moq.TrackProducer] = None
        self._cert_fingerprints: list[str] = []
        self._cancel_event = asyncio.Event()

        self._broadcast_path = f"{params.namespace}/{params.participant_id}"
        self._peer_broadcast_path = f"{params.namespace}/{params.peer_id}"
        self._cleaned_up = False

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_track_published")
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

    def publish_audio(self, audio: bytes):
        """Push a PCM chunk to the bot's audio track.

        The library does the Opus encode + resample inside the FFI, so
        we just write raw bytes with a presentation timestamp. The
        encoder is set up lazily — :meth:`MOQOutputTransport.start`
        calls :meth:`prepare_audio_encoder` with the pipeline's rate.
        """
        if self._audio_out is None:
            return
        self._audio_out.write(
            moq.AudioFrame(timestamp_us=self._audio_out_pts_us, data=audio)
        )
        # Advance the presentation timestamp by however much real time
        # this PCM chunk represents (2 bytes/sample, 1 channel).
        if self._audio_out_rate:
            self._audio_out_pts_us += (len(audio) * 1_000_000) // (
                self._audio_out_rate * 2
            )

    def prepare_audio_encoder(self, sample_rate: int):
        """Open the bot's audio track via ``publish_audio``.

        Called by :class:`MOQOutputTransport` once it knows the
        pipeline's output sample rate. No-op if the encoder is already
        set up or if the broadcast isn't published yet.
        """
        if self._audio_out is not None or self._publish_broadcast is None:
            return
        if not self._params.audio_out_enabled:
            return

        self._audio_out_rate = sample_rate
        self._audio_out_pts_us = 0
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
                frame_duration_ms=int(self._params.audio_out_frame_ms),
            ),
        )
        logger.info(
            f"MOQ: publishing audio as Opus "
            f"(pipeline rate={sample_rate}Hz, frame={self._params.audio_out_frame_ms}ms, "
            f"track={self._params.audio_out_track!r})"
        )

    def publish_transcript(self, payload: bytes):
        """Write a transcript payload (RTVI JSON) to the transcript track."""
        if self._transcript_out is not None:
            self._transcript_out.write_frame(payload)

    @property
    def cert_fingerprints(self) -> list[str]:
        """SHA-256 fingerprints of the serving cert (server mode only).

        Empty until the bot has finished binding. Useful for telling a
        browser client which self-signed cert to pin.
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

        try:
            if self._params.serve:
                ctx_label = f"serving {self._serve_bind} as {self._broadcast_path}"
            else:
                ctx_label = f"connecting to {self._url} as {self._broadcast_path}"
            logger.info(f"MOQ: {ctx_label}")

            async with self._make_transport(origin) as transport:
                if self._params.serve:
                    self._cert_fingerprints = transport.cert_fingerprints()
                    logger.info(
                        f"MOQ: bound on {transport.local_addr} "
                        f"(cert sha256: {self._cert_fingerprints})"
                    )

                self._publish_broadcast = moq.BroadcastProducer()
                # The audio media track is created lazily by
                # prepare_audio_encoder() once the pipeline tells us the
                # output sample rate — we can't initialise Opus until then.
                self._transcript_out = self._publish_broadcast.publish_track(
                    self._params.transcript_track
                )

                origin.publish(self._broadcast_path, self._publish_broadcast)
                logger.info(
                    f"MOQ: published broadcast {self._broadcast_path!r} "
                    f"(transcript: {self._params.transcript_track!r}, "
                    f"audio: pending pipeline start)"
                )

                # If MOQOutputTransport.start() already ran and tried to
                # set up the encoder, the broadcast was None at that
                # point so it bailed. Retry now that the broadcast exists.
                if (
                    self._output is not None
                    and self._output._initialized
                    and self._audio_out is None
                ):
                    self.prepare_audio_encoder(self._output.sample_rate)

                await self._call_event_handler("on_connected")

                # In serve mode, drive the accept loop in the background.
                # serve() holds each session task until the session closes,
                # so memory doesn't grow with past connections.
                serve_task: Optional[asyncio.Task] = None
                if self._params.serve:
                    serve_task = asyncio.create_task(transport.serve())

                try:
                    # Drive the subscribe side. The output side keeps writing
                    # frames into self._audio_out / self._transcript_out from
                    # whichever task is running the pipeline.
                    await self._consume_peer(origin)
                finally:
                    if serve_task is not None:
                        serve_task.cancel()
                        try:
                            await serve_task
                        except (asyncio.CancelledError, Exception):
                            pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"MOQ transport error: {e}", exc_info=True)
            await self._call_event_handler("on_error", str(e), e)
        finally:
            self._audio_out = None
            self._audio_out_pts_us = 0
            self._audio_out_rate = 0
            self._transcript_out = None
            self._publish_broadcast = None
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
        """Wait for the peer broadcast and forward its audio track."""
        if not self._params.audio_in_enabled:
            await self._cancel_event.wait()
            return

        consumer = origin.consume()
        logger.info(f"MOQ: waiting for peer broadcast {self._peer_broadcast_path!r}")
        announced = consumer.announced_broadcast(self._peer_broadcast_path)
        wait_task = asyncio.create_task(announced.available())
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                [wait_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                wait_task.cancel()
                announced.cancel()
                return
            peer_broadcast = wait_task.result()
        finally:
            cancel_task.cancel()

        logger.info(f"MOQ: peer broadcast {self._peer_broadcast_path!r} available")
        await self._call_event_handler("on_client_connected")

        # Read the catalog to find the peer's audio track. The publisher
        # (browser) decides the codec + sample rate; we just pick the
        # first Opus rendition.
        catalog_sub = peer_broadcast.subscribe_catalog()
        catalog_task = asyncio.create_task(catalog_sub.__anext__())
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                [catalog_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                catalog_task.cancel()
                catalog_sub.cancel()
                return
            catalog = catalog_task.result()
        except StopAsyncIteration:
            logger.warning("MOQ: peer closed before publishing a catalog")
            return
        finally:
            cancel_task.cancel()

        if not catalog.audio:
            logger.warning(
                f"MOQ: peer broadcast has no audio in catalog "
                f"(video={list(catalog.video)})"
            )
            return

        track_name, audio = next(iter(catalog.audio.items()))
        target_rate = self._params.audio_in_sample_rate
        logger.info(
            f"MOQ: subscribing to peer audio {track_name!r} "
            f"(codec={audio.codec}, source rate={audio.sample_rate}Hz, "
            f"channels={audio.channel_count}, "
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

        consumer = peer_broadcast.subscribe_audio(
            track_name,
            audio,
            moq.AudioDecoderOutput(
                format=moq.AudioFormat.S16,
                sample_rate=target_rate,
                channels=1,
                latency_max_ms=self._params.audio_in_max_latency_ms,
            ),
        )
        try:
            await self._forward_audio(consumer, target_rate)
        finally:
            consumer.cancel()
            await self._call_event_handler("on_client_disconnected")

    async def _forward_audio(self, consumer: "moq.AudioConsumer", sample_rate: int):
        """Pump decoded PCM from an AudioConsumer into the input transport."""
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        try:
            while True:
                recv_task = asyncio.create_task(consumer.__anext__())
                done, _ = await asyncio.wait(
                    [recv_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task in done:
                    recv_task.cancel()
                    return
                try:
                    frame = recv_task.result()
                except StopAsyncIteration:
                    return

                if frame.data and self._input is not None:
                    await self._input.push_received_audio(frame.data, sample_rate)
        finally:
            cancel_task.cancel()

    async def disconnect(self):
        """Disconnect from the MOQ relay."""
        self._cancel_event.set()
        if self._connection_task is not None:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except (asyncio.CancelledError, Exception):
                pass
            self._connection_task = None

    async def cleanup(self):
        """Tear down the underlying MOQ connection.

        Both the input and output processors call this on shutdown, so we
        guard against repeating the work.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        await self.disconnect()

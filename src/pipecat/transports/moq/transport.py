#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) transport implementation for Pipecat — moq-lite-02.

This module provides MOQ transport functionality for real-time media streaming
using the QUIC protocol, connecting to MOQ relays for low-latency audio and
video transmission with pub/sub semantics.

moq-lite-02 uses a stream-per-request model: each operation (setup,
subscribe, announce) opens its own bidirectional QUIC stream, and media
data flows on unidirectional streams as GROUP + FRAME sequences.

Based on moq-lite-02 (version code 0xff0dad02).
"""

import asyncio
import json
import ssl
import struct
import time
from typing import Awaitable, Callable, Dict, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.moq.protocol import (
    CLIENT_SETUP_TYPE,
    MOQL_ALPN,
    MOQL_VERSION,
    SERVER_SETUP_TYPE,
    STREAM_TYPE_ANNOUNCE,
    STREAM_TYPE_SESSION,
    STREAM_TYPE_SUBSCRIBE,
    MOQCodec,
    MOQRole,
    MOQSession,
    MOQTrack,
    MOQTrackType,
)

try:
    from aioquic.asyncio import connect
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import (
        ConnectionTerminated,
        HandshakeCompleted,
        QuicEvent,
        StreamDataReceived,
        StreamReset,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use MOQ transport, you need to `pip install pipecat-ai[moq]`.")
    raise Exception(f"Missing module: {e}")


# Default track settings
DEFAULT_NAMESPACE = "pipecat"
DEFAULT_PARTICIPANT_ID = "bot0"
DEFAULT_PEER_ID = "client0"
DEFAULT_AUDIO_TRACK_NAME = "bot-audio"
DEFAULT_AUDIO_IN_TRACK_NAME = "user-audio"
DEFAULT_VIDEO_TRACK_NAME = "video"
DEFAULT_TRANSCRIPT_TRACK_NAME = "transcript"


class MOQParams(TransportParams):
    """Configuration parameters for MOQ transport.

    Each MOQ participant publishes under its own broadcast path:
    ``<namespace>/<participant_id>``. The bot subscribes to a peer's broadcast
    at ``<namespace>/<peer_id>``. Tracks live inside a broadcast and are
    named by their role (e.g. ``bot-audio``, ``user-audio``,
    ``custom-audio``). Using distinct broadcast paths lets the relay route
    SUBSCRIBEs to the right participant when multiple bots/clients are on
    the same namespace.

    Parameters:
        role: The MOQ role (publisher, subscriber, or pubsub).
        relay_url: URL of the MOQ relay (e.g., "https://localhost:4080/moq").
        namespace: Top-level namespace (defaults to "pipecat").
        participant_id: This bot's unique suffix inside ``namespace``.
            Combined with ``namespace`` gives the broadcast this bot
            publishes under, e.g. ``pipecat/bot0``.
        peer_id: The peer participant id to subscribe to, e.g.
            ``client0`` → bot subscribes to ``pipecat/client0/<track>``.
        audio_track_name: Name for the output audio track the bot
            publishes inside its broadcast (bot → client).
        audio_in_track_name: Name for the input audio track the bot
            subscribes to inside the peer's broadcast (client → bot).
        video_track_name: Name for the video track inside the bot's
            broadcast.
        priority: Default publisher priority (lower is higher priority).
        verify_ssl: Whether to verify SSL certificates.
        connection_timeout: Connection timeout in seconds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: MOQRole = MOQRole.PUBSUB
    relay_url: Optional[str] = None
    namespace: str = DEFAULT_NAMESPACE
    participant_id: str = DEFAULT_PARTICIPANT_ID
    peer_id: str = DEFAULT_PEER_ID
    audio_track_name: str = DEFAULT_AUDIO_TRACK_NAME
    audio_in_track_name: str = DEFAULT_AUDIO_IN_TRACK_NAME
    video_track_name: str = DEFAULT_VIDEO_TRACK_NAME
    transcript_track_name: str = DEFAULT_TRANSCRIPT_TRACK_NAME
    priority: int = 128
    verify_ssl: bool = True
    connection_timeout: float = 30.0


class MOQCallbacks(BaseModel):
    """Callback functions for MOQ transport events.

    Parameters:
        on_connected: Called when connection is established.
        on_disconnected: Called when connection is lost.
        on_track_published: Called when a track is successfully published.
        on_track_subscribed: Called when a subscription is confirmed.
        on_error: Called when an error occurs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_track_published: Callable[[MOQTrack], Awaitable[None]]
    on_track_subscribed: Callable[[MOQTrack], Awaitable[None]]
    on_error: Callable[[str, Optional[Exception]], Awaitable[None]]


class MOQClientProtocol(QuicConnectionProtocol):
    """QUIC protocol handler for MOQ client connections (moq-lite-02).

    Uses stream-per-request model: each operation opens its own bidi stream.
    Media data flows on unidirectional streams as GROUP + FRAME sequences.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the MOQ protocol handler."""
        super().__init__(*args, **kwargs)
        self._session: MOQSession = MOQSession()
        self._setup_complete = asyncio.Event()
        self._connected = False

        # Callbacks for received media and events
        self._on_subscribe_received: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_audio_frame: Optional[Callable[[bytes, int], Awaitable[None]]] = None
        self._on_video_frame: Optional[Callable[[bytes, int, int], Awaitable[None]]] = None
        self._on_message: Optional[Callable[[bytes], Awaitable[None]]] = None

        # Subscribe IDs for our subscriptions (receiving data)
        self._audio_subscribe_id: Optional[int] = None
        self._video_subscribe_id: Optional[int] = None

        # Subscribe ID assigned by relay for our published tracks
        self._publish_subscribe_ids: Dict[str, int] = {}

        # Stream buffers for incoming data
        self._stream_buffers: Dict[int, bytes] = {}
        self._stream_types: Dict[int, Optional[int]] = {}

        # Setup stream
        self._setup_stream_id: Optional[int] = None

        # Track which subscribe IDs map to which track names
        self._subscribe_id_to_track: Dict[int, str] = {}

        # Track subscribe stream IDs for detecting rejections
        self._subscribe_streams: Dict[int, int] = {}  # stream_id -> subscribe_id
        self._rejected_subscribe_ids: set = set()

        # Namespace + per-participant identity. Broadcast path this client
        # publishes under is "<namespace>/<participant_id>".
        self._namespace: str = DEFAULT_NAMESPACE
        self._participant_id: str = DEFAULT_PARTICIPANT_ID
        self._audio_track_name: str = DEFAULT_AUDIO_TRACK_NAME
        self._video_track_name: str = DEFAULT_VIDEO_TRACK_NAME
        self._transcript_track_name: str = DEFAULT_TRANSCRIPT_TRACK_NAME

    def set_audio_callback(self, callback: Callable[[bytes, int], Awaitable[None]]):
        """Set the callback for received audio frames."""
        self._on_audio_frame = callback

    def set_video_callback(self, callback: Callable[[bytes, int, int], Awaitable[None]]):
        """Set the callback for received video frames."""
        self._on_video_frame = callback

    def set_message_callback(self, callback: Callable[[bytes], Awaitable[None]]):
        """Set the callback for received messages."""
        self._on_message = callback

    def quic_event_received(self, event: QuicEvent):
        """Handle QUIC events."""
        if isinstance(event, HandshakeCompleted):
            logger.info(
                f"🟠 QUIC handshake completed (alpn={event.alpn_protocol}, "
                f"transport=moq-lite-02, version={MOQL_VERSION:#x})"
            )
            if event.alpn_protocol != MOQL_ALPN:
                logger.error(
                    f"💔❌💔 UNEXPECTED ALPN! "
                    f"Expected '{MOQL_ALPN}' but relay negotiated "
                    f"'{event.alpn_protocol}'. Connection may not work!"
                )
            self._connected = True

        elif isinstance(event, StreamDataReceived):
            self._handle_stream_data(event.stream_id, event.data, event.end_stream)

        elif isinstance(event, StreamReset):
            logger.warning(f"Stream {event.stream_id} reset (error_code={event.error_code})")
            # Track which subscribe streams were reset (subscription rejected)
            if event.stream_id in self._subscribe_streams:
                sub_id = self._subscribe_streams[event.stream_id]
                logger.warning(f"Subscribe {sub_id} was rejected (stream reset)")
                self._rejected_subscribe_ids.add(sub_id)
                if sub_id == self._audio_subscribe_id:
                    self._audio_subscribe_id = None

        elif isinstance(event, ConnectionTerminated):
            logger.info(f"Connection terminated: {event.reason_phrase}")
            self._connected = False

    def _is_unidirectional(self, stream_id: int) -> bool:
        """Check if a stream ID is unidirectional."""
        return (stream_id & 0x02) != 0

    def _is_server_initiated(self, stream_id: int) -> bool:
        """Check if a stream ID is server-initiated."""
        return (stream_id & 0x01) != 0

    def _handle_stream_data(self, stream_id: int, data: bytes, end_stream: bool):
        """Handle data received on a QUIC stream."""
        if not data and not end_stream:
            return

        logger.trace(
            f"Stream {stream_id} data ({len(data)} bytes, end={end_stream}): "
            f"[{' '.join(f'{b:02x}' for b in data[:40])}]"
        )

        # Buffer data
        if stream_id not in self._stream_buffers:
            self._stream_buffers[stream_id] = b""
        self._stream_buffers[stream_id] += data

        if self._is_unidirectional(stream_id):
            # Uni stream: buffer until FIN, then parse GROUP + FRAMEs
            if end_stream:
                self._handle_uni_stream_complete(stream_id)
        elif stream_id == self._setup_stream_id:
            # Our setup stream — look for SERVER_SETUP response
            self._try_parse_server_setup(stream_id)
        elif self._is_server_initiated(stream_id):
            # Server-initiated bidi stream — could be SUBSCRIBE or ANNOUNCE
            self._handle_incoming_bidi(stream_id, end_stream)

    def _try_parse_server_setup(self, stream_id: int):
        """Try to parse SERVER_SETUP from buffered data on setup stream."""
        buf = self._stream_buffers.get(stream_id, b"")
        if len(buf) < 2:
            return

        # Look for 0x21 (SERVER_SETUP type byte)
        if buf[0] != SERVER_SETUP_TYPE:
            logger.warning(f"Expected SERVER_SETUP (0x21), got {buf[0]:#x}")
            return

        try:
            version, _ = MOQCodec.decode_server_setup(buf, 0)
            client_name = "moq-lite-02"
            server_name = "moq-lite-02" if version == MOQL_VERSION else f"unknown"
            logger.info(
                f"🟠 MOQ protocol: client={client_name} ({MOQL_VERSION:#x}), "
                f"relay={server_name} ({version:#x})"
            )
            if version != MOQL_VERSION:
                logger.error(
                    f"💔❌💔 PROTOCOL MISMATCH! "
                    f"Bot speaks {client_name} ({MOQL_VERSION:#x}) but "
                    f"relay responded with {server_name} ({version:#x}). "
                    f"Connection will likely fail!"
                )
            self._session.setup_complete = True
            self._setup_complete.set()
        except (IndexError, struct.error):
            # Incomplete data, wait for more
            pass

    def _handle_incoming_bidi(self, stream_id: int, end_stream: bool):
        """Handle data on a server-initiated bidi stream."""
        buf = self._stream_buffers.get(stream_id, b"")
        if len(buf) < 1:
            return

        # Determine stream type from first varint if not yet known
        if stream_id not in self._stream_types:
            try:
                stream_type, _ = MOQCodec.decode_varint(buf, 0)
                self._stream_types[stream_id] = stream_type
            except (IndexError, struct.error):
                return

        stream_type = self._stream_types.get(stream_id)

        if stream_type == STREAM_TYPE_SUBSCRIBE:
            self._handle_incoming_subscribe(stream_id, buf)
        elif stream_type == STREAM_TYPE_ANNOUNCE:
            self._handle_incoming_announce(stream_id, buf)
        else:
            logger.debug(f"Unknown incoming bidi stream type: {stream_type}")

    def _handle_incoming_subscribe(self, stream_id: int, buf: bytes):
        """Handle an incoming SUBSCRIBE from the relay.

        The relay sends SUBSCRIBE when a downstream subscriber wants our track.
        We respond with SUBSCRIBE_OK for known tracks, or reset the stream
        for unknown tracks so the relay can try other publishers.
        """
        try:
            # Skip stream type varint
            _, offset = MOQCodec.decode_varint(buf, 0)

            # Decode subscribe body
            subscribe_id, broadcast_path, track_name, priority, _ = MOQCodec.decode_subscribe(
                buf, offset
            )

            logger.debug(
                f"Received SUBSCRIBE for {broadcast_path}/{track_name} "
                f"(subscribe_id={subscribe_id}, priority={priority})"
            )

            # Only accept subscribes for tracks we actually publish, which
            # live inside our broadcast: "<namespace>/<participant_id>".
            our_broadcast = self._namespace + "/" + self._participant_id
            known_tracks = {
                our_broadcast + "/" + self._audio_track_name,
                our_broadcast + "/" + self._video_track_name,
                our_broadcast + "/" + self._transcript_track_name,
            }
            full_track = broadcast_path + "/" + track_name
            if full_track not in known_tracks:
                logger.debug(
                    f"Rejecting SUBSCRIBE for unknown track {full_track} "
                    f"(we only publish: {known_tracks})"
                )
                self._quic.reset_stream(stream_id, error_code=0)
                self.transmit()
                return

            # Store the subscribe_id for publishing to this subscriber
            self._publish_subscribe_ids[track_name] = subscribe_id
            self._subscribe_id_to_track[subscribe_id] = track_name

            # Send SUBSCRIBE_OK response on the same stream
            ok_data = MOQCodec.encode_subscribe_ok()
            self._quic.send_stream_data(stream_id, ok_data, end_stream=False)
            self.transmit()

            logger.debug(f"Sent SUBSCRIBE_OK for subscribe_id={subscribe_id}")

            # Notify transport that someone subscribed to our track
            if self._on_subscribe_received:
                asyncio.create_task(self._on_subscribe_received(track_name))

        except Exception as e:
            logger.error(f"Error handling incoming SUBSCRIBE: {e}")

    def _handle_incoming_announce(self, stream_id: int, buf: bytes):
        """Handle an incoming ANNOUNCE stream from the relay.

        The relay sends ANNOUNCE_PLEASE with a path prefix, and we respond
        with ANNOUNCE_INIT listing our broadcast path suffixes under that prefix.
        """
        try:
            # Skip stream type varint
            _, offset = MOQCodec.decode_varint(buf, 0)

            # This is ANNOUNCE_PLEASE from the relay
            path_prefix, _ = MOQCodec.decode_announce_please(buf, offset)
            logger.debug(f"Received ANNOUNCE_PLEASE for prefix: '{path_prefix}'")

            # Our broadcast path is "<namespace>/<participant_id>". Compute
            # its suffix relative to the relay-requested prefix.
            broadcast = self._namespace + "/" + self._participant_id
            if path_prefix and broadcast.startswith(path_prefix):
                suffix = broadcast[len(path_prefix) :].lstrip("/")
            elif not path_prefix:
                suffix = broadcast
            else:
                # Our broadcast doesn't match the prefix — announce nothing
                suffix = None

            if suffix is not None:
                logger.debug(f"Announcing broadcast suffix: '{suffix}'")
                init_data = MOQCodec.encode_announce_init([suffix])
            else:
                logger.debug("No matching broadcasts for prefix, announcing empty")
                init_data = MOQCodec.encode_announce_init([])

            self._quic.send_stream_data(stream_id, init_data)
            self.transmit()

        except Exception as e:
            logger.error(f"Error handling incoming ANNOUNCE: {e}")

    def _handle_uni_stream_complete(self, stream_id: int):
        """Handle a completed unidirectional stream (GROUP + FRAMEs)."""
        buf = self._stream_buffers.pop(stream_id, b"")
        if not buf:
            return

        try:
            subscribe_id, group_sequence, offset = MOQCodec.decode_group_header(buf, 0)
            frames = MOQCodec.decode_frames(buf, offset)

            track_name = self._subscribe_id_to_track.get(subscribe_id)

            for frame_data in frames:
                if not frame_data:
                    continue

                if subscribe_id == self._audio_subscribe_id and self._on_audio_frame:
                    asyncio.create_task(self._on_audio_frame(frame_data, 16000))
                elif subscribe_id == self._video_subscribe_id and self._on_video_frame:
                    asyncio.create_task(self._on_video_frame(frame_data, 0, 0))
                elif self._on_message:
                    asyncio.create_task(self._on_message(frame_data))

        except Exception as e:
            logger.error(f"Error processing uni stream data: {e}")

        # Clean up
        self._stream_types.pop(stream_id, None)

    async def send_client_setup(self, role: MOQRole, path: str = "/moq"):
        """Send CLIENT_SETUP message on a new bidi stream."""
        self._setup_stream_id = self._quic.get_next_available_stream_id(is_unidirectional=False)

        setup_msg = MOQCodec.encode_client_setup(role, [MOQL_VERSION], path)
        logger.info(
            f"CLIENT_SETUP ({len(setup_msg)} bytes) on stream {self._setup_stream_id}: "
            f"[{' '.join(f'{b:02x}' for b in setup_msg[:40])}]"
        )
        self._quic.send_stream_data(self._setup_stream_id, setup_msg)
        self.transmit()

        logger.debug(f"Sent CLIENT_SETUP on stream {self._setup_stream_id}")

    async def wait_for_setup(self, timeout: float = 10.0):
        """Wait for setup to complete."""
        try:
            await asyncio.wait_for(self._setup_complete.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error("Setup timeout")
            return False

    async def subscribe_track(
        self, namespace: str, track_name: str, track_type: MOQTrackType
    ) -> Optional[int]:
        """Subscribe to a track by opening a new bidi stream.

        moq-lite-02: Each subscription gets its own bidi stream.
        Stream format: varint(2) + SUBSCRIBE body, then read SUBSCRIBE_OK.
        """
        subscribe_id = self._session.next_subscribe_id()

        # Open a new bidi stream
        stream_id = self._quic.get_next_available_stream_id(is_unidirectional=False)

        # Write stream type + subscribe body
        msg = MOQCodec.encode_varint(STREAM_TYPE_SUBSCRIBE)
        msg += MOQCodec.encode_subscribe(subscribe_id, namespace, track_name)

        self._quic.send_stream_data(stream_id, msg, end_stream=False)
        self.transmit()

        # Track stream for detecting rejections (RESET_STREAM)
        self._subscribe_streams[stream_id] = subscribe_id

        # Track the subscribe_id for receiving data
        if track_type == MOQTrackType.AUDIO:
            self._audio_subscribe_id = subscribe_id
        elif track_type == MOQTrackType.VIDEO:
            self._video_subscribe_id = subscribe_id

        self._subscribe_id_to_track[subscribe_id] = track_name

        logger.debug(
            f"Subscribed to {namespace}/{track_name} "
            f"(subscribe_id={subscribe_id}, stream={stream_id})"
        )
        return subscribe_id

    async def publish_audio(self, audio: bytes, priority: int = 128):
        """Publish audio data on a new unidirectional stream.

        moq-lite-02: Each group is sent on its own uni stream with
        GROUP header + FRAME + FIN.
        """
        subscribe_id = self._publish_subscribe_ids.get(self._audio_track_name)
        if subscribe_id is None:
            return

        group_seq = self._session.get_next_group_sequence(subscribe_id)

        # Open uni stream, write GROUP header + FRAME, FIN
        stream_id = self._quic.get_next_available_stream_id(is_unidirectional=True)

        data = MOQCodec.encode_group_header(subscribe_id, group_seq)
        data += MOQCodec.encode_frame(audio)

        self._quic.send_stream_data(stream_id, data, end_stream=True)
        self.transmit()

    async def publish_video(self, image: bytes, priority: int = 129):
        """Publish video data on a new unidirectional stream."""
        subscribe_id = self._publish_subscribe_ids.get(self._video_track_name)
        if subscribe_id is None:
            return

        group_seq = self._session.get_next_group_sequence(subscribe_id)

        stream_id = self._quic.get_next_available_stream_id(is_unidirectional=True)

        data = MOQCodec.encode_group_header(subscribe_id, group_seq)
        data += MOQCodec.encode_frame(image)

        self._quic.send_stream_data(stream_id, data, end_stream=True)
        self.transmit()

    async def publish_text(self, payload: bytes) -> bool:
        """Publish a text/control payload on the transcript track.

        Returns False if no subscriber has SUBSCRIBE'd to our transcript
        track yet — caller can drop or buffer.
        """
        subscribe_id = self._publish_subscribe_ids.get(self._transcript_track_name)
        if subscribe_id is None:
            return False

        group_seq = self._session.get_next_group_sequence(subscribe_id)
        stream_id = self._quic.get_next_available_stream_id(is_unidirectional=True)

        data = MOQCodec.encode_group_header(subscribe_id, group_seq)
        data += MOQCodec.encode_frame(payload)

        self._quic.send_stream_data(stream_id, data, end_stream=True)
        self.transmit()
        return True


class MOQInputTransport(BaseInputTransport):
    """MOQ input transport for receiving media from a relay.

    Handles subscribing to tracks and receiving audio/video streams.
    """

    def __init__(
        self,
        transport: "MOQTransport",
        params: MOQParams,
        callbacks: MOQCallbacks,
        **kwargs,
    ):
        """Initialize the MOQ input transport."""
        super().__init__(params, **kwargs)

        self._moq_transport = transport
        self._params = params
        self._callbacks = callbacks
        self._protocol: Optional[MOQClientProtocol] = None
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the MOQ input transport and connect to relay."""
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        # Auto-connect to relay when pipeline starts
        self._moq_transport._connection_task = self.create_task(
            self._moq_transport.connect(), "moq_connect"
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

    def set_protocol(self, protocol: MOQClientProtocol):
        """Set the MOQ protocol handler."""
        self._protocol = protocol
        protocol.set_audio_callback(self._on_audio_received)
        protocol.set_video_callback(self._on_video_received)
        protocol.set_message_callback(self._on_message_received)

    async def _on_audio_received(self, audio: bytes, sample_rate: int):
        """Handle received audio data."""
        frame = InputAudioRawFrame(
            audio=audio,
            sample_rate=sample_rate,
            num_channels=1,
        )
        await self.push_audio_frame(frame)

    async def _on_video_received(self, image: bytes, width: int, height: int):
        """Handle received video data."""
        frame = InputImageRawFrame(
            image=image,
            size=(width, height),
            format="RGB",
        )
        await self.push_video_frame(frame)

    async def _on_message_received(self, message: bytes):
        """Handle received transport message."""
        await self.broadcast_frame(
            InputTransportMessageFrame,
            message=message.decode("utf-8"),
        )

    async def subscribe_audio(self):
        """Subscribe to the peer's audio track on the relay.

        Target broadcast is ``<namespace>/<peer_id>`` so the relay routes
        the SUBSCRIBE to the peer (e.g. the browser) and not back to us.
        Retries because the peer may not yet be announced when we first try.
        """
        if not self._protocol:
            return

        broadcast = self._params.namespace + "/" + self._params.peer_id
        track = self._params.audio_in_track_name
        max_retries = 10
        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep(1)
            logger.info(
                f"Subscribing to client audio: {broadcast}/{track}"
                f" (attempt {attempt + 1}/{max_retries})"
            )
            sub_id = await self._protocol.subscribe_track(
                broadcast,
                track,
                MOQTrackType.AUDIO,
            )
            # Wait for the relay to respond (accept or reset)
            await asyncio.sleep(0.5)
            # Check if this subscribe was rejected
            if sub_id in self._protocol._rejected_subscribe_ids:
                logger.warning(f"Subscribe attempt {attempt + 1} rejected, retrying...")
                continue
            logger.info(f"Subscribed to client audio: {broadcast}/{track}")
            return
        logger.error(f"Failed to subscribe to client audio after {max_retries} attempts")

    async def subscribe_video(self):
        """Subscribe to the peer's video track on the relay."""
        if self._protocol:
            await self._protocol.subscribe_track(
                self._params.namespace + "/" + self._params.peer_id,
                self._params.video_track_name,
                MOQTrackType.VIDEO,
            )


class MOQOutputTransport(BaseOutputTransport):
    """MOQ output transport for publishing media to a relay.

    In moq-lite-02, the announce flow is reversed: the relay/subscriber sends
    SUBSCRIBE to us. We register tracks locally and respond to incoming
    SUBSCRIBE requests with SUBSCRIBE_OK.
    """

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
        self._protocol: Optional[MOQClientProtocol] = None

        # Timing for pacing output
        self._send_interval = 0
        self._next_send_time = 0
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

    def set_protocol(self, protocol: MOQClientProtocol):
        """Set the MOQ protocol handler."""
        self._protocol = protocol

    async def announce_tracks(self):
        """Register tracks for publishing.

        In moq-lite-02, we don't proactively announce. The relay forwards
        subscriptions to us. We just need to be ready to respond to incoming
        SUBSCRIBE requests, which the protocol handler does automatically.
        """
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._next_send_time = 0

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message to subscribers on the transcript track.

        ``RTVIObserver`` (auto-attached to ``PipelineTask``) converts
        pipeline frames (transcriptions, bot output, speech events, …) into
        RTVI message models and pushes them through this method as
        ``OutputTransportMessage[Urgent]Frame`` instances whose ``message``
        is the model's JSON-ready dict. We serialize and send as-is — the
        browser parses RTVI message types client-side.
        """
        if not self._protocol:
            return
        payload = frame.message
        if not isinstance(payload, (bytes, bytearray)):
            payload = json.dumps(payload).encode("utf-8")
        try:
            await self._protocol.publish_text(payload)
        except Exception as e:
            logger.warning(f"Failed to publish transport message: {e}")

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the relay."""
        if not self._protocol:
            return False

        try:
            await self._protocol.publish_audio(frame.audio, self._params.priority)
            await self._write_audio_sleep()
            return True
        except Exception as e:
            logger.error(f"Error writing audio frame: {e}")
            return False

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the relay."""
        if not self._protocol:
            return False

        try:
            await self._protocol.publish_video(frame.image, self._params.priority + 1)
            return True
        except Exception as e:
            logger.error(f"Error writing video frame: {e}")
            return False

    async def _write_audio_sleep(self):
        """Pace audio output."""
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class MOQTransport(BaseTransport):
    """MOQ transport for connecting to a MOQ relay (moq-lite-02).

    Provides a complete MOQ client implementation with separate input and output
    transports for publishing and subscribing to media streams over QUIC.

    Example::

        transport = MOQTransport(
            params=MOQParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                role=MOQRole.PUBSUB,
                namespace="my-room",
            ),
            host="localhost",
            port=4080,
        )

        @transport.event_handler("on_connected")
        async def on_connected():
            print("Connected to MOQ relay")

        # Connect to relay
        await transport.connect()

    Event handlers available:

    - on_connected: Called when connection to relay is established
    - on_disconnected: Called when connection is lost
    - on_track_published: Called when a track is successfully published
    - on_track_subscribed: Called when a subscription is confirmed
    - on_error: Called when an error occurs
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

        self._host = host
        self._port = port
        self._path = path
        self._params = params

        self._callbacks = MOQCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_track_published=self._on_track_published,
            on_track_subscribed=self._on_track_subscribed,
            on_error=self._on_error,
        )

        self._input: Optional[MOQInputTransport] = None
        self._output: Optional[MOQOutputTransport] = None
        self._protocol: Optional[MOQClientProtocol] = None
        self._connection_task: Optional[asyncio.Task] = None

        self._client_connected = False

        # Register event handlers
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
            self._input = MOQInputTransport(
                self,
                self._params,
                self._callbacks,
                name=self._input_name,
            )
        return self._input

    def output(self) -> MOQOutputTransport:
        """Get the output transport for sending media."""
        if not self._output:
            self._output = MOQOutputTransport(
                self,
                self._params,
                name=self._output_name,
            )
        return self._output

    async def connect(self):
        """Connect to the MOQ relay using moq-lite-02 protocol."""
        logger.debug("MOQTransport.connect() starting")

        try:
            configuration = QuicConfiguration(
                alpn_protocols=[MOQL_ALPN],
                is_client=True,
            )

            # moq-lite-02 opens many streams
            configuration.max_stream_data = 1048576  # 1MB per stream

            if not self._params.verify_ssl:
                configuration.verify_mode = ssl.CERT_NONE

            # Force IPv4 for localhost — QUIC/UDP doesn't auto-fallback from IPv6
            connect_host = "127.0.0.1" if self._host == "localhost" else self._host

            # Set SNI explicitly — IP addresses don't carry SNI in TLS,
            # so the relay needs the original hostname for certificate lookup
            configuration.server_name = self._host

            logger.info(f"Connecting to MOQ relay at {connect_host}:{self._port}{self._path}")

            async with connect(
                connect_host,
                self._port,
                configuration=configuration,
                create_protocol=MOQClientProtocol,
            ) as protocol:
                self._protocol = protocol

                # Store identity on protocol for ANNOUNCE/SUBSCRIBE handling
                protocol._namespace = self._params.namespace
                protocol._participant_id = self._params.participant_id
                protocol._audio_track_name = self._params.audio_track_name
                protocol._video_track_name = self._params.video_track_name
                protocol._transcript_track_name = self._params.transcript_track_name
                protocol._on_subscribe_received = self._on_subscribe_received

                # Set up input/output with protocol
                if self._input:
                    self._input.set_protocol(protocol)
                if self._output:
                    self._output.set_protocol(protocol)

                # Send CLIENT_SETUP
                await protocol.send_client_setup(self._params.role, self._path)

                # Wait for SERVER_SETUP
                if not await protocol.wait_for_setup(self._params.connection_timeout):
                    await self._callbacks.on_error("Setup timeout", None)
                    return

                logger.info("MOQ setup complete, connection established")

                # Don't subscribe yet — wait for a client to connect first.
                # Subscribing now would fail with "not found" if no client
                # has announced yet. We subscribe in _on_subscribe_received()
                # when the relay forwards a SUBSCRIBE for our audio track,
                # which signals that a client is present.

                # In moq-lite-02, publishing is reactive: we wait for the relay
                # to send us SUBSCRIBE requests for our tracks. The protocol
                # handler responds automatically with SUBSCRIBE_OK.

                await self._callbacks.on_connected()

                # Keep connection alive
                while protocol._connected:
                    await asyncio.sleep(0.1)

                await self._callbacks.on_disconnected()

        except Exception as e:
            logger.error(f"MOQ connection error: {e}", exc_info=True)
            await self._callbacks.on_error(str(e), e)

    async def disconnect(self):
        """Disconnect from the MOQ relay."""
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        if self._protocol:
            self._protocol.close()
            self._protocol = None

    async def _on_connected(self):
        """Handle connection established."""
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self):
        """Handle connection lost."""
        await self._call_event_handler("on_disconnected")

    async def _on_track_published(self, track: MOQTrack):
        """Handle track published."""
        await self._call_event_handler("on_track_published", track)

    async def _on_track_subscribed(self, track: MOQTrack):
        """Handle track subscribed."""
        await self._call_event_handler("on_track_subscribed", track)

    async def _on_subscribe_received(self, track_name: str):
        """Handle incoming SUBSCRIBE from relay (a client wants our track)."""
        if track_name == self._params.audio_track_name and not self._client_connected:
            self._client_connected = True
            logger.info("Client connected (subscribed to our audio track)")

            # Now that a client is present, subscribe to their audio
            if self._params.role in (MOQRole.SUBSCRIBER, MOQRole.PUBSUB):
                if self._input:
                    if self._params.audio_in_enabled:
                        await self._input.subscribe_audio()

            await self._call_event_handler("on_client_connected")

    async def _on_error(self, message: str, exception: Optional[Exception]):
        """Handle error."""
        await self._call_event_handler("on_error", message, exception)

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) protocol implementation — moq-lite-02.

This module implements the moq-lite-02 protocol layer for pub/sub media
streaming over QUIC. It provides message types, encoding/decoding utilities,
and session management for real-time media transmission.

moq-lite-02 uses a stream-per-request model: each operation (setup,
subscribe, announce) opens its own bidirectional QUIC stream, and media
data flows on unidirectional streams as GROUP + FRAME sequences.

Based on moq-lite-02 (version code 0xff0dad02).
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

MOQL_VERSION = 0xFF0DAD02  # moq-lite-02
MOQL_ALPN = "moql"

# Stream type bytes (written as first varint on bidi streams)
STREAM_TYPE_SESSION = 0
STREAM_TYPE_ANNOUNCE = 1
STREAM_TYPE_SUBSCRIBE = 2

# Setup message type bytes (u8)
CLIENT_SETUP_TYPE = 0x20
SERVER_SETUP_TYPE = 0x21

# Unidirectional stream type (u8)
UNI_STREAM_TYPE_GROUP = 0

# Announce update status
ANNOUNCE_STATUS_ACTIVE = 0
ANNOUNCE_STATUS_ENDED = 1


class MOQRole(IntEnum):
    """MOQ endpoint roles."""

    PUBLISHER = 0x01
    SUBSCRIBER = 0x02
    PUBSUB = 0x03


class MOQTrackType(IntEnum):
    """Types of media tracks in MOQ."""

    AUDIO = 0x01
    VIDEO = 0x02
    DATA = 0x03


@dataclass
class MOQTrack:
    """Represents a MOQ media track.

    Parameters:
        broadcast_path: The broadcast path (e.g., "pipecat").
        name: The track name (e.g., "audio" or "video").
        track_type: The type of media track.
        priority: Track priority (lower is higher priority).
    """

    broadcast_path: str
    name: str
    track_type: MOQTrackType = MOQTrackType.DATA
    priority: int = 128

    @property
    def full_name(self) -> str:
        """Get the full track identifier."""
        return f"{self.broadcast_path}/{self.name}"


class MOQCodec:
    """Encoder/decoder for MOQ wire protocol messages (moq-lite-02)."""

    @staticmethod
    def encode_varint(value: int) -> bytes:
        """Encode an integer as a QUIC variable-length integer.

        Args:
            value: The integer to encode.

        Returns:
            The encoded bytes.
        """
        if value < 0x40:
            return struct.pack("!B", value)
        elif value < 0x4000:
            return struct.pack("!H", value | 0x4000)
        elif value < 0x40000000:
            return struct.pack("!I", value | 0x80000000)
        else:
            return struct.pack("!Q", value | 0xC000000000000000)

    @staticmethod
    def decode_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Decode a QUIC variable-length integer.

        Args:
            data: The bytes to decode from.
            offset: Starting offset in the data.

        Returns:
            Tuple of (decoded value, new offset).
        """
        first_byte = data[offset]
        length_bits = first_byte >> 6

        if length_bits == 0:
            return first_byte, offset + 1
        elif length_bits == 1:
            value = struct.unpack("!H", data[offset : offset + 2])[0] & 0x3FFF
            return value, offset + 2
        elif length_bits == 2:
            value = struct.unpack("!I", data[offset : offset + 4])[0] & 0x3FFFFFFF
            return value, offset + 4
        else:
            value = struct.unpack("!Q", data[offset : offset + 8])[0] & 0x3FFFFFFFFFFFFFFF
            return value, offset + 8

    @staticmethod
    def encode_string(value: str) -> bytes:
        """Encode a string with varint length prefix.

        Args:
            value: The string to encode.

        Returns:
            The encoded bytes.
        """
        encoded = value.encode("utf-8")
        return MOQCodec.encode_varint(len(encoded)) + encoded

    @staticmethod
    def decode_string(data: bytes, offset: int = 0) -> Tuple[str, int]:
        """Decode a length-prefixed string.

        Args:
            data: The bytes to decode from.
            offset: Starting offset in the data.

        Returns:
            Tuple of (decoded string, new offset).
        """
        length, offset = MOQCodec.decode_varint(data, offset)
        value = data[offset : offset + length].decode("utf-8")
        return value, offset + length

    # ------------------------------------------------------------------
    # Setup messages (on a dedicated bidi stream with stream_type=0)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_client_setup(
        role: MOQRole,
        supported_versions: List[int],
        path: Optional[str] = None,
    ) -> bytes:
        """Encode a CLIENT_SETUP message for raw QUIC (ALPN "moql").

        Format: u8(0x20) + varint(body_len) + body
        Body: varint(num_versions) + versions... + varint(num_params) + params...

        Args:
            role: The role this client is taking.
            supported_versions: List of supported MOQ versions.
            path: Optional connection path.

        Returns:
            The encoded message (including stream type).
        """
        body = MOQCodec.encode_varint(len(supported_versions))
        for version in supported_versions:
            body += MOQCodec.encode_varint(version)

        # Parameters
        num_params = 1  # role
        if path:
            num_params += 1

        body += MOQCodec.encode_varint(num_params)

        # Role parameter (key=0)
        body += MOQCodec.encode_varint(0)  # Parameter key
        body += MOQCodec.encode_varint(1)  # Parameter length
        body += MOQCodec.encode_varint(role)

        # Path parameter (key=1) if provided
        if path:
            body += MOQCodec.encode_varint(1)  # Parameter key
            path_bytes = path.encode("utf-8")
            body += MOQCodec.encode_varint(len(path_bytes))
            body += path_bytes

        # Frame: u8(0x20) + u16(body_len) + body
        # The relay uses Draft14 wire encoding for ALPN "moql",
        # which expects a big-endian u16 for the body length.
        msg = struct.pack("!B", CLIENT_SETUP_TYPE)
        msg += struct.pack("!H", len(body))
        msg += body

        return msg

    @staticmethod
    def decode_server_setup(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Decode a SERVER_SETUP message.

        Format: u8(0x21) + u16(body_len) + body
        Body: varint(selected_version) + params...

        The relay uses Draft14 wire encoding for ALPN "moql",
        which uses big-endian u16 for the body length.

        Args:
            data: The bytes to decode from.
            offset: Starting offset in the data.

        Returns:
            Tuple of (selected_version, new offset).
        """
        # Skip type byte (0x21)
        msg_type = data[offset]
        offset += 1
        assert msg_type == SERVER_SETUP_TYPE, f"Expected SERVER_SETUP (0x21), got {msg_type:#x}"

        # Body length (u16 big-endian, Draft14 wire encoding)
        body_len = struct.unpack("!H", data[offset : offset + 2])[0]
        offset += 2
        body_start = offset

        # Selected version
        version, offset = MOQCodec.decode_varint(data, offset)

        # Skip remaining params
        offset = body_start + body_len

        return version, offset

    # ------------------------------------------------------------------
    # Subscribe messages (on a dedicated bidi stream with stream_type=2)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_subscribe(
        subscribe_id: int,
        broadcast_path: str,
        track_name: str,
        priority: int = 128,
    ) -> bytes:
        """Encode a SUBSCRIBE message body.

        Body format: varint(sub_id) + string(broadcast_path) + string(track_name) + u8(priority)

        The stream type varint(2) is written separately by the caller.

        Args:
            subscribe_id: Unique subscription identifier.
            broadcast_path: Broadcast path (namespace).
            track_name: Name of the track.
            priority: Subscriber priority (u8).

        Returns:
            The encoded subscribe body with varint length prefix.
        """
        body = MOQCodec.encode_varint(subscribe_id)
        body += MOQCodec.encode_string(broadcast_path)
        body += MOQCodec.encode_string(track_name)
        body += struct.pack("!B", priority)

        # Wrap: varint(body_len) + body
        return MOQCodec.encode_varint(len(body)) + body

    @staticmethod
    def encode_subscribe_ok() -> bytes:
        """Encode a SUBSCRIBE_OK response.

        Format: varint(0) — empty body.

        Returns:
            The encoded SUBSCRIBE_OK.
        """
        return MOQCodec.encode_varint(0)

    @staticmethod
    def decode_subscribe(data: bytes, offset: int = 0) -> Tuple[int, str, str, int, int]:
        """Decode a SUBSCRIBE message body.

        Args:
            data: The bytes to decode from.
            offset: Starting offset.

        Returns:
            Tuple of (subscribe_id, broadcast_path, track_name, priority, new offset).
        """
        # Body length
        body_len, offset = MOQCodec.decode_varint(data, offset)
        body_end = offset + body_len

        subscribe_id, offset = MOQCodec.decode_varint(data, offset)
        broadcast_path, offset = MOQCodec.decode_string(data, offset)
        track_name, offset = MOQCodec.decode_string(data, offset)
        priority = data[offset]
        offset += 1

        return subscribe_id, broadcast_path, track_name, priority, body_end

    @staticmethod
    def decode_subscribe_ok(data: bytes, offset: int = 0) -> int:
        """Decode a SUBSCRIBE_OK response.

        Format: varint(body_len) where body_len should be 0.

        Args:
            data: The bytes to decode from.
            offset: Starting offset.

        Returns:
            New offset after decoding.
        """
        body_len, offset = MOQCodec.decode_varint(data, offset)
        return offset + body_len

    # ------------------------------------------------------------------
    # Announce messages (on a dedicated bidi stream with stream_type=1)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_announce_please(path_prefix: str) -> bytes:
        """Encode an ANNOUNCE_PLEASE message body.

        Body format: string(path_prefix)
        Framed as: varint(1) + varint(body_len) + body

        The stream type varint(1) is written separately by the caller.

        Args:
            path_prefix: The path prefix to request announcements for.

        Returns:
            The encoded announce_please body with varint length prefix.
        """
        body = MOQCodec.encode_string(path_prefix)
        return MOQCodec.encode_varint(len(body)) + body

    @staticmethod
    def encode_announce_init(suffixes: List[str]) -> bytes:
        """Encode an ANNOUNCE_INIT response.

        Body format: varint(count) + string(suffix)...
        Framed as: varint(body_len) + body

        Args:
            suffixes: List of path suffixes to announce.

        Returns:
            The encoded announce_init.
        """
        body = MOQCodec.encode_varint(len(suffixes))
        for suffix in suffixes:
            body += MOQCodec.encode_string(suffix)
        return MOQCodec.encode_varint(len(body)) + body

    @staticmethod
    def encode_announce_update(status: int, path_suffix: str) -> bytes:
        """Encode an ANNOUNCE_UPDATE message.

        Body format: u8(status) + string(path_suffix)
        Framed as: varint(body_len) + body

        Args:
            status: Announce status (0=active, 1=ended).
            path_suffix: The path suffix being updated.

        Returns:
            The encoded announce_update.
        """
        body = struct.pack("!B", status)
        body += MOQCodec.encode_string(path_suffix)
        return MOQCodec.encode_varint(len(body)) + body

    @staticmethod
    def decode_announce_please(data: bytes, offset: int = 0) -> Tuple[str, int]:
        """Decode an ANNOUNCE_PLEASE message body.

        Args:
            data: The bytes to decode from.
            offset: Starting offset.

        Returns:
            Tuple of (path_prefix, new offset).
        """
        body_len, offset = MOQCodec.decode_varint(data, offset)
        body_end = offset + body_len
        path_prefix, offset = MOQCodec.decode_string(data, offset)
        return path_prefix, body_end

    # ------------------------------------------------------------------
    # Data messages (on unidirectional streams)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_group_header(subscribe_id: int, group_sequence: int) -> bytes:
        """Encode a GROUP header for a unidirectional data stream.

        Format: u8(0) + varint(body_len) + varint(subscribe_id) + varint(group_sequence)

        Args:
            subscribe_id: The subscription ID this data is for.
            group_sequence: The group sequence number.

        Returns:
            The encoded GROUP header.
        """
        body = MOQCodec.encode_varint(subscribe_id)
        body += MOQCodec.encode_varint(group_sequence)

        msg = struct.pack("!B", UNI_STREAM_TYPE_GROUP)
        msg += MOQCodec.encode_varint(len(body))
        msg += body

        return msg

    @staticmethod
    def encode_frame(payload: bytes) -> bytes:
        """Encode a FRAME within a group.

        Format: varint(payload_len) + payload

        Args:
            payload: The media payload data.

        Returns:
            The encoded frame.
        """
        return MOQCodec.encode_varint(len(payload)) + payload

    @staticmethod
    def decode_group_header(data: bytes, offset: int = 0) -> Tuple[int, int, int]:
        """Decode a GROUP header from a unidirectional stream.

        Format: u8(0) + varint(body_len) + varint(subscribe_id) + varint(group_sequence)

        Args:
            data: The bytes to decode from.
            offset: Starting offset.

        Returns:
            Tuple of (subscribe_id, group_sequence, new offset after header).
        """
        # Stream type byte (should be 0)
        stream_type = data[offset]
        offset += 1

        # Body length
        body_len, offset = MOQCodec.decode_varint(data, offset)
        body_start = offset

        subscribe_id, offset = MOQCodec.decode_varint(data, offset)
        group_sequence, offset = MOQCodec.decode_varint(data, offset)

        # Ensure offset aligns with body_start + body_len
        offset = body_start + body_len

        return subscribe_id, group_sequence, offset

    @staticmethod
    def decode_frames(data: bytes, offset: int = 0) -> List[bytes]:
        """Decode all FRAMEs from remaining data after GROUP header.

        Format: varint(payload_len) + payload, repeated.

        Args:
            data: The bytes to decode from.
            offset: Starting offset (after GROUP header).

        Returns:
            List of payload byte arrays.
        """
        frames = []
        while offset < len(data):
            payload_len, offset = MOQCodec.decode_varint(data, offset)
            payload = data[offset : offset + payload_len]
            offset += payload_len
            frames.append(bytes(payload))
        return frames


class MOQSession:
    """Manages MOQ session state for moq-lite-02.

    Tracks subscriptions and group sequences for a single MOQ connection.
    In moq-lite-02, track aliases are gone — subscribe_id is used directly.
    """

    def __init__(self, role: MOQRole = MOQRole.PUBSUB):
        """Initialize the MOQ session.

        Args:
            role: The role for this session (publisher, subscriber, or both).
        """
        self.role = role
        self.version: int = MOQL_VERSION
        self.setup_complete: bool = False

        # Subscription tracking
        self._next_subscribe_id: int = 1

        # Group sequencing per subscribe_id (for publishing)
        self._group_sequences: Dict[int, int] = {}

    def next_subscribe_id(self) -> int:
        """Get the next available subscribe ID.

        Returns:
            The next subscribe ID.
        """
        sid = self._next_subscribe_id
        self._next_subscribe_id += 1
        return sid

    def get_next_group_sequence(self, subscribe_id: int) -> int:
        """Get the next group sequence number for publishing.

        Args:
            subscribe_id: The subscription ID to get the sequence for.

        Returns:
            The next group sequence number.
        """
        seq = self._group_sequences.get(subscribe_id, 0)
        self._group_sequences[subscribe_id] = seq + 1
        return seq

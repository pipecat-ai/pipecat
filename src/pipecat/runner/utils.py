#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transport utility functions and FastAPI route setup helpers.

This module provides common functionality for setting up transport-specific
FastAPI routes and handling WebRTC/WebSocket connections. It includes SDP
manipulation utilities for WebRTC compatibility and transport detection helpers.

Key features:

- WebRTC route setup with connection management
- WebSocket route setup for telephony providers
- SDP munging for ESP32 and other WebRTC compatibility
- Transport client ID detection across different transport types
- Video capture utilities for Daily transports

The utilities are designed to be transport-agnostic where possible, with
specific handlers for each transport type's unique requirements.

Example::

    from pipecat.runner.utils import parse_telephony_websocket

    async def telephony_websocket_handler(websocket: WebSocket):
        transport_type, stream_id, call_id = await parse_telephony_websocket(websocket)
"""

import json
import re
from typing import Any

from fastapi import WebSocket
from loguru import logger

from pipecat.transports.base_transport import BaseTransport


def _detect_transport_type_from_message(message_data: dict) -> str:
    """Attempt to auto-detect transport type from WebSocket message structure."""
    logger.trace("=== Auto-Detection Analysis ===")

    # Twilio detection
    if (
        message_data.get("event") == "start"
        and "start" in message_data
        and "streamSid" in message_data.get("start", {})
        and "callSid" in message_data.get("start", {})
    ):
        logger.trace("Auto-detected: TWILIO")
        return "twilio"

    # Telnyx detection
    if (
        "stream_id" in message_data
        and "start" in message_data
        and "call_control_id" in message_data.get("start", {})
    ):
        logger.trace("Auto-detected: TELNYX")
        return "telnyx"

    # Plivo detection
    if (
        "start" in message_data
        and "streamId" in message_data.get("start", {})
        and "callId" in message_data.get("start", {})
    ):
        logger.trace("Auto-detected: PLIVO")
        return "plivo"

    logger.trace("Auto-detection failed - unknown format")
    return "unknown"


async def parse_telephony_websocket(websocket: WebSocket):
    """Parse telephony WebSocket messages and return transport type and basic call data.

    Returns:
        tuple: (transport_type: str, stream_id: str, call_id: str)

    Example usage::

        transport_type, stream_id, call_id = await parse_telephony_websocket(websocket)
    """
    # Read first two messages
    start_data = websocket.iter_text()

    try:
        # First message
        first_message_raw = await start_data.__anext__()
        logger.trace(f"First message: {first_message_raw}")
        try:
            first_message = json.loads(first_message_raw)
        except json.JSONDecodeError:
            first_message = {}

        # Second message
        second_message_raw = await start_data.__anext__()
        logger.trace(f"Second message: {second_message_raw}")
        try:
            second_message = json.loads(second_message_raw)
        except json.JSONDecodeError:
            second_message = {}

        # Try auto-detection on both messages
        detected_type_first = _detect_transport_type_from_message(first_message)
        detected_type_second = _detect_transport_type_from_message(second_message)

        # Use the successful detection
        if detected_type_first != "unknown":
            transport_type = detected_type_first
            call_data = first_message
            logger.debug(f"Detected transport: {transport_type} (from first message)")
        elif detected_type_second != "unknown":
            transport_type = detected_type_second
            call_data = second_message
            logger.debug(f"Detected transport: {transport_type} (from second message)")
        else:
            transport_type = "unknown"
            call_data = second_message
            logger.warning("Could not auto-detect transport type")

        if transport_type == "twilio":
            start_data = call_data.get("start", {})
            stream_id = start_data.get("streamSid")
            call_id = start_data.get("callSid")

        elif transport_type == "telnyx":
            stream_id = call_data.get("stream_id")
            call_id = call_data.get("start", {}).get("call_control_id")

        elif transport_type == "plivo":
            start_data = call_data.get("start", {})
            stream_id = start_data.get("streamId")
            call_id = start_data.get("callId")

        else:
            stream_id = None
            call_id = None

        logger.debug(f"Parsed - Type: {transport_type}, StreamId: {stream_id}, CallId: {call_id}")
        return transport_type, stream_id, call_id

    except Exception as e:
        logger.error(f"Error parsing telephony WebSocket: {e}")
        raise


def get_install_command(transport: str) -> str:
    """Get the pip install command for a specific transport.

    Args:
        transport: The transport name.

    Returns:
        The pip install command string.
    """
    install_map = {
        "daily": "pip install pipecat-ai[daily]",
        "livekit": "pip install pipecat-ai[livekit]",
        "webrtc": "pip install pipecat-ai[webrtc]",
        "twilio": "pip install pipecat-ai[websocket]",
        "telnyx": "pip install pipecat-ai[websocket]",
        "plivo": "pip install pipecat-ai[websocket]",
    }
    return install_map.get(transport, f"pip install pipecat-ai[{transport}]")


def get_transport_client_id(transport: BaseTransport, client: Any) -> str:
    """Get client identifier from transport-specific client object.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.

    Returns:
        Client identifier string, empty if transport not supported.
    """
    # Import conditionally to avoid dependency issues
    try:
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

        if isinstance(transport, SmallWebRTCTransport):
            return client.pc_id
    except ImportError:
        pass

    try:
        from pipecat.transports.services.daily import DailyTransport

        if isinstance(transport, DailyTransport):
            return client["id"]
    except ImportError:
        pass

    logger.warning(f"Unable to get client id from unsupported transport {type(transport)}")
    return ""


async def maybe_capture_participant_camera(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant camera video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    try:
        from pipecat.transports.services.daily import DailyTransport

        if isinstance(transport, DailyTransport):
            await transport.capture_participant_video(
                client["id"], framerate=framerate, video_source="camera"
            )
    except ImportError:
        pass


async def maybe_capture_participant_screen(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant screen video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    try:
        from pipecat.transports.services.daily import DailyTransport

        if isinstance(transport, DailyTransport):
            await transport.capture_participant_video(
                client["id"], framerate=framerate, video_source="screenVideo"
            )
    except ImportError:
        pass


def _smallwebrtc_sdp_cleanup_ice_candidates(text: str, pattern: str) -> str:
    """Clean up ICE candidates in SDP text for SmallWebRTC.

    Args:
        text: SDP text to clean up.
        pattern: Pattern to match for candidate filtering.

    Returns:
        Cleaned SDP text with filtered ICE candidates.
    """
    result = []
    lines = text.splitlines()
    for line in lines:
        if re.search("a=candidate", line):
            if re.search(pattern, line) and not re.search("raddr", line):
                result.append(line)
        else:
            result.append(line)
    return "\r\n".join(result)


def _smallwebrtc_sdp_cleanup_fingerprints(text: str) -> str:
    """Remove unsupported fingerprint algorithms from SDP text.

    Args:
        text: SDP text to clean up.

    Returns:
        SDP text with sha-384 and sha-512 fingerprints removed.
    """
    result = []
    lines = text.splitlines()
    for line in lines:
        if not re.search("sha-384", line) and not re.search("sha-512", line):
            result.append(line)
    return "\r\n".join(result)


def smallwebrtc_sdp_munging(sdp: str, host: str) -> str:
    """Apply SDP modifications for SmallWebRTC compatibility.

    Args:
        sdp: Original SDP string.
        host: Host address for ICE candidate filtering.

    Returns:
        Modified SDP string with fingerprint and ICE candidate cleanup.
    """
    sdp = _smallwebrtc_sdp_cleanup_fingerprints(sdp)
    sdp = _smallwebrtc_sdp_cleanup_ice_candidates(sdp, host)
    return sdp

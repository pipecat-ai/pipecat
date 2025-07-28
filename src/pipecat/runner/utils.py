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

    from pipecat.runner.utils import setup_webrtc_routes

    app = FastAPI()
    setup_webrtc_routes(app, bot_runner_function, host="localhost")
"""

import json
import re
from typing import Any, Callable, Dict

from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.responses import RedirectResponse
from loguru import logger

from pipecat.transports.base_transport import BaseTransport


def detect_transport_type_from_message(message_data: dict) -> str:
    """Attempt to auto-detect transport type from WebSocket message structure."""
    logger.debug("=== Auto-Detection Analysis ===")

    # Twilio detection
    if (
        message_data.get("event") == "start"
        and "start" in message_data
        and "streamSid" in message_data.get("start", {})
        and "callSid" in message_data.get("start", {})
    ):
        logger.debug("Auto-detected: TWILIO")
        return "twilio"

    # Telnyx detection
    if (
        "stream_id" in message_data
        and "start" in message_data
        and "call_control_id" in message_data.get("start", {})
    ):
        logger.debug("Auto-detected: TELNYX")
        return "telnyx"

    # Plivo detection
    if (
        "start" in message_data
        and "streamId" in message_data.get("start", {})
        and "callId" in message_data.get("start", {})
    ):
        logger.debug("Auto-detected: PLIVO")
        return "plivo"

    logger.debug("Auto-detection failed - unknown format")
    return "unknown"


async def parse_telephony_websocket(websocket: WebSocket):
    """Parse telephony WebSocket messages and return transport type and basic call data.

    Returns:
        tuple: (transport_type: str, stream_id: str, call_id: str)
    """
    logger.info("=== Parsing Telephony WebSocket ===")

    # Read first two messages
    start_data = websocket.iter_text()

    try:
        # First message
        first_message_raw = await start_data.__anext__()
        logger.debug(f"First message: {first_message_raw}")
        try:
            first_message = json.loads(first_message_raw)
        except json.JSONDecodeError:
            first_message = {}

        # Second message
        second_message_raw = await start_data.__anext__()
        logger.debug(f"Second message: {second_message_raw}")
        try:
            second_message = json.loads(second_message_raw)
        except json.JSONDecodeError:
            second_message = {}

        # Try auto-detection on both messages
        detected_type_first = detect_transport_type_from_message(first_message)
        detected_type_second = detect_transport_type_from_message(second_message)

        # Use the successful detection
        if detected_type_first != "unknown":
            transport_type = detected_type_first
            call_data = first_message
            logger.info(f"Detected transport: {transport_type} (from first message)")
        elif detected_type_second != "unknown":
            transport_type = detected_type_second
            call_data = second_message
            logger.info(f"Detected transport: {transport_type} (from second message)")
        else:
            transport_type = "unknown"
            call_data = second_message
            logger.warning("Could not auto-detect transport type")

        # Extract just the essential fields
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

        logger.info(f"Parsed - Type: {transport_type}, StreamId: {stream_id}, CallId: {call_id}")
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


def smallwebrtc_sdp_cleanup_ice_candidates(text: str, pattern: str) -> str:
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


def smallwebrtc_sdp_cleanup_fingerprints(text: str) -> str:
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
    sdp = smallwebrtc_sdp_cleanup_fingerprints(sdp)
    sdp = smallwebrtc_sdp_cleanup_ice_candidates(sdp, host)
    return sdp


def setup_webrtc_routes(app: FastAPI, transport_runner: Callable, host: str = None):
    """Set up WebRTC routes for an app."""
    try:
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
    except ImportError as e:
        logger.error(
            f"WebRTC transport dependencies not installed. Install with: {get_install_command('webrtc')}"
        )
        logger.debug(f"Import error: {e}")
        return

    # Store connections by pc_id
    pcs_map: Dict[str, SmallWebRTCConnection] = {}

    # Mount the frontend at /
    app.mount("/client", SmallWebRTCPrebuiltUI)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root requests to client interface."""
        return RedirectResponse(url="/client/")

    @app.post("/api/offer")
    async def offer(request: dict, background_tasks: BackgroundTasks):
        """Handle WebRTC offer requests and manage peer connections."""
        pc_id = request.get("pc_id")

        if pc_id and pc_id in pcs_map:
            pipecat_connection = pcs_map[pc_id]
            logger.info(f"Reusing existing connection for pc_id: {pc_id}")
            await pipecat_connection.renegotiate(
                sdp=request["sdp"],
                type=request["type"],
                restart_pc=request.get("restart_pc", False),
            )
        else:
            pipecat_connection = SmallWebRTCConnection()
            await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                """Handle WebRTC connection closure and cleanup."""
                logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)

            # Run transport with the connection
            background_tasks.add_task(
                transport_runner, "webrtc", webrtc_connection=pipecat_connection
            )

        answer = pipecat_connection.get_answer()

        if host:
            answer["sdp"] = smallwebrtc_sdp_munging(answer["sdp"], host)

        # Updating the peer connection inside the map
        pcs_map[answer["pc_id"]] = pipecat_connection

        return answer

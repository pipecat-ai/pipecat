#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Common transport utility functions shared between server.py and run.py."""

import json
import os
import re
from typing import Any, Callable, Dict

from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from loguru import logger

from pipecat.transports.base_transport import BaseTransport


def get_install_command(transport: str) -> str:
    """Get the pip install command for a specific transport."""
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


def setup_websocket_routes(
    app: FastAPI, transport_runner: Callable, transport_type: str, proxy: str = None
):
    """Set up WebSocket routes for telephony providers."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/")
    async def start_call():
        """Handle telephony webhook and return XML response."""
        logger.debug(f"POST {transport_type.upper()} XML")

        if transport_type == "twilio":
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{proxy}/ws"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""
        elif transport_type == "telnyx":
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{proxy}/ws" bidirectionalMode="rtp"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""
        elif transport_type == "plivo":
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{proxy}/ws</Stream>
</Response>"""
        else:
            xml_content = "<Response></Response>"

        return HTMLResponse(content=xml_content, media_type="application/xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle WebSocket connections for telephony."""
        await websocket.accept()
        logger.debug("WebSocket connection accepted")

        # Parse transport-specific data
        start_data = websocket.iter_text()

        if transport_type == "twilio":
            await start_data.__anext__()
            call_data = json.loads(await start_data.__anext__())
            print(call_data, flush=True)
            stream_sid = call_data["start"]["streamSid"]
            call_sid = call_data["start"]["callSid"]
            call_info = {"stream_sid": stream_sid, "call_sid": call_sid}

        elif transport_type == "telnyx":
            await start_data.__anext__()
            call_data = json.loads(await start_data.__anext__())
            print(call_data, flush=True)
            stream_id = call_data["stream_id"]
            call_control_id = call_data["start"]["call_control_id"]
            outbound_encoding = call_data["start"]["media_format"]["encoding"]
            call_info = {
                "stream_id": stream_id,
                "call_control_id": call_control_id,
                "outbound_encoding": outbound_encoding,
            }

        elif transport_type == "plivo":
            start_message = json.loads(await start_data.__anext__())
            logger.debug(f"Received start message: {start_message}")

            start_info = start_message.get("start", {})
            stream_id = start_info.get("streamId")
            call_id = start_info.get("callId")

            if not stream_id:
                logger.error("No streamId found in start message")
                await websocket.close()
                return

            logger.info(f"WebSocket connection accepted for stream: {stream_id}, call: {call_id}")
            call_info = {"stream_id": stream_id, "call_id": call_id}
        else:
            call_info = {}

        # Run transport with the websocket connection
        await transport_runner(transport_type, websocket=websocket, call_info=call_info)

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
        transport_type, call_data = await parse_telephony_websocket(websocket)
"""

import json
import os
import re
from typing import Any, Callable, Dict, Optional

from fastapi import WebSocket
from loguru import logger

from pipecat.runner.types import (
    DailyRunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
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

    # Exotel detection
    if (
        message_data.get("event") == "start"
        and "start" in message_data
        and "stream_sid" in message_data.get("start", {})
        and "call_sid" in message_data.get("start", {})
        and "account_sid" in message_data.get("start", {})
    ):
        logger.trace("Auto-detected: EXOTEL")
        return "exotel"

    logger.trace("Auto-detection failed - unknown format")
    return "unknown"


async def parse_telephony_websocket(websocket: WebSocket):
    """Parse telephony WebSocket messages and return transport type and call data.

    Returns:
        tuple: (transport_type: str, call_data: dict)

        call_data contains provider-specific fields:

        - Twilio::

            {
                "stream_id": str,
                "call_id": str,
                "body": dict
            }

        - Telnyx::

            {
                "stream_id": str,
                "call_control_id": str,
                "outbound_encoding": str,
                "from": str,
                "to": str,
            }

        - Plivo::

            {
                "stream_id": str,
                "call_id": str,
            }

        - Exotel::

            {
                "stream_id": str,
                "call_id": str,
                "account_sid": str,
                "from": str,
                "to": str,
            }

    Example usage::

        transport_type, call_data = await parse_telephony_websocket(websocket)
        if transport_type == "twilio":
            user_id = call_data["body"]["user_id"]
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
            call_data_raw = first_message
            logger.debug(f"Detected transport: {transport_type} (from first message)")
        elif detected_type_second != "unknown":
            transport_type = detected_type_second
            call_data_raw = second_message
            logger.debug(f"Detected transport: {transport_type} (from second message)")
        else:
            transport_type = "unknown"
            call_data_raw = second_message
            logger.warning("Could not auto-detect transport type")

        # Extract provider-specific data
        if transport_type == "twilio":
            start_data = call_data_raw.get("start", {})
            body_data = start_data.get("customParameters", {})
            call_data = {
                "stream_id": start_data.get("streamSid"),
                "call_id": start_data.get("callSid"),
                # All custom parameters
                "body": body_data,
            }

        elif transport_type == "telnyx":
            call_data = {
                "stream_id": call_data_raw.get("stream_id"),
                "call_control_id": call_data_raw.get("start", {}).get("call_control_id"),
                "outbound_encoding": call_data_raw.get("start", {})
                .get("media_format", {})
                .get("encoding"),
                "from": call_data_raw.get("start", {}).get("from", ""),
                "to": call_data_raw.get("start", {}).get("to", ""),
            }

        elif transport_type == "plivo":
            start_data = call_data_raw.get("start", {})
            call_data = {
                "stream_id": start_data.get("streamId"),
                "call_id": start_data.get("callId"),
            }

        elif transport_type == "exotel":
            start_data = call_data_raw.get("start", {})
            call_data = {
                "stream_id": start_data.get("stream_sid"),
                "call_id": start_data.get("call_sid"),
                "account_sid": start_data.get("account_sid"),
                "from": start_data.get("from", ""),
                "to": start_data.get("to", ""),
            }

        else:
            call_data = {}

        logger.debug(f"Parsed - Type: {transport_type}, Data: {call_data}")
        return transport_type, call_data

    except Exception as e:
        logger.error(f"Error parsing telephony WebSocket: {e}")
        raise


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
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        if isinstance(transport, SmallWebRTCTransport):
            return client.pc_id
    except ImportError:
        pass

    try:
        from pipecat.transports.daily.transport import DailyTransport

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
        from pipecat.transports.daily.transport import DailyTransport

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
        from pipecat.transports.daily.transport import DailyTransport

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
    logger.debug("Removing unsupported ICE candidates from SDP")
    result = []
    lines = text.splitlines()
    for line in lines:
        if re.search("a=candidate", line):
            if re.search(pattern, line) and not re.search("raddr", line):
                result.append(line)
        else:
            result.append(line)
    return "\r\n".join(result) + "\r\n"


def _smallwebrtc_sdp_cleanup_fingerprints(text: str) -> str:
    """Remove unsupported fingerprint algorithms from SDP text.

    Args:
        text: SDP text to clean up.

    Returns:
        SDP text with sha-384 and sha-512 fingerprints removed.
    """
    logger.debug("Removing unsupported fingerprints from SDP")
    result = []
    lines = text.splitlines()
    for line in lines:
        if not re.search("sha-384", line) and not re.search("sha-512", line):
            result.append(line)
    return "\r\n".join(result) + "\r\n"


def smallwebrtc_sdp_munging(sdp: str, host: Optional[str]) -> str:
    """Apply SDP modifications for SmallWebRTC compatibility.

    Args:
        sdp: Original SDP string.
        host: Host address for ICE candidate filtering.

    Returns:
        Modified SDP string with fingerprint and ICE candidate cleanup.
    """
    sdp = _smallwebrtc_sdp_cleanup_fingerprints(sdp)
    if host:
        sdp = _smallwebrtc_sdp_cleanup_ice_candidates(sdp, host)
    return sdp


def _get_transport_params(transport_key: str, transport_params: Dict[str, Callable]) -> Any:
    """Get transport parameters from factory function.

    Args:
        transport_key: The transport key to look up
        transport_params: Dict mapping transport names to parameter factory functions

    Returns:
        Transport parameters from the factory function

    Raises:
        ValueError: If transport key is missing from transport_params
    """
    if transport_key not in transport_params:
        raise ValueError(
            f"Missing transport params for '{transport_key}'. "
            f"Please add '{transport_key}' key to your transport_params dict."
        )

    params = transport_params[transport_key]()
    logger.debug(f"Using transport params for {transport_key}")
    return params


async def _create_telephony_transport(
    websocket: WebSocket,
    params: Optional[Any] = None,
    transport_type: str = None,
    call_data: dict = None,
) -> BaseTransport:
    """Create a telephony transport with pre-parsed WebSocket data.

    Args:
        websocket: FastAPI WebSocket connection from telephony provider
        params: FastAPIWebsocketParams (required)
        transport_type: Pre-detected provider type ("twilio", "telnyx", "plivo")
        call_data: Pre-parsed call data dict with provider-specific fields

    Returns:
        Configured FastAPIWebsocketTransport ready for telephony use.
    """
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

    if params is None:
        raise ValueError(
            "FastAPIWebsocketParams must be provided. "
            "The serializer and add_wav_header will be set automatically."
        )

    # Always set add_wav_header to False for telephony
    params.add_wav_header = False

    logger.info(f"Using pre-detected telephony provider: {transport_type}")

    if transport_type == "twilio":
        from pipecat.serializers.twilio import TwilioFrameSerializer

        params.serializer = TwilioFrameSerializer(
            stream_sid=call_data["stream_id"],
            call_sid=call_data["call_id"],
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        )
    elif transport_type == "telnyx":
        from pipecat.serializers.telnyx import TelnyxFrameSerializer

        params.serializer = TelnyxFrameSerializer(
            stream_id=call_data["stream_id"],
            call_control_id=call_data["call_control_id"],
            outbound_encoding=call_data["outbound_encoding"],
            inbound_encoding="PCMU",  # Standard default
            api_key=os.getenv("TELNYX_API_KEY", ""),
        )
    elif transport_type == "plivo":
        from pipecat.serializers.plivo import PlivoFrameSerializer

        params.serializer = PlivoFrameSerializer(
            stream_id=call_data["stream_id"],
            call_id=call_data["call_id"],
            auth_id=os.getenv("PLIVO_AUTH_ID", ""),
            auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
        )
    elif transport_type == "exotel":
        from pipecat.serializers.exotel import ExotelFrameSerializer

        params.serializer = ExotelFrameSerializer(
            stream_sid=call_data["stream_id"],
            call_sid=call_data["call_id"],
        )
    else:
        raise ValueError(
            f"Unsupported telephony provider: {transport_type}. "
            f"Supported providers: twilio, telnyx, plivo, exotel"
        )

    return FastAPIWebsocketTransport(websocket=websocket, params=params)


async def create_transport(
    runner_args: Any, transport_params: Dict[str, Callable]
) -> BaseTransport:
    """Create a transport from runner arguments using factory functions.

    This function uses the clean transport_params factory pattern where users
    define a dictionary mapping transport names to parameter factory functions.

    Args:
        runner_args: Arguments from the runner.
        transport_params: Dict mapping transport names to parameter factory functions.
            Keys should be: "daily", "webrtc", "twilio", "telnyx", "plivo", "exotel"
            Values should be functions that return transport parameters when called.

    Returns:
        Configured transport instance.

    Raises:
        ValueError: If transport key is missing from transport_params or runner_args type is unsupported.
        ImportError: If required dependencies are not installed.

    Example::

        transport_params = {
            "daily": lambda: DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
            "twilio": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                # add_wav_header and serializer will be set automatically
            ),
            "telnyx": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                # add_wav_header and serializer will be set automatically
            ),
            "plivo": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                # add_wav_header and serializer will be set automatically
            ),
            "exotel": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                # add_wav_header and serializer will be set automatically
            ),
        }

        transport = await create_transport(runner_args, transport_params)
    """
    # Create transport based on runner args type
    if isinstance(runner_args, DailyRunnerArguments):
        params = _get_transport_params("daily", transport_params)

        from pipecat.transports.daily.transport import DailyTransport

        return DailyTransport(
            runner_args.room_url,
            runner_args.token,
            "Pipecat Bot",
            params=params,
        )

    elif isinstance(runner_args, SmallWebRTCRunnerArguments):
        params = _get_transport_params("webrtc", transport_params)

        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        return SmallWebRTCTransport(
            params=params,
            webrtc_connection=runner_args.webrtc_connection,
        )

    elif isinstance(runner_args, WebSocketRunnerArguments):
        # Parse once to determine the provider and get data
        transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
        params = _get_transport_params(transport_type, transport_params)

        # Create telephony transport with pre-parsed data
        return await _create_telephony_transport(
            runner_args.websocket, params, transport_type, call_data
        )

    else:
        raise ValueError(f"Unsupported runner arguments type: {type(runner_args)}")

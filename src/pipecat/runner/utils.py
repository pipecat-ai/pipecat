#
# Copyright (c) 2024-2026, Daily
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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fastapi import WebSocket
from loguru import logger

from pipecat.runner.types import (
    CallData,
    DailyRunnerArguments,
    EvalRunnerArguments,
    ExotelCallData,
    LiveKitRunnerArguments,
    SmallWebRTCRunnerArguments,
    TelnyxCallData,
    VonageRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams

if TYPE_CHECKING:
    # Imported for type-checking only so the typed guard functions (e.g.
    # _is_daily) can narrow to the concrete transport types
    from typing import TypeGuard

    from pipecat.transports.daily.transport import DailyTransport
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat.transports.vonage.video_connector import VonageVideoConnectorTransport


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

    Args:
        websocket: FastAPI WebSocket connection from telephony provider.

    Returns:
        tuple: (transport_type: str, call_data: CallData)

        ``call_data`` is a :class:`~pipecat.runner.types.CallData` model with typed
        attribute access (``call_data.to_number``) that is also dict-compatible
        (``call_data["call_id"]``, ``call_data.get("body", {})``). Fields populated
        per provider:

        - Twilio::

            {
                "stream_id": str,
                "call_id": str,
                "body": dict
            }

        - Telnyx::

            {
                "stream_id": str,
                "call_id": str,  # normalized from Telnyx's call_control_id
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

    Raises:
        ValueError: If WebSocket closes before sending any messages.

    Example usage::

        transport_type, call_data = await parse_telephony_websocket(websocket)
        caller = call_data.from_number  # typed attribute access
        user_id = call_data.body.get("user_id")  # custom params still a dict
        # dict-style access also works: call_data["call_id"]

    The parsed result is cached on the websocket, so this is idempotent: the
    underlying ``websocket.iter_text()`` stream is single-use, but calling this
    function again (e.g. once inside ``create_transport`` and once in bot code)
    returns the same ``(transport_type, call_data)`` without re-consuming it.
    """
    # Return the cached parse if this websocket has already been parsed — the
    # message stream below can only be consumed once. The cache is always a
    # (transport_type, call_data) tuple; isinstance keeps this robust against
    # mock/auto-attr websockets in tests.
    cached = getattr(websocket, "_pipecat_parsed_telephony", None)
    if isinstance(cached, tuple):
        return cached

    # Read first two messages
    message_stream = websocket.iter_text()
    first_message = {}
    second_message = {}

    try:
        # First message - required
        first_message_raw = await message_stream.__anext__()
        logger.trace(f"First message: {first_message_raw}")
        first_message = json.loads(first_message_raw) if first_message_raw else {}
    except json.JSONDecodeError:
        pass
    except StopAsyncIteration:
        raise ValueError("WebSocket closed before receiving telephony handshake messages")

    try:
        # Second message - optional, some providers may only send one
        second_message_raw = await message_stream.__anext__()
        logger.trace(f"Second message: {second_message_raw}")
        second_message = json.loads(second_message_raw) if second_message_raw else {}
    except json.JSONDecodeError:
        pass
    except StopAsyncIteration:
        logger.warning("Only received one WebSocket message, expected two")

    try:
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
                # Promote common custom params so the typed API is uniform across
                # providers (Twilio carries from/to as TwiML stream parameters).
                "from": body_data.get("from_number"),
                "to": body_data.get("to_number"),
            }

        elif transport_type == "telnyx":
            call_data = {
                "stream_id": call_data_raw.get("stream_id"),
                # Telnyx's call identifier is its call_control_id; normalize it onto
                # the common `call_id` field.
                "call_id": call_data_raw.get("start", {}).get("call_control_id"),
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
                "custom_parameters": start_data.get("custom_parameters", ""),
            }

        else:
            call_data = {}

        logger.debug(f"Parsed - Type: {transport_type}, Data: {call_data}")
        # Return a typed, dict-compatible CallData model (attribute access for new
        # code; subscript/.get for existing dict-style code). model_validate maps the
        # wire keys (e.g. "from"/"to") onto the aliased fields. Providers with extra
        # fields get a specific subclass so those fields are typed, not just extras.
        call_data_type = {"telnyx": TelnyxCallData, "exotel": ExotelCallData}.get(
            transport_type, CallData
        )
        result = (transport_type, call_data_type.model_validate(call_data))
        # Cache on the websocket so subsequent calls don't re-consume the stream.
        # Only successful parses are cached; the raising paths stay retryable.
        # setattr (not attribute assignment) since WebSocket has no such declared
        # field; mirrors the getattr-based read above.
        setattr(websocket, "_pipecat_parsed_telephony", result)
        return result

    except Exception as e:
        logger.error(f"Error parsing telephony WebSocket: {e}")
        raise


def _transport_is(transport: BaseTransport, class_name: str) -> bool:
    """Return whether ``transport`` is an instance of ``class_name``.

    Do this without importing, to avoid triggering import-time errors for
    transports that aren't installed and aren't needed.

    Assumes transport class names are unique, so that matching by name alone
    (e.g. "DailyTransport") is sufficient to identify it.

    Args:
        transport: The transport instance to check.
        class_name: Unqualified name of the transport class to match.

    Returns:
        ``True`` if ``transport`` is an instance of ``class_name``, else ``False``.
    """
    candidates = {type(transport), transport.__class__}
    return any(base.__name__ == class_name for klass in candidates for base in klass.__mro__)


# Typed guards over _transport_is.
# They narrow the transport to its concrete type so call sites can use
# transport-specific methods/properties in a type-checker- and
# auto-complete-friendly way.
def _is_daily(transport: BaseTransport) -> "TypeGuard[DailyTransport]":
    return _transport_is(transport, "DailyTransport")


def _is_smallwebrtc(transport: BaseTransport) -> "TypeGuard[SmallWebRTCTransport]":
    return _transport_is(transport, "SmallWebRTCTransport")


def _is_vonage(transport: BaseTransport) -> "TypeGuard[VonageVideoConnectorTransport]":
    return _transport_is(transport, "VonageVideoConnectorTransport")


def get_transport_client_id(transport: BaseTransport, client: Any) -> str:
    """Get client identifier from transport-specific client object.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.

    Returns:
        Client identifier string, empty if transport not supported.
    """
    if _is_smallwebrtc(transport):
        return client.pc_id
    if _is_daily(transport):
        return client["id"]
    if _is_vonage(transport):
        return client["streamId"]

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
    if _is_daily(transport):
        await transport.capture_participant_video(
            client["id"], framerate=framerate, video_source="camera"
        )
    elif _is_smallwebrtc(transport):
        await transport.capture_participant_video(video_source="camera")
    elif _is_vonage(transport):
        # Imported in-branch (not at module scope) to avoid a hard Vonage dependency;
        # we only get here when the transport is Vonage, so the extra is installed.
        from pipecat.transports.vonage.video_connector import SubscribeSettings

        await transport.subscribe_to_stream(
            client["streamId"],
            SubscribeSettings(
                subscribe_to_audio=True,
                subscribe_to_video=True,
                preferred_framerate=framerate if framerate != 0 else None,
            ),
        )


async def maybe_capture_participant_screen(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant screen video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    if _is_daily(transport):
        await transport.capture_participant_video(
            client["id"], framerate=framerate, video_source="screenVideo"
        )
    elif _is_smallwebrtc(transport):
        await transport.capture_participant_video(video_source="screenVideo")


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


def smallwebrtc_sdp_munging(sdp: str, host: str | None) -> str:
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


def _get_transport_params(transport_key: str, transport_params: dict[str, Callable]) -> Any:
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
    params: Any,
    transport_type: str,
    call_data: CallData,
) -> BaseTransport:
    """Create a telephony transport with pre-parsed WebSocket data.

    Args:
        websocket: FastAPI WebSocket connection from telephony provider
        params: FastAPIWebsocketParams (required)
        transport_type: Pre-detected provider type ("twilio", "telnyx", "plivo")
        call_data: Pre-parsed :class:`CallData` with provider-specific fields

    Returns:
        Configured FastAPIWebsocketTransport ready for telephony use.
    """
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

    # Always set add_wav_header to False for telephony
    params.add_wav_header = False

    logger.info(f"Using pre-detected telephony provider: {transport_type}")

    # Build serializers from the raw wire values via subscript access (the detected
    # provider guarantees these identifier fields are present). Bots use the typed
    # attribute API instead — call_data.from_number, call_data.call_id, etc.
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
            call_control_id=call_data["call_id"],
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


def _maybe_apply_daily_dialin(params: Any, body: Any) -> None:
    """Wire Daily PSTN dial-in settings from ``runner_args.body`` into ``DailyParams``.

    The dev runner places a ``DailyDialinRequest`` in ``runner_args.body`` for
    inbound PSTN calls. When present, merge its dial-in settings (and the Daily
    API key/url it carries, which are load-bearing for the pinless handshake) into
    the ``DailyParams`` the bot's factory produced. No-op when ``body`` doesn't
    carry dial-in, so non-dial-in Daily bots are unaffected.

    Args:
        params: The ``DailyParams`` instance from the bot's transport factory.
        body: ``runner_args.body`` — a ``DailyDialinRequest``, its ``model_dump()``
            dict, or unrelated content.
    """
    if not body:
        return

    from pipecat.runner.types import DailyDialinRequest

    try:
        if isinstance(body, DailyDialinRequest):
            request = body
        elif isinstance(body, dict) and "dialin_settings" in body:
            request = DailyDialinRequest.model_validate(body)
        else:
            return
    except Exception as e:
        logger.debug(f"runner_args.body present but not a Daily dial-in request, skipping: {e}")
        return

    from pipecat.transports.daily.transport import DailyDialinSettings

    params.dialin_settings = DailyDialinSettings(
        call_id=request.dialin_settings.call_id,
        call_domain=request.dialin_settings.call_domain,
    )
    # The dial-in request is authoritative for these (matches the inbound flow).
    params.api_key = request.daily_api_key
    params.api_url = request.daily_api_url


async def create_transport(
    runner_args: Any, transport_params: dict[str, Callable]
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
            ),
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
            "twilio": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                # add_wav_header and serializer will be set automatically
            ),
            "telnyx": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                # add_wav_header and serializer will be set automatically
            ),
            "plivo": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                # add_wav_header and serializer will be set automatically
            ),
            "exotel": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                # add_wav_header and serializer will be set automatically
            ),
            "vonage": lambda: VonageVideoConnectorTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True
            ),
        }

        transport = await create_transport(runner_args, transport_params)
    """
    # Create transport based on runner args type
    if isinstance(runner_args, DailyRunnerArguments):
        params = _get_transport_params("daily", transport_params)

        # Transparently wire PSTN dial-in (no-op when body has none).
        _maybe_apply_daily_dialin(params, runner_args.body)

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
        if runner_args.transport_type == "websocket":
            params = _get_transport_params("websocket", transport_params)
            from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

            return FastAPIWebsocketTransport(websocket=runner_args.websocket, params=params)

        # Parse once to determine the provider and get data
        transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)

        # Expose the parsed handshake to the bot so it can personalize the call
        # (caller lookup, routing, etc.) without re-parsing the single-use stream.
        runner_args.transport_type = transport_type
        runner_args.call_data = call_data

        params = _get_transport_params(transport_type, transport_params)

        # Create telephony transport with pre-parsed data
        return await _create_telephony_transport(
            runner_args.websocket, params, transport_type, call_data
        )
    elif isinstance(runner_args, LiveKitRunnerArguments):
        params = _get_transport_params("livekit", transport_params)

        from pipecat.transports.livekit.transport import LiveKitTransport

        return LiveKitTransport(
            runner_args.url,
            runner_args.token,
            runner_args.room_name,
            params=params,
        )
    elif isinstance(runner_args, EvalRunnerArguments):
        # The eval transport is a plain WebSocket server speaking RTVI. The
        # harness connects as an RTVI client; the bot pipeline must include an
        # RTVIProcessor and pass an RTVIObserver to the task. Default the
        # serializer to RTVIEvalSerializer so examples only need to opt into
        # audio input.
        from pipecat.evals.serializer import RTVIEvalSerializer
        from pipecat.evals.transport import EvalTransport, EvalTransportParams

        params = _get_transport_params("eval", transport_params)
        if not isinstance(params, EvalTransportParams):
            raise ValueError(
                "Eval transport params must be an EvalTransportParams instance. "
                "Set transport_params['eval'] to a lambda returning "
                "EvalTransportParams(audio_in_enabled=True)."
            )
        if params.serializer is None:
            params.serializer = RTVIEvalSerializer()

        # EvalTransport handles the eval-only behavior: the virtual mic, skip-TTS
        # before an on-connect greeting, and audio capture/recording.
        return EvalTransport(
            params=params,
            host=runner_args.host,
            port=runner_args.port,
        )
    elif isinstance(runner_args, VonageRunnerArguments):
        from pipecat.transports.vonage.video_connector import (
            VonageVideoConnectorTransport,
            VonageVideoConnectorTransportParams,
        )

        try:
            params = cast(
                VonageVideoConnectorTransportParams,
                _get_transport_params("vonage", transport_params),
            )
        except ValueError:
            webrtc_params: TransportParams = cast(
                TransportParams, _get_transport_params("webrtc", transport_params)
            )
            params = VonageVideoConnectorTransportParams(
                **webrtc_params.model_dump(),
                video_in_auto_subscribe=True,
            )

        return VonageVideoConnectorTransport(
            runner_args.application_id,
            runner_args.vonage_session_id,
            runner_args.token,
            params=params,
        )
    else:
        raise ValueError(f"Unsupported runner arguments type: {type(runner_args)}")

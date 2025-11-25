#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WhatsApp API Client.

This module provides a client for communicating with the WhatsApp Cloud API,
handling webhook requests, managing WebRTC connections, and processing
WhatsApp call events.
"""

import asyncio
import hashlib
import hmac
from typing import Awaitable, Callable, Dict, List, Optional

import aiohttp
from loguru import logger

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.whatsapp.api import (
    WhatsAppApi,
    WhatsAppConnectCall,
    WhatsAppConnectCallValue,
    WhatsAppTerminateCall,
    WhatsAppTerminateCallValue,
    WhatsAppWebhookRequest,
)


class WhatsAppClient:
    """WhatsApp Cloud API client for handling calls and webhook requests.

    This client manages WhatsApp call connections using WebRTC, processes webhook
    events from WhatsApp, and maintains ongoing call state. It supports both
    incoming call handling and call termination through the WhatsApp Cloud API.

    Attributes:
        _whatsapp_api: WhatsApp API instance for making API calls
        _ongoing_calls_map: Dictionary mapping call IDs to WebRTC connections
        _ice_servers: List of ICE servers for WebRTC connections
    """

    def __init__(
        self,
        whatsapp_token: str,
        phone_number_id: str,
        session: aiohttp.ClientSession,
        ice_servers: Optional[List[IceServer]] = None,
        whatsapp_secret: Optional[str] = None,
    ) -> None:
        """Initialize the WhatsApp client.

        Args:
            whatsapp_token: WhatsApp API access token
            phone_number_id: WhatsApp phone number ID for the business account
            session: aiohttp session for making HTTP requests
            ice_servers: List of ICE servers for WebRTC connections. If None,
                        defaults to Google's public STUN server
            whatsapp_secret: WhatsApp APP secret for validating that the webhook request came from WhatsApp.
        """
        self._whatsapp_api = WhatsAppApi(
            whatsapp_token=whatsapp_token, phone_number_id=phone_number_id, session=session
        )
        self._whatsapp_secret = whatsapp_secret
        self._ongoing_calls_map: Dict[str, SmallWebRTCConnection] = {}

        # Set default ICE servers if none provided
        if ice_servers is None:
            self._ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]
        else:
            self._ice_servers = ice_servers

    def update_ice_servers(self, ice_servers: Optional[List[IceServer]] = None):
        """Update the list of ICE servers used for WebRTC connections."""
        self._ice_servers = ice_servers

    def update_whatsapp_secret(self, whatsapp_secret: Optional[str] = None):
        """Update the WhatsApp APP secret for validating that the webhook request came from WhatsApp."""
        self._whatsapp_secret = whatsapp_secret

    def update_whatsapp_token(self, whatsapp_token: str):
        """Update the WhatsApp API access token."""
        self._whatsapp_api.update_whatsapp_token(whatsapp_token)

    def update_whatsapp_phone_number_id(self, phone_number_id: str):
        """Update the WhatsApp phone number ID for authentication."""
        self._whatsapp_api.update_whatsapp_phone_number_id(phone_number_id)

    async def terminate_all_calls(self) -> None:
        """Terminate all ongoing WhatsApp calls.

        This method will:
        1. Send termination requests to WhatsApp API for each ongoing call
        2. Disconnect all WebRTC connections
        3. Clear the ongoing calls map

        All terminations are executed concurrently for efficiency.
        """
        logger.debug("Will terminate all ongoing WhatsApp calls")

        if not self._ongoing_calls_map:
            logger.debug("No ongoing calls to terminate")
            return

        logger.debug(f"Terminating {len(self._ongoing_calls_map)} ongoing calls")

        # Terminate each call via WhatsApp API
        termination_tasks = []
        for call_id, pipecat_connection in self._ongoing_calls_map.items():
            logger.debug(f"Terminating call {call_id}")
            # Call WhatsApp API to terminate the call
            if self._whatsapp_api:
                termination_tasks.append(self._whatsapp_api.terminate_call_to_whatsapp(call_id))
            # Disconnect the pipecat connection
            termination_tasks.append(pipecat_connection.disconnect())

        # Execute all terminations concurrently
        await asyncio.gather(*termination_tasks, return_exceptions=True)

        # Clear the ongoing calls map
        self._ongoing_calls_map.clear()
        logger.debug("All calls terminated successfully")

    async def handle_verify_webhook_request(
        self, params: Dict[str, str], expected_verification_token: str
    ) -> int:
        """Handle a verify webhook request from WhatsApp.

        Args:
            params: Dictionary containing webhook parameters from query string
            expected_verification_token: The expected verification token to validate against

        Returns:
            int: The challenge value if verification succeeds

        Raises:
            ValueError: If verification fails due to missing parameters or invalid token
        """
        mode = params.get("hub.mode")
        challenge = params.get("hub.challenge")
        verify_token = params.get("hub.verify_token")

        if not mode or not challenge or not verify_token:
            raise ValueError("Missing required webhook verification parameters")

        if mode != "subscribe":
            raise ValueError(f"Invalid hub mode: expected 'subscribe', got '{mode}'")

        if verify_token != expected_verification_token:
            raise ValueError("Webhook verification token mismatch")

        return int(challenge)

    async def _validate_whatsapp_webhook_request(self, raw_body: bytes, sha256_signature: str):
        """Common handler for both /start and /connect endpoints."""
        # Compute HMAC SHA256 using your App Secret
        expected_signature = hmac.new(
            key=self._whatsapp_secret.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        # Extract signature from header (strip 'sha256=' prefix)
        if not sha256_signature:
            raise Exception("Missing X-Hub-Signature-256 header")
        received_signature = sha256_signature.split("sha256=")[-1]

        # Compare signatures securely
        if not hmac.compare_digest(expected_signature, received_signature):
            raise Exception("Invalid webhook signature")

        logger.debug(f"Webhook signature verified!")

    async def handle_webhook_request(
        self,
        request: WhatsAppWebhookRequest,
        connection_callback: Optional[Callable[[SmallWebRTCConnection], Awaitable[None]]] = None,
        raw_body: Optional[bytes] = None,
        sha256_signature: Optional[str] = None,
    ) -> bool:
        """Handle a webhook request from WhatsApp.

        This method processes incoming webhook requests and handles both
        connect and terminate events. For connect events, it establishes
        a WebRTC connection and optionally invokes a callback with the
        new connection.

        Args:
            request: The webhook request from WhatsApp containing call events
            connection_callback: Optional callback function to invoke when a new
                               WebRTC connection is established. The callback
                               receives the SmallWebRTCConnection instance.
            raw_body: Optional bytes containing the raw request body.
            sha256_signature: Optional X-Hub-Signature-256 header value from the request.

        Returns:
            bool: True if the webhook request was handled successfully, False otherwise

        Raises:
            ValueError: If the webhook request contains no supported events
            Exception: If connection establishment or API calls fail
        """
        try:
            if self._whatsapp_secret:
                await self._validate_whatsapp_webhook_request(raw_body, sha256_signature)
            for entry in request.entry:
                for change in entry.changes:
                    # Handle connect events
                    if isinstance(change.value, WhatsAppConnectCallValue):
                        for call in change.value.calls:
                            if call.event == "connect":
                                logger.debug(f"Processing connect event for call {call.id}")
                                try:
                                    connection = await self._handle_connect_event(call)

                                    # Invoke callback if provided
                                    if connection_callback and connection:
                                        try:
                                            await connection_callback(connection)
                                            logger.debug(
                                                f"Connection callback executed successfully for call {call.id}"
                                            )
                                        except Exception as callback_error:
                                            logger.error(
                                                f"Connection callback failed for call {call.id}: {callback_error}"
                                            )
                                            # Continue execution despite callback failure

                                    return True
                                except Exception as connect_error:
                                    logger.error(
                                        f"Failed to handle connect event for call {call.id}: {connect_error}"
                                    )
                                    raise

                    # Handle terminate events
                    elif isinstance(change.value, WhatsAppTerminateCallValue):
                        for call in change.value.calls:
                            if call.event == "terminate":
                                logger.debug(f"Processing terminate event for call {call.id}")
                                try:
                                    return await self._handle_terminate_event(call)
                                except Exception as terminate_error:
                                    logger.error(
                                        f"Failed to handle terminate event for call {call.id}: {terminate_error}"
                                    )
                                    raise

            # No supported events found
            error_msg = "No supported event found in webhook request"
            logger.warning(f"{error_msg}: {request}")
            raise ValueError(error_msg)

        except Exception as e:
            logger.error(f"Error processing webhook request: {e}")
            logger.debug(f"Webhook request details: {request}")
            raise

    def _filter_sdp_for_whatsapp(self, sdp: str) -> str:
        """Filter SDP to be compatible with WhatsApp requirements.

        WhatsApp only supports SHA-256 fingerprints, so this method removes
        other fingerprint types from the SDP.

        Args:
            sdp: The original SDP string

        Returns:
            Filtered SDP string compatible with WhatsApp
        """
        lines = sdp.splitlines()
        filtered = []
        for line in lines:
            if line.startswith("a=fingerprint:") and not line.startswith("a=fingerprint:sha-256"):
                continue  # drop sha-384 / sha-512
            filtered.append(line)
        return "\r\n".join(filtered) + "\r\n"

    async def _handle_connect_event(self, call: WhatsAppConnectCall) -> SmallWebRTCConnection:
        """Handle a CONNECT event by establishing WebRTC connection and accepting the call.

        This method:
        1. Creates a new WebRTC connection using configured ICE servers
        2. Initializes the connection with the provided SDP
        3. Generates an SDP answer and filters it for WhatsApp compatibility
        4. Pre-accepts the call with WhatsApp API
        5. Accepts the call with WhatsApp API
        6. Stores the connection for later management

        Args:
            call: WhatsApp connect call event

        Returns:
            The established SmallWebRTCConnection instance

        Raises:
            Exception: If pre-accept or accept API calls fail
        """
        logger.debug(f"Incoming call from {call.from_}, call_id: {call.id}")

        pipecat_connection = None
        try:
            # Create and initialize WebRTC connection
            pipecat_connection = SmallWebRTCConnection(self._ice_servers)
            await pipecat_connection.initialize(sdp=call.session.sdp, type=call.session.sdp_type)
            sdp_answer = pipecat_connection.get_answer().get("sdp")
            sdp_answer = self._filter_sdp_for_whatsapp(sdp_answer)

            logger.debug(f"SDP answer generated for call {call.id}")

            # Pre-accept the call
            try:
                pre_accept_resp = await self._whatsapp_api.answer_call_to_whatsapp(
                    call.id, "pre_accept", sdp_answer, call.from_
                )
                if not pre_accept_resp.get("success", False):
                    logger.error(f"Failed to pre-accept call {call.id}: {pre_accept_resp}")
                    raise Exception(f"Failed to pre-accept call: {pre_accept_resp}")

                logger.debug(f"Pre-accept successful for call {call.id}")
            except Exception as e:
                logger.error(f"Pre-accept API call failed for call {call.id}: {e}")
                raise Exception(f"Failed to pre-accept call: {e}")

            # Accept the call
            try:
                accept_resp = await self._whatsapp_api.answer_call_to_whatsapp(
                    call.id, "accept", sdp_answer, call.from_
                )
                if not accept_resp.get("success", False):
                    logger.error(f"Failed to accept call {call.id}: {accept_resp}")
                    raise Exception(f"Failed to accept call: {accept_resp}")

                logger.debug(f"Accept successful for call {call.id}")
            except Exception as e:
                logger.error(f"Accept API call failed for call {call.id}: {e}")
                raise Exception(f"Failed to accept call: {e}")

            # Store the connection for management
            self._ongoing_calls_map[call.id] = pipecat_connection

            # Set up disconnect handler
            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                logger.debug(
                    f"Peer connection closed: {webrtc_connection.pc_id} for call {call.id}"
                )
                # Clean up from ongoing calls map
                self._ongoing_calls_map.pop(call.id, None)

            logger.debug(f"WebRTC connection established successfully for call {call.id}")
            return pipecat_connection

        except Exception as e:
            # Clean up connection on failure
            if pipecat_connection:
                try:
                    await pipecat_connection.disconnect()
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to cleanup connection for call {call.id}: {cleanup_error}"
                    )

            logger.error(f"Failed to handle connect event for call {call.id}: {e}")
            raise

    async def _handle_terminate_event(self, call: WhatsAppTerminateCall) -> bool:
        """Handle a TERMINATE event by cleaning up resources and logging call completion.

        This method:
        1. Logs call termination details including duration if available
        2. Disconnects the associated WebRTC connection
        3. Removes the call from the ongoing calls map

        Args:
            call: WhatsApp terminate call event

        Returns:
            bool: True if the call was terminated successfully, False otherwise
        """
        logger.debug(f"Call terminated from {call.from_}, call_id: {call.id}")
        logger.debug(f"Call status: {call.status}")
        if call.duration:
            logger.debug(f"Call duration: {call.duration} seconds")

        try:
            if call.id in self._ongoing_calls_map:
                pipecat_connection = self._ongoing_calls_map[call.id]
                logger.debug(f"Disconnecting WebRTC connection for call {call.id}")

                try:
                    await pipecat_connection.disconnect()
                    logger.debug(f"WebRTC connection disconnected successfully for call {call.id}")
                except Exception as disconnect_error:
                    logger.error(
                        f"Failed to disconnect WebRTC connection for call {call.id}: {disconnect_error}"
                    )

                # Remove from ongoing calls map
                self._ongoing_calls_map.pop(call.id, None)
            else:
                logger.warning(f"Call {call.id} not found in ongoing calls map")

            return True

        except Exception as e:
            logger.error(f"Error handling terminate event for call {call.id}: {e}")
            return False

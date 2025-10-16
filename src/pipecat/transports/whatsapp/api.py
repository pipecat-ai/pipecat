#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WhatsApp API.

API to communicate with WhatsApp Cloud API.
"""

from typing import Any, Dict, List, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


# ----------------------------
# Pydantic Models for WhatsApp
# ----------------------------
class WhatsAppSession(BaseModel):
    """WebRTC session information for WhatsApp calls.

    Parameters:
        sdp: Session Description Protocol (SDP) data for WebRTC connection
        sdp_type: Type of SDP (e.g., "offer", "answer")
    """

    sdp: str
    sdp_type: str


class WhatsAppError(BaseModel):
    """Error information from WhatsApp API responses.

    Parameters:
        code: Error code number
        message: Human-readable error message
        href: URL for more information about the error
        error_data: Additional error-specific data
    """

    code: int
    message: str
    href: str
    error_data: Dict[str, Any]


class WhatsAppConnectCall(BaseModel):
    """Incoming call connection event data.

    Represents a user-initiated call that requires handling. This is sent
    when a WhatsApp user initiates a call to your business number.

    Parameters:
        id: Unique call identifier
        from_: Phone number of the caller (WhatsApp ID format)
        to: Your business phone number that received the call
        event: Always "connect" for incoming calls
        timestamp: ISO 8601 timestamp when the call was initiated
        direction: Optional call direction ("inbound" for user-initiated calls)
        session: WebRTC session data containing SDP offer from the caller
    """

    id: str
    from_: str = Field(..., alias="from")
    to: str
    event: str  # "connect"
    timestamp: str
    direction: Optional[str]
    session: WhatsAppSession


class WhatsAppTerminateCall(BaseModel):
    """Call termination event data.

    Represents the end of a call session, whether completed successfully,
    failed, or was rejected by either party.

    Parameters:
        id: Unique call identifier (matches the connect event)
        from_: Phone number of the caller
        to: Your business phone number
        event: Always "terminate" for call end events
        timestamp: ISO 8601 timestamp when the call ended
        direction: Optional call direction
        biz_opaque_callback_data: Optional business-specific callback data
        status: Call completion status ("FAILED", "COMPLETED", "REJECTED")
        start_time: ISO 8601 timestamp when call actually started (after acceptance)
        end_time: ISO 8601 timestamp when call ended
        duration: Call duration in seconds (only for completed calls)
    """

    id: str
    from_: str = Field(..., alias="from")
    to: str
    event: str  # "terminate"
    timestamp: str
    direction: Optional[str]
    biz_opaque_callback_data: Optional[str] = None
    status: Optional[str] = None  # "FAILED" or "COMPLETED" or "REJECTED"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[int] = None


class WhatsAppProfile(BaseModel):
    """User profile information.

    Parameters:
        name: Display name of the WhatsApp user
    """

    name: str


class WhatsAppContact(BaseModel):
    """Contact information for a WhatsApp user.

    Parameters:
        profile: User's profile information
        wa_id: WhatsApp ID (phone number in international format without +)
    """

    profile: WhatsAppProfile
    wa_id: str


class WhatsAppMetadata(BaseModel):
    """Business phone number metadata.

    Parameters:
        display_phone_number: Formatted phone number for display
        phone_number_id: WhatsApp Business API phone number ID
    """

    display_phone_number: str
    phone_number_id: str


class WhatsAppConnectCallValue(BaseModel):
    """Webhook payload for incoming call events.

    Parameters:
        messaging_product: Always "whatsapp"
        metadata: Business phone number information
        contacts: List of contact information for involved parties
        calls: List of call connection events
    """

    messaging_product: str
    metadata: WhatsAppMetadata
    contacts: List[WhatsAppContact]
    calls: List[WhatsAppConnectCall]


class WhatsAppTerminateCallValue(BaseModel):
    """Webhook payload for call termination events.

    Parameters:
        messaging_product: Always "whatsapp"
        metadata: Business phone number information
        calls: List of call termination events
        errors: Optional list of errors that occurred during the call
    """

    messaging_product: str
    metadata: WhatsAppMetadata
    calls: List[WhatsAppTerminateCall]
    errors: Optional[List[WhatsAppError]] = None


class WhatsAppChange(BaseModel):
    """Webhook change event wrapper.

    Parameters:
        value: The actual event data (connect or terminate)
        field: Always "calls" for calling webhooks
    """

    value: Union[WhatsAppConnectCallValue, WhatsAppTerminateCallValue]
    field: str


class WhatsAppEntry(BaseModel):
    """Webhook entry containing one or more changes.

    Parameters:
        id: WhatsApp Business Account ID
        changes: List of change events in this webhook delivery
    """

    id: str
    changes: List[WhatsAppChange]


class WhatsAppWebhookRequest(BaseModel):
    """Complete webhook request from WhatsApp.

    This is the top-level structure for all webhook deliveries from
    the WhatsApp Cloud API for calling events.

    Parameters:
        object: Always "whatsapp_business_account"
        entry: List of webhook entries (usually contains one entry)
    """

    object: str
    entry: List[WhatsAppEntry]


class WhatsAppApi:
    """WhatsApp Cloud API client for handling calls.

    This class provides methods to interact with the WhatsApp Cloud API
    for managing voice calls, including answering, rejecting, and terminating calls.

    Parameters:
        BASE_URL: Base URL for WhatsApp Graph API v23.0
        phone_number_id: Your WhatsApp Business phone number ID
        session: aiohttp client session for making HTTP requests
        whatsapp_url: Complete URL for the calls endpoint
        whatsapp_token: Bearer token for API authentication
    """

    BASE_URL = f"https://graph.facebook.com/v23.0/"

    def __init__(
        self, whatsapp_token: str, phone_number_id: str, session: aiohttp.ClientSession
    ) -> None:
        """Initialize the WhatsApp API client.

        Args:
            whatsapp_token: WhatsApp access token for authentication
            phone_number_id: Your business phone number ID from WhatsApp Business API
            session: aiohttp ClientSession for making HTTP requests
        """
        self._phone_number_id = phone_number_id
        self._session = session
        self._whatsapp_url = f"{self.BASE_URL}{phone_number_id}/calls"
        self._whatsapp_token = whatsapp_token

    def update_whatsapp_token(self, whatsapp_token: str):
        """Update the WhatsApp access token for authentication."""
        self._whatsapp_token = whatsapp_token

    def update_whatsapp_phone_number_id(self, phone_number_id: str):
        """Update the WhatsApp phone number ID for authentication."""
        self._phone_number_id = phone_number_id

    async def answer_call_to_whatsapp(self, call_id: str, action: str, sdp: str, from_: str):
        """Answer an incoming WhatsApp call.

        This method handles the call answering process, supporting both "pre_accept"
        and "accept" actions as required by the WhatsApp calling workflow.

        Args:
            call_id: Unique identifier for the call (from connect webhook)
            action: Action to perform ("pre_accept" or "accept")
            sdp: Session Description Protocol answer for WebRTC connection
            from_: Caller's phone number (WhatsApp ID format)

        Returns:
            Dict containing the API response with success status and any error details

        Note:
            Calls must be pre-accepted before being accepted. The typical flow is:
            1. Receive connect webhook
            2. Call with action="pre_accept"
            3. Call with action="accept"
        """
        logger.debug(f"Answering call {call_id} to WhatsApp, action:{action}")
        async with self._session.post(
            self._whatsapp_url,
            headers={
                "Authorization": f"Bearer {self._whatsapp_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": from_,
                "action": action,
                "call_id": call_id,
                "session": {"sdp": sdp, "sdp_type": "answer"},
            },
        ) as response:
            return await response.json()

    async def reject_call_to_whatsapp(self, call_id: str):
        """Reject an incoming WhatsApp call.

        This method rejects a call that was received via connect webhook.
        The caller will receive a rejection notification and a terminate
        webhook will be sent with status "REJECTED".

        Args:
            call_id: Unique identifier for the call (from connect webhook)

        Returns:
            Dict containing the API response with success status and any error details

        Note:
            This should be called instead of answer_call_to_whatsapp when you want
            to decline the incoming call. The caller will see the call as rejected.
        """
        logger.debug(f"Rejecting call {call_id}")
        async with self._session.post(
            self._whatsapp_url,
            headers={
                "Authorization": f"Bearer {self._whatsapp_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "action": "reject",
                "call_id": call_id,
            },
        ) as response:
            return await response.json()

    async def terminate_call_to_whatsapp(self, call_id: str):
        """Terminate an active WhatsApp call.

        This method ends an ongoing call that has been previously accepted.
        Both parties will be disconnected and a terminate webhook will be
        sent with status "COMPLETED".

        Args:
            call_id: Unique identifier for the active call

        Returns:
            Dict containing the API response with success status and any error details

        Note:
            This should only be called for calls that have been accepted and are
            currently active. For incoming calls that haven't been accepted yet,
            use reject_call_to_whatsapp instead.
        """
        logger.debug(f"Terminating call {call_id}")
        async with self._session.post(
            self._whatsapp_url,
            headers={
                "Authorization": f"Bearer {self._whatsapp_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "action": "terminate",
                "call_id": call_id,
            },
        ) as response:
            return await response.json()

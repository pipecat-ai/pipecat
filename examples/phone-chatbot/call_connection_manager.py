#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""call_connection_manager.py.

Manages customer/operator relationships and call routing for voice bots.
Provides mapping between customers and operators, and functions for retrieving
contact information. Also includes call state management.
"""

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger


class CallFlowState:
    """State for tracking call flow operations and state transitions."""

    def __init__(self):
        # Operator-related state
        self.dialed_operator = False
        self.operator_connected = False
        self.current_operator_index = 0
        self.operator_dialout_settings = []
        self.summary_finished = False

        # Voicemail detection state
        self.voicemail_detected = False
        self.human_detected = False
        self.voicemail_message_left = False

        # Call termination state
        self.call_terminated = False
        self.participant_left_early = False

    # Operator-related methods
    def set_operator_dialed(self):
        """Mark that an operator has been dialed."""
        self.dialed_operator = True

    def set_operator_connected(self):
        """Mark that an operator has connected to the call."""
        self.operator_connected = True
        # Summary is not finished when operator first connects
        self.summary_finished = False

    def set_operator_disconnected(self):
        """Handle operator disconnection."""
        self.operator_connected = False
        self.summary_finished = False

    def set_summary_finished(self):
        """Mark the summary as finished."""
        self.summary_finished = True

    def set_operator_dialout_settings(self, settings):
        """Set the list of operator dialout settings to try."""
        self.operator_dialout_settings = settings
        self.current_operator_index = 0

    def get_current_dialout_setting(self):
        """Get the current operator dialout setting to try."""
        if not self.operator_dialout_settings or self.current_operator_index >= len(
            self.operator_dialout_settings
        ):
            return None
        return self.operator_dialout_settings[self.current_operator_index]

    def move_to_next_operator(self):
        """Move to the next operator in the list."""
        self.current_operator_index += 1
        return self.get_current_dialout_setting()

    # Voicemail detection methods
    def set_voicemail_detected(self):
        """Mark that a voicemail system has been detected."""
        self.voicemail_detected = True
        self.human_detected = False

    def set_human_detected(self):
        """Mark that a human has been detected (not voicemail)."""
        self.human_detected = True
        self.voicemail_detected = False

    def set_voicemail_message_left(self):
        """Mark that a voicemail message has been left."""
        self.voicemail_message_left = True

    # Call termination methods
    def set_call_terminated(self):
        """Mark that the call has been terminated by the bot."""
        self.call_terminated = True

    def set_participant_left_early(self):
        """Mark that a participant left the call early."""
        self.participant_left_early = True


class SessionManager:
    """Centralized management of session IDs and state for all call participants."""

    def __init__(self):
        # Track session IDs of different participant types
        self.session_ids = {
            "operator": None,
            "customer": None,
            "bot": None,
            # Add other participant types as needed
        }

        # References for easy access in processors that need mutable containers
        self.session_id_refs = {
            "operator": [None],
            "customer": [None],
            "bot": [None],
            # Add other participant types as needed
        }

        # State object for call flow
        self.call_flow_state = CallFlowState()

    def set_session_id(self, participant_type, session_id):
        """Set the session ID for a specific participant type.

        Args:
            participant_type: Type of participant (e.g., "operator", "customer", "bot")
            session_id: The session ID to set
        """
        if participant_type in self.session_ids:
            self.session_ids[participant_type] = session_id

            # Also update the corresponding reference if it exists
            if participant_type in self.session_id_refs:
                self.session_id_refs[participant_type][0] = session_id

    def get_session_id(self, participant_type):
        """Get the session ID for a specific participant type.

        Args:
            participant_type: Type of participant (e.g., "operator", "customer", "bot")

        Returns:
            The session ID or None if not set
        """
        return self.session_ids.get(participant_type)

    def get_session_id_ref(self, participant_type):
        """Get the mutable reference for a specific participant type.

        Args:
            participant_type: Type of participant (e.g., "operator", "customer", "bot")

        Returns:
            A mutable list container holding the session ID or None if not available
        """
        return self.session_id_refs.get(participant_type)

    def is_participant_type(self, session_id, participant_type):
        """Check if a session ID belongs to a specific participant type.

        Args:
            session_id: The session ID to check
            participant_type: Type of participant (e.g., "operator", "customer", "bot")

        Returns:
            True if the session ID matches the participant type, False otherwise
        """
        return self.session_ids.get(participant_type) == session_id

    def reset_participant(self, participant_type):
        """Reset the state for a specific participant type.

        Args:
            participant_type: Type of participant (e.g., "operator", "customer", "bot")
        """
        if participant_type in self.session_ids:
            self.session_ids[participant_type] = None

            if participant_type in self.session_id_refs:
                self.session_id_refs[participant_type][0] = None

            # Additional reset actions for specific participant types
            if participant_type == "operator":
                self.call_flow_state.set_operator_disconnected()


class CallConfigManager:
    """Manages customer/operator relationships and call routing."""

    def __init__(self, body_data: Dict[str, Any] = None):
        """Initialize with optional body data.

        Args:
            body_data: Optional dictionary containing request body data
        """
        self.body = body_data or {}

        # Get environment variables with fallbacks
        self.dial_in_from_number = os.getenv("DIAL_IN_FROM_NUMBER", "+10000000001")
        self.dial_out_to_number = os.getenv("DIAL_OUT_TO_NUMBER", "+10000000002")
        self.operator_number = os.getenv("OPERATOR_NUMBER", "+10000000003")

        # Initialize maps with dynamic values
        self._initialize_maps()
        self._build_reverse_lookup_maps()

    def _initialize_maps(self):
        """Initialize the customer and operator maps with environment variables."""
        # Maps customer names to their contact information
        self.CUSTOMER_MAP = {
            "Dominic": {
                "phoneNumber": self.dial_in_from_number,  # I have two phone numbers, one for dialing in and one for dialing out. I give myself a separate name for each.
            },
            "Stewart": {
                "phoneNumber": self.dial_out_to_number,
            },
            "James": {
                "phoneNumber": "+10000000000",
                "callerId": "james-caller-id-uuid",
                "sipUri": "sip:james@example.com",
            },
            "Sarah": {
                "sipUri": "sip:sarah@example.com",
            },
            "Michael": {
                "phoneNumber": "+16505557890",
                "callerId": "michael-caller-id-uuid",
            },
        }

        # Maps customer names to their assigned operator names
        self.CUSTOMER_TO_OPERATOR_MAP = {
            "Dominic": ["Yunyoung", "Maria"],  # Try Yunyoung first, then Maria
            "Stewart": "Yunyoung",
            "James": "Yunyoung",
            "Sarah": "Jennifer",
            "Michael": "Paul",
            # Default mapping to ensure all customers have an operator
            "Default": "Yunyoung",
        }

        # Maps operator names to their contact details
        self.OPERATOR_CONTACT_MAP = {
            "Paul": {
                "phoneNumber": "+12345678904",
                "callerId": "paul-caller-id-uuid",
            },
            "Yunyoung": {
                "phoneNumber": self.operator_number,  # Dials out to my other phone number.
            },
            "Maria": {
                "sipUri": "sip:maria@example.com",
            },
            "Jennifer": {"phoneNumber": "+14155559876", "callerId": "jennifer-caller-id-uuid"},
            "Default": {
                "phoneNumber": self.operator_number,  # Use the operator number as default
            },
        }

    def _build_reverse_lookup_maps(self):
        """Build reverse lookup maps for phone numbers and SIP URIs to customer names."""
        self._PHONE_TO_CUSTOMER_MAP = {}
        self._SIP_TO_CUSTOMER_MAP = {}

        for customer_name, contact_info in self.CUSTOMER_MAP.items():
            if "phoneNumber" in contact_info:
                self._PHONE_TO_CUSTOMER_MAP[contact_info["phoneNumber"]] = customer_name
            if "sipUri" in contact_info:
                self._SIP_TO_CUSTOMER_MAP[contact_info["sipUri"]] = customer_name

    @classmethod
    def from_json_string(cls, json_string: str):
        """Create a CallRoutingManager from a JSON string.

        Args:
            json_string: JSON string containing body data

        Returns:
            CallRoutingManager instance with parsed data

        Raises:
            json.JSONDecodeError: If JSON string is invalid
        """
        body_data = json.loads(json_string)
        return cls(body_data)

    def find_customer_by_contact(self, contact_info: str) -> Optional[str]:
        """Find customer name from a contact identifier (phone number or SIP URI).

        Args:
            contact_info: The contact identifier (phone number or SIP URI)

        Returns:
            The customer name or None if not found
        """
        # Check if it's a phone number
        if contact_info in self._PHONE_TO_CUSTOMER_MAP:
            return self._PHONE_TO_CUSTOMER_MAP[contact_info]

        # Check if it's a SIP URI
        if contact_info in self._SIP_TO_CUSTOMER_MAP:
            return self._SIP_TO_CUSTOMER_MAP[contact_info]

        return None

    def get_customer_name(self, phone_number: str) -> Optional[str]:
        """Get customer name from their phone number.

        Args:
            phone_number: The customer's phone number

        Returns:
            The customer name or None if not found
        """
        # Note: In production, this would likely query a database
        return self.find_customer_by_contact(phone_number)

    def get_operators_for_customer(self, customer_name: Optional[str]) -> List[str]:
        """Get the operator name(s) assigned to a customer.

        Args:
            customer_name: The customer's name

        Returns:
            List of operator names (single item or multiple)
        """
        # Note: In production, this would likely query a database
        if not customer_name or customer_name not in self.CUSTOMER_TO_OPERATOR_MAP:
            return ["Default"]

        operators = self.CUSTOMER_TO_OPERATOR_MAP[customer_name]
        # Convert single string to list for consistency
        if isinstance(operators, str):
            return [operators]
        return operators

    def get_operator_dialout_settings(self, operator_name: str) -> Dict[str, str]:
        """Get an operator's dialout settings from their name.

        Args:
            operator_name: The operator's name

        Returns:
            Dictionary with dialout settings for the operator
        """
        # Note: In production, this would likely query a database
        return self.OPERATOR_CONTACT_MAP.get(operator_name, self.OPERATOR_CONTACT_MAP["Default"])

    def get_dialout_settings_for_caller(
        self, from_number: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Determine the appropriate operator dialout settings based on caller's number.

        This method uses the caller's number to look up the customer name,
        then finds the assigned operators for that customer, and returns
        an array of operator dialout settings to try in sequence.

        Args:
            from_number: The caller's phone number (from dialin_settings)

        Returns:
            List of operator dialout settings to try
        """
        if not from_number:
            # If we don't have dialin settings, use the Default operator
            return [self.get_operator_dialout_settings("Default")]

        # Get customer name from phone number
        customer_name = self.get_customer_name(from_number)

        # Get operator names assigned to this customer
        operator_names = self.get_operators_for_customer(customer_name)

        # Get dialout settings for each operator
        return [self.get_operator_dialout_settings(name) for name in operator_names]

    def get_caller_info(self) -> Dict[str, Optional[str]]:
        """Get caller and dialed numbers from dialin settings in the body.

        Returns:
            Dictionary containing caller_number and dialed_number
        """
        raw_dialin_settings = self.body.get("dialin_settings")
        if not raw_dialin_settings:
            return {"caller_number": None, "dialed_number": None}

        # Handle different case variations
        dialed_number = raw_dialin_settings.get("To") or raw_dialin_settings.get("to")
        caller_number = raw_dialin_settings.get("From") or raw_dialin_settings.get("from")

        return {"caller_number": caller_number, "dialed_number": dialed_number}

    def get_caller_number(self) -> Optional[str]:
        """Get the caller's phone number from dialin settings in the body.

        Returns:
            The caller's phone number or None if not available
        """
        return self.get_caller_info()["caller_number"]

    async def start_dialout(self, transport, dialout_settings=None):
        """Helper function to start dialout using the provided settings or from body.

        Args:
            transport: The transport instance to use for dialout
            dialout_settings: Optional override for dialout settings

        Returns:
            None
        """
        # Use provided settings or get from body
        settings = dialout_settings or self.get_dialout_settings()
        if not settings:
            logger.warning("No dialout settings available")
            return

        for setting in settings:
            if "phoneNumber" in setting:
                logger.info(f"Dialing number: {setting['phoneNumber']}")
                if "callerId" in setting:
                    logger.info(f"with callerId: {setting['callerId']}")
                    await transport.start_dialout(
                        {"phoneNumber": setting["phoneNumber"], "callerId": setting["callerId"]}
                    )
                else:
                    logger.info("with no callerId")
                    await transport.start_dialout({"phoneNumber": setting["phoneNumber"]})
            elif "sipUri" in setting:
                logger.info(f"Dialing sipUri: {setting['sipUri']}")
                await transport.start_dialout({"sipUri": setting["sipUri"]})
            else:
                logger.warning(f"Unknown dialout setting format: {setting}")

    def get_dialout_settings(self) -> Optional[List[Dict[str, Any]]]:
        """Extract dialout settings from the body.

        Returns:
            List of dialout setting objects or None if not present
        """
        # Check if we have dialout settings
        if "dialout_settings" in self.body:
            dialout_settings = self.body["dialout_settings"]

            # Convert to list if it's an object (for backward compatibility)
            if isinstance(dialout_settings, dict):
                return [dialout_settings]
            elif isinstance(dialout_settings, list):
                return dialout_settings

        return None

    def get_dialin_settings(self) -> Optional[Dict[str, Any]]:
        """Extract dialin settings from the body.

        Handles both camelCase and snake_case variations of fields for backward compatibility,
        but normalizes to snake_case for internal usage.

        Returns:
            Dictionary containing dialin settings or None if not present
        """
        raw_dialin_settings = self.body.get("dialin_settings")
        if not raw_dialin_settings:
            return None

        # Normalize dialin settings to handle different case variations
        # Prioritize snake_case (call_id, call_domain) but fall back to camelCase (callId, callDomain)
        dialin_settings = {
            "call_id": raw_dialin_settings.get("call_id") or raw_dialin_settings.get("callId"),
            "call_domain": raw_dialin_settings.get("call_domain")
            or raw_dialin_settings.get("callDomain"),
            "to": raw_dialin_settings.get("to") or raw_dialin_settings.get("To"),
            "from": raw_dialin_settings.get("from") or raw_dialin_settings.get("From"),
        }

        return dialin_settings

    # Bot prompt helper functions - no defaults provided, just return what's in the body

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Retrieve the prompt text for a given prompt name.

        Args:
            prompt_name: The name of the prompt to retrieve.

        Returns:
            The prompt string corresponding to the provided name, or None if not configured.
        """
        prompts = self.body.get("prompts", [])
        for prompt in prompts:
            if prompt.get("name") == prompt_name:
                return prompt.get("text")
        return None

    def get_transfer_mode(self) -> Optional[str]:
        """Get transfer mode from the body.

        Returns:
            Transfer mode string or None if not configured
        """
        if "call_transfer" in self.body:
            return self.body["call_transfer"].get("mode")
        return None

    def get_speak_summary(self) -> Optional[bool]:
        """Get speak summary from the body.

        Returns:
            Boolean indicating if summary should be spoken or None if not configured
        """
        if "call_transfer" in self.body:
            return self.body["call_transfer"].get("speakSummary")
        return None

    def get_store_summary(self) -> Optional[bool]:
        """Get store summary from the body.

        Returns:
            Boolean indicating if summary should be stored or None if not configured
        """
        if "call_transfer" in self.body:
            return self.body["call_transfer"].get("storeSummary")
        return None

    def is_test_mode(self) -> bool:
        """Check if running in test mode.

        Returns:
            Boolean indicating if test mode is enabled
        """
        if "voicemail_detection" in self.body:
            return bool(self.body["voicemail_detection"].get("testInPrebuilt"))
        if "call_transfer" in self.body:
            return bool(self.body["call_transfer"].get("testInPrebuilt"))
        if "simple_dialin" in self.body:
            return bool(self.body["simple_dialin"].get("testInPrebuilt"))
        if "simple_dialout" in self.body:
            return bool(self.body["simple_dialout"].get("testInPrebuilt"))
        return False

    def is_voicemail_detection_enabled(self) -> bool:
        """Check if voicemail detection is enabled in the body.

        Returns:
            Boolean indicating if voicemail detection is enabled
        """
        return bool(self.body.get("voicemail_detection"))

    def customize_prompt(self, prompt: str, customer_name: Optional[str] = None) -> str:
        """Insert customer name into prompt template if available.

        Args:
            prompt: The prompt template containing optional {customer_name} placeholders
            customer_name: Optional customer name to insert

        Returns:
            Customized prompt with customer name inserted
        """
        if customer_name and prompt:
            return prompt.replace("{customer_name}", customer_name)
        return prompt

    def create_system_message(self, content: str) -> Dict[str, str]:
        """Create a properly formatted system message.

        Args:
            content: The message content

        Returns:
            Dictionary with role and content for the system message
        """
        return {"role": "system", "content": content}

    def create_user_message(self, content: str) -> Dict[str, str]:
        """Create a properly formatted user message.

        Args:
            content: The message content

        Returns:
            Dictionary with role and content for the user message
        """
        return {"role": "user", "content": content}

    def get_customer_info_suffix(
        self, customer_name: Optional[str] = None, preposition: str = "for"
    ) -> str:
        """Create a consistent customer info suffix.

        Args:
            customer_name: Optional customer name
            preposition: Preposition to use before the name (e.g., "for", "to", "")

        Returns:
            String with formatted customer info suffix
        """
        if not customer_name:
            return ""

        # Add a space before the preposition if it's not empty
        space_prefix = " " if preposition else ""
        # For non-empty prepositions, add a space after it
        space_suffix = " " if preposition else ""

        return f"{space_prefix}{preposition}{space_suffix}{customer_name}"

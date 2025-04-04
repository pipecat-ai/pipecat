# bot_runner_helpers.py
from typing import Any, Dict, Optional

from bot_constants import (
    DEFAULT_CALLTRANSFER_MODE,
    DEFAULT_DIALIN_EXAMPLE,
    DEFAULT_SPEAK_SUMMARY,
    DEFAULT_STORE_SUMMARY,
    DEFAULT_TEST_IN_PREBUILT,
)
from call_connection_manager import CallConfigManager

# ----------------- Configuration Helpers ----------------- #


def determine_room_capabilities(config_body: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Determine room capabilities based on the configuration.

    This function examines the configuration to determine which capabilities
    the Daily room should have enabled.

    Args:
        config_body: Configuration dictionary that determines room capabilities

    Returns:
        Dictionary of capability flags
    """
    capabilities = {
        "enable_dialin": False,
        "enable_dialout": False,
        # Add more capabilities here in the future as needed
    }

    if not config_body:
        return capabilities

    # Check for dialin capability
    capabilities["enable_dialin"] = "dialin_settings" in config_body

    # Check for dialout capability - needed for outbound calls or transfers
    has_dialout_settings = "dialout_settings" in config_body

    # Check if there's a transfer to an operator configured
    has_call_transfer = "call_transfer" in config_body

    # Enable dialout if any condition requires it
    capabilities["enable_dialout"] = has_dialout_settings or has_call_transfer

    return capabilities


def ensure_dialout_settings_array(body: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures dialout_settings is an array of objects.

    Args:
        body: The configuration dictionary

    Returns:
        Updated configuration with dialout_settings as an array
    """
    if "dialout_settings" in body:
        # Convert to array if it's not already one
        if not isinstance(body["dialout_settings"], list):
            body["dialout_settings"] = [body["dialout_settings"]]

    return body


def ensure_prompt_config(body: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures the body has appropriate prompts settings, but doesn't add defaults.

    Only makes sure the prompt section exists, allowing the bot script to handle defaults.

    Args:
        body: The configuration dictionary

    Returns:
        Updated configuration with prompt settings section
    """
    if "prompts" not in body:
        body["prompts"] = []
    return body


def create_call_transfer_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create call transfer settings based on configuration and customer mapping.

    Args:
        body: The configuration dictionary

    Returns:
        Call transfer settings dictionary
    """
    # Default transfer settings
    transfer_settings = {
        "mode": DEFAULT_CALLTRANSFER_MODE,
        "speakSummary": DEFAULT_SPEAK_SUMMARY,
        "storeSummary": DEFAULT_STORE_SUMMARY,
        "testInPrebuilt": DEFAULT_TEST_IN_PREBUILT,
    }

    # If call_transfer already exists, merge the defaults with the existing settings
    # This ensures all required fields exist while preserving user-specified values
    if "call_transfer" in body:
        existing_settings = body["call_transfer"]
        # Update defaults with existing settings (existing values will override defaults)
        for key, value in existing_settings.items():
            transfer_settings[key] = value
    else:
        # No existing call_transfer - check if we have dialin settings for customer lookup
        if "dialin_settings" in body:
            # Create a temporary routing manager just for customer lookup
            call_config_manager = CallConfigManager(body)

            # Get caller info
            caller_info = call_config_manager.get_caller_info()
            from_number = caller_info.get("caller_number")

            if from_number:
                # Get customer name from phone number
                customer_name = call_config_manager.get_customer_name(from_number)

                # If we know the customer name, add it to the config for the bot to use
                if customer_name:
                    transfer_settings["customerName"] = customer_name

    return transfer_settings


def create_simple_dialin_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create simple dialin settings based on configuration.

    Args:
        body: The configuration dictionary

    Returns:
        Simple dialin settings dictionary
    """
    # Default simple dialin settings
    simple_dialin_settings = {
        "testInPrebuilt": DEFAULT_TEST_IN_PREBUILT,
    }

    # If simple_dialin already exists, merge the defaults with the existing settings
    if "simple_dialin" in body:
        existing_settings = body["simple_dialin"]
        # Update defaults with existing settings (existing values will override defaults)
        for key, value in existing_settings.items():
            simple_dialin_settings[key] = value

    return simple_dialin_settings


def create_simple_dialout_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create simple dialout settings based on configuration.

    Args:
        body: The configuration dictionary

    Returns:
        Simple dialout settings dictionary
    """
    # Default simple dialout settings
    simple_dialout_settings = {
        "testInPrebuilt": DEFAULT_TEST_IN_PREBUILT,
    }

    # If simple_dialout already exists, merge the defaults with the existing settings
    if "simple_dialout" in body:
        existing_settings = body["simple_dialout"]
        # Update defaults with existing settings (existing values will override defaults)
        for key, value in existing_settings.items():
            simple_dialout_settings[key] = value

    return simple_dialout_settings


async def process_dialin_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming dial-in request data to create a properly formatted body.

    Converts camelCase fields received from webhook to snake_case format
    for internal consistency across the codebase.

    Args:
        data: Raw dialin data from webhook

    Returns:
        Properly formatted configuration with snake_case keys
    """
    # Create base body with dialin settings
    body = {
        "dialin_settings": {
            "to": data.get("To", ""),
            "from": data.get("From", ""),
            "call_id": data.get("callId", data.get("CallSid", "")),  # Convert to snake_case
            "call_domain": data.get("callDomain", ""),  # Convert to snake_case
        }
    }

    # Use the global default to determine which example to run for dialin webhooks
    example = DEFAULT_DIALIN_EXAMPLE

    # Configure the bot based on the example
    if example == "call_transfer":
        # Create call transfer settings
        body["call_transfer"] = create_call_transfer_settings(body)
    elif example == "simple_dialin":
        # Create simple dialin settings
        body["simple_dialin"] = create_simple_dialin_settings(body)

    return body

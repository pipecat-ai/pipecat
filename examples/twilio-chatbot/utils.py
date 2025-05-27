"""This module contains the utils for the twilio chatbot."""

from typing import Dict, List


def get_unallowed_variable_names() -> list:
    """Get the unallowed variable names."""
    return ["call_id", "user_number", "agent_number"]

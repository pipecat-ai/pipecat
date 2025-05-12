# bot_definitions.py
"""Definitions of different bot types for the bot registry."""

from typing import Any, Dict

from bot_constants import DEFAULT_DIALIN_EXAMPLE
from bot_registry import BotRegistry, BotType
from bot_runner_helpers import (
    create_call_transfer_settings,
    create_simple_dialout_settings,
)

# Create and configure the bot registry
bot_registry = BotRegistry()

# Register bot types
bot_registry.register(
    BotType(
        name="call_transfer",
        settings_creator=create_call_transfer_settings,
        required_settings=["dialin_settings"],
        incompatible_with=["simple_dialin", "simple_dialout", "voicemail_detection"],
        auto_add_settings={"dialin_settings": {}},
    )
)

# Define the enhanced dial-in bot type
def create_enhanced_dialin_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create settings for the enhanced dial-in bot."""
    settings = body.get("enhanced_dialin", {})
    settings.setdefault("testInPrebuilt", False)
    return settings

enhanced_dialin_bot = BotType(
    name="enhanced_dialin",
    settings_creator=create_enhanced_dialin_settings,
    required_settings=["dialin_settings"],
    incompatible_with=["simple_dialin", "voicemail_detection", "call_transfer"],
    auto_add_settings={
        "llm": "openai",
        "tts": "cartesia",
    },
)

# Define the simple dial-in bot type
def create_simple_dialin_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create settings for the simple dial-in bot."""
    settings = body.get("simple_dialin", {})
    settings.setdefault("testInPrebuilt", False)
    return settings

simple_dialin_bot = BotType(
    name="simple_dialin",
    settings_creator=create_simple_dialin_settings,
    required_settings=["dialin_settings"],
    incompatible_with=["enhanced_dialin", "voicemail_detection", "call_transfer"],
    auto_add_settings={
        "llm": "openai",
        "tts": "cartesia",
    },
)

bot_registry.register(
    BotType(
        name="simple_dialout",
        settings_creator=create_simple_dialout_settings,
        required_settings=["dialout_settings"],
        incompatible_with=["call_transfer", "simple_dialin", "voicemail_detection"],
        auto_add_settings={"dialout_settings": [{}]},
    )
)

bot_registry.register(
    BotType(
        name="voicemail_detection",
        settings_creator=lambda body: body.get(
            "voicemail_detection", {}
        ),  # No creator function in original code
        required_settings=["dialout_settings"],
        incompatible_with=["call_transfer", "simple_dialin", "simple_dialout"],
        auto_add_settings={"dialout_settings": [{}]},
    )
)

# Register the bot types
bot_registry.register(enhanced_dialin_bot)
bot_registry.register(simple_dialin_bot)

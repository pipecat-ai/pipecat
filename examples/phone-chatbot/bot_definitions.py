# bot_definitions.py
"""Definitions of different bot types for the bot registry."""

from bot_registry import BotRegistry, BotType
from bot_runner_helpers import (
    create_call_transfer_settings,
    create_simple_dialin_settings,
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

bot_registry.register(
    BotType(
        name="simple_dialin",
        settings_creator=create_simple_dialin_settings,
        required_settings=["dialin_settings"],
        incompatible_with=["call_transfer", "simple_dialout", "voicemail_detection"],
        auto_add_settings={"dialin_settings": {}},
    )
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

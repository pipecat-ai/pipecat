# bot_registry.py
"""Bot registry pattern for managing different bot types."""

from typing import Any, Callable, Dict, List, Optional

from bot_constants import DEFAULT_DIALIN_EXAMPLE
from bot_runner_helpers import ensure_dialout_settings_array
from fastapi import HTTPException


class BotType:
    """Bot type configuration and handling."""

    def __init__(
        self,
        name: str,
        settings_creator: Callable[[Dict[str, Any]], Dict[str, Any]],
        required_settings: list = None,
        incompatible_with: list = None,
        auto_add_settings: dict = None,
    ):
        """Initialize a bot type.

        Args:
            name: Name of the bot type
            settings_creator: Function to create/update settings for this bot type
            required_settings: List of settings this bot type requires
            incompatible_with: List of bot types this one cannot be used with
            auto_add_settings: Settings to add if this bot is being run in test mode
        """
        self.name = name
        self.settings_creator = settings_creator
        self.required_settings = required_settings or []
        self.incompatible_with = incompatible_with or []
        self.auto_add_settings = auto_add_settings or {}

    def has_test_mode(self, body: Dict[str, Any]) -> bool:
        """Check if this bot type is configured for test mode."""
        return self.name in body and body[self.name].get("testInPrebuilt", False)

    def create_settings(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update settings for this bot type."""
        body[self.name] = self.settings_creator(body)
        return body

    def prepare_for_test(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Add required settings for test mode if they don't exist."""
        for setting, default_value in self.auto_add_settings.items():
            if setting not in body:
                body[setting] = default_value
        return body


class BotRegistry:
    """Registry for managing different bot types."""

    def __init__(self):
        self.bots = {}
        self.bot_validation_rules = []

    def register(self, bot_type: BotType):
        """Register a bot type."""
        self.bots[bot_type.name] = bot_type
        return self

    def get_bot(self, name: str) -> BotType:
        """Get a bot type by name."""
        return self.bots.get(name)

    def detect_bot_type(self, body: Dict[str, Any]) -> Optional[str]:
        """Detect which bot type to use based on configuration."""
        # First check for test mode bots
        for name, bot in self.bots.items():
            if bot.has_test_mode(body):
                return name

        # Then check for specific combinations of settings
        for name, bot in self.bots.items():
            if name in body and all(req in body for req in bot.required_settings):
                return name

        # Default for dialin settings
        if "dialin_settings" in body:
            return DEFAULT_DIALIN_EXAMPLE

        return None

    def validate_bot_combination(self, body: Dict[str, Any]) -> List[str]:
        """Validate that bot types in the configuration are compatible."""
        errors = []
        bot_types_in_config = [name for name in self.bots.keys() if name in body]

        # Check each bot type against its incompatible list
        for bot_name in bot_types_in_config:
            bot = self.bots[bot_name]
            for incompatible in bot.incompatible_with:
                if incompatible in body:
                    errors.append(
                        f"Cannot have both '{bot_name}' and '{incompatible}' in the same configuration"
                    )

        return errors

    def setup_configuration(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Set up bot configuration based on detected bot type."""
        # Ensure dialout_settings is an array if present
        body = ensure_dialout_settings_array(body)

        # Detect which bot type to use
        bot_type_name = self.detect_bot_type(body)
        if not bot_type_name:
            raise HTTPException(
                status_code=400, detail="Configuration doesn't match any supported scenario"
            )

        # If we have a dialin scenario but no explicit bot type, add the default
        if "dialin_settings" in body and bot_type_name == DEFAULT_DIALIN_EXAMPLE:
            if bot_type_name not in body:
                body[bot_type_name] = {}

        # Get the bot type object
        bot_type = self.get_bot(bot_type_name)

        # Create/update settings for the bot type
        body = bot_type.create_settings(body)

        # If in test mode, add any required settings
        if bot_type.has_test_mode(body):
            body = bot_type.prepare_for_test(body)

        # Validate bot combinations
        errors = self.validate_bot_combination(body)
        if errors:
            error_message = "Invalid configuration: " + "; ".join(errors)
            raise HTTPException(status_code=400, detail=error_message)

        return body

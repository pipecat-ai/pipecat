#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Function schema utilities for AI tool definitions.

This module provides standardized function schema representation for defining
tools and functions used with AI models, ensuring consistent formatting
across different AI service providers.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallHandler


class FunctionSchema:
    """Standardized function schema representation for tool definition.

    Provides a structured way to define function tools used with AI models like OpenAI.
    This schema defines the function's name, description, parameter properties, and
    required parameters, following specifications required by AI service providers.

    A schema may also carry the ``handler`` that runs when the function is called.
    When set, the LLM service registers it automatically for any `LLMContext` that
    advertises the schema, so no separate ``register_function`` call is needed.
    """

    def __init__(
        self,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
        handler: "FunctionCallHandler | None" = None,
    ) -> None:
        """Initialize the function schema.

        Args:
            name: Name of the function to be called.
            description: Description of what the function does.
            properties: Dictionary defining parameter types, descriptions, and constraints.
            required: List of property names that are required parameters.
            handler: Optional handler for this function. When provided, the LLM
                service registers it automatically wherever the schema is
                advertised in the `LLMContext`, making a separate
                ``register_function`` call unnecessary. Decorate the handler with
                ``@tool_options`` to override its default call options
                (``cancel_on_interruption``, ``timeout_secs``).
        """
        self._name = name
        self._description = description
        self._properties = properties
        self._required = required
        self._handler = handler

    def to_default_dict(self) -> dict[str, Any]:
        """Converts the function schema to a dictionary.

        Returns:
            Dictionary representation of the function schema.
        """
        return {
            "name": self._name,
            "description": self._description,
            "parameters": {
                "type": "object",
                "properties": self._properties,
                "required": self._required,
            },
        }

    @property
    def name(self) -> str:
        """Get the function name.

        Returns:
            The function name.
        """
        return self._name

    @property
    def description(self) -> str:
        """Get the function description.

        Returns:
            The function description.
        """
        return self._description

    @property
    def properties(self) -> dict[str, Any]:
        """Get the function properties.

        Returns:
            Dictionary of parameter specifications.
        """
        return self._properties

    @property
    def required(self) -> list[str]:
        """Get the required parameters.

        Returns:
            List of required parameter names.
        """
        return self._required

    @property
    def handler(self) -> "FunctionCallHandler | None":
        """Get the handler for this function, if any.

        Returns:
            The handler for this function, or ``None`` if it's provided
            separately, through ``register_function``.
        """
        return self._handler

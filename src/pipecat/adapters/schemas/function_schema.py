#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List


class FunctionSchema:
    """Standardized function schema representation for tool definition.

    Provides a structured way to define function tools used with AI models like OpenAI.
    This schema defines the function's name, description, parameter properties, and
    required parameters, following specifications required by AI service providers.

    Args:
        name: Name of the function to be called.
        description: Description of what the function does.
        properties: Dictionary defining parameter types, descriptions, and constraints.
        required: List of property names that are required parameters.
    """

    def __init__(
        self, name: str, description: str, properties: Dict[str, Any], required: List[str]
    ) -> None:
        self._name = name
        self._description = description
        self._properties = properties
        self._required = required

    def to_default_dict(self) -> Dict[str, Any]:
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
    def properties(self) -> Dict[str, Any]:
        """Get the function properties.

        Returns:
            Dictionary of parameter specifications.
        """
        return self._properties

    @property
    def required(self) -> List[str]:
        """Get the required parameters.

        Returns:
            List of required parameter names.
        """
        return self._required

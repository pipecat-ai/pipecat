#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List


class FunctionSchema:
    def __init__(
        self, name: str, description: str, properties: Dict[str, Any], required: List[str]
    ) -> None:
        """Standardized function schema representation.

        :param name: Name of the function.
        :param description: Description of the function.
        :param properties: Dictionary defining properties types and descriptions.
        :param required: List of required parameters.
        """
        self._name = name
        self._description = description
        self._properties = properties
        self._required = required

    def to_default_dict(self) -> Dict[str, Any]:
        """Converts the function schema to a dictionary.

        :return: Dictionary representation of the function schema.
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
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    @property
    def required(self) -> List[str]:
        return self._required

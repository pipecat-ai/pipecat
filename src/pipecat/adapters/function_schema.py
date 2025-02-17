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
        self.name = name
        self.description = description
        self.properties = properties
        self.required = required

    def to_default_dict(self) -> Dict[str, Any]:
        """Converts the function schema to a dictionary.

        :return: Dictionary representation of the function schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.properties,
                "required": self.required,
            },
        }

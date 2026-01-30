#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools schema definitions for function calling adapters.

This module provides schemas for managing both standardized function tools
and custom adapter-specific tools in the Pipecat framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pipecat.adapters.schemas.direct_function import DirectFunction, DirectFunctionWrapper
from pipecat.adapters.schemas.function_schema import FunctionSchema


@dataclass
class ToolsSchemaDiff:
    """Represents the differences between two ToolsSchema instances.

    Parameters:
        standard_tools_added: Names of newly added standard tools.
        standard_tools_removed: Names of removed standard tools.
        standard_tools_modified: True if any existing standard tool's definition changed.
        custom_tools_changed: True if the custom_tools dictionary differs.
    """

    standard_tools_added: List[str] = field(default_factory=list)
    standard_tools_removed: List[str] = field(default_factory=list)
    standard_tools_modified: bool = False
    custom_tools_changed: bool = False

    def has_changes(self) -> bool:
        """Check if there are any differences.

        Returns:
            True if any field indicates a change, False otherwise.
        """
        return bool(
            self.standard_tools_added
            or self.standard_tools_removed
            or self.standard_tools_modified
            or self.custom_tools_changed
        )

    def change_description(self) -> str:
        """Generate a human-readable description of the differences.

        Returns:
            A string summarizing the changes.
        """
        changes = []
        if self.standard_tools_added:
            changes.append(f"Added standard tools: {', '.join(self.standard_tools_added)}")
        if self.standard_tools_removed:
            changes.append(f"Removed standard tools: {', '.join(self.standard_tools_removed)}")
        if self.standard_tools_modified:
            changes.append("Modified definitions of existing standard tools")
        if self.custom_tools_changed:
            changes.append("Custom tools changed")
        return "; ".join(changes) if changes else "No changes"


class AdapterType(Enum):
    """Supported adapter types for custom tools.

    Parameters:
        GEMINI: Google Gemini adapter - currently the only service supporting custom tools.
        SHIM: Backward compatibility shim for creating ToolsSchemas from lists of tools in
              any format, used by LLMContext.from_openai_context.
    """

    GEMINI = "gemini"  # that is the only service where we are able to add custom tools for now
    SHIM = "shim"  # for use as backward compatibility shim for creating ToolsSchemas from list of tools in any format


class ToolsSchema:
    """Schema for managing both standard and custom function calling tools.

    This class provides a unified interface for handling standardized function
    schemas alongside custom tools that may not follow the standard format,
    such as adapter-specific search tools.
    """

    def __init__(
        self,
        standard_tools: List[FunctionSchema | DirectFunction],
        custom_tools: Optional[Dict[AdapterType, List[Dict[str, Any]]]] = None,
    ) -> None:
        """Initialize the tools schema.

        Args:
            standard_tools: List of tools following the standardized FunctionSchema format.
            custom_tools: Dictionary mapping adapter types to their custom tool definitions.
                These tools may not follow the FunctionSchema format (e.g., search_tool).
        """

        def _map_standard_tools(tools):
            schemas = []
            for tool in tools:
                if isinstance(tool, FunctionSchema):
                    schemas.append(tool)
                elif callable(tool):
                    wrapper = DirectFunctionWrapper(tool)
                    schemas.append(wrapper.to_function_schema())
                else:
                    raise TypeError(f"Unsupported tool type: {type(tool)}")
            return schemas

        self._standard_tools = _map_standard_tools(standard_tools)
        self._custom_tools = custom_tools

    @property
    def standard_tools(self) -> List[FunctionSchema]:
        """Get the list of standard function schema tools.

        Returns:
            List of tools following the FunctionSchema format.
        """
        return self._standard_tools

    @property
    def custom_tools(self) -> Dict[AdapterType, List[Dict[str, Any]]]:
        """Get the custom tools dictionary.

        Returns:
            Dictionary mapping adapter types to their custom tool definitions.
        """
        return self._custom_tools

    @custom_tools.setter
    def custom_tools(self, value: Dict[AdapterType, List[Dict[str, Any]]]) -> None:
        """Set the custom tools dictionary.

        Args:
            value: Dictionary mapping adapter types to their custom tool definitions.
        """
        self._custom_tools = value

    def diff(self, other: "ToolsSchema") -> ToolsSchemaDiff:
        """Compare this ToolsSchema to another and return the differences.

        Args:
            other: The ToolsSchema to compare against (the "after" state).

        Returns:
            ToolsSchemaDiff containing the differences between self and other.
        """
        result = ToolsSchemaDiff()

        # Build maps of tool name -> FunctionSchema for comparison
        self_tools_by_name: Dict[str, FunctionSchema] = {
            tool.name: tool for tool in self._standard_tools
        }
        other_tools_by_name: Dict[str, FunctionSchema] = {
            tool.name: tool for tool in other._standard_tools
        }

        self_names = set(self_tools_by_name.keys())
        other_names = set(other_tools_by_name.keys())

        # Find added and removed tools
        result.standard_tools_added = sorted(other_names - self_names)
        result.standard_tools_removed = sorted(self_names - other_names)

        # Check for modified tools (same name, different definition)
        common_names = self_names & other_names
        for name in common_names:
            self_tool = self_tools_by_name[name]
            other_tool = other_tools_by_name[name]
            # Compare using to_default_dict() for full schema comparison
            if self_tool.to_default_dict() != other_tool.to_default_dict():
                result.standard_tools_modified = True
                break

        # Compare custom tools
        result.custom_tools_changed = self._custom_tools != other._custom_tools

        return result

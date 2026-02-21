#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema, ToolsSchemaDiff


class TestToolsSchemaDiff(unittest.TestCase):
    """Tests for the ToolsSchemaDiff dataclass."""

    def test_has_changes_empty(self):
        """Test has_changes returns False for empty diff."""
        diff = ToolsSchemaDiff()
        self.assertFalse(diff.has_changes())

    def test_has_changes_with_added(self):
        """Test has_changes returns True when tools are added."""
        diff = ToolsSchemaDiff(standard_tools_added=["tool1"])
        self.assertTrue(diff.has_changes())

    def test_has_changes_with_removed(self):
        """Test has_changes returns True when tools are removed."""
        diff = ToolsSchemaDiff(standard_tools_removed=["tool1"])
        self.assertTrue(diff.has_changes())

    def test_has_changes_with_modified(self):
        """Test has_changes returns True when tools are modified."""
        diff = ToolsSchemaDiff(standard_tools_modified=True)
        self.assertTrue(diff.has_changes())

    def test_has_changes_with_custom_changed(self):
        """Test has_changes returns True when custom tools changed."""
        diff = ToolsSchemaDiff(custom_tools_changed=True)
        self.assertTrue(diff.has_changes())


class TestToolsSchemaDiffMethod(unittest.TestCase):
    """Tests for the ToolsSchema.diff() method."""

    def _create_function_schema(
        self, name: str, description: str = "Test function", properties: dict = None
    ) -> FunctionSchema:
        """Helper to create a FunctionSchema."""
        return FunctionSchema(
            name=name,
            description=description,
            properties=properties or {},
            required=[],
        )

    def test_diff_identical_schemas(self):
        """Test diff of two identical schemas returns no changes."""
        tool1 = self._create_function_schema("get_weather")
        schema1 = ToolsSchema(standard_tools=[tool1])
        schema2 = ToolsSchema(standard_tools=[self._create_function_schema("get_weather")])

        diff = schema1.diff(schema2)
        self.assertFalse(diff.has_changes())
        self.assertEqual(diff.standard_tools_added, [])
        self.assertEqual(diff.standard_tools_removed, [])
        self.assertFalse(diff.standard_tools_modified)
        self.assertFalse(diff.custom_tools_changed)

    def test_diff_tool_added(self):
        """Test diff detects added tools."""
        tool1 = self._create_function_schema("get_weather")
        tool2 = self._create_function_schema("get_time")

        schema1 = ToolsSchema(standard_tools=[tool1])
        schema2 = ToolsSchema(standard_tools=[tool1, tool2])

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.standard_tools_added, ["get_time"])
        self.assertEqual(diff.standard_tools_removed, [])
        self.assertFalse(diff.standard_tools_modified)

    def test_diff_tool_removed(self):
        """Test diff detects removed tools."""
        tool1 = self._create_function_schema("get_weather")
        tool2 = self._create_function_schema("get_time")

        schema1 = ToolsSchema(standard_tools=[tool1, tool2])
        schema2 = ToolsSchema(standard_tools=[tool1])

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.standard_tools_added, [])
        self.assertEqual(diff.standard_tools_removed, ["get_time"])
        self.assertFalse(diff.standard_tools_modified)

    def test_diff_tool_modified(self):
        """Test diff detects modified tools (same name, different definition)."""
        tool1_v1 = self._create_function_schema(
            "get_weather", description="Get weather v1", properties={"location": {"type": "string"}}
        )
        tool1_v2 = self._create_function_schema(
            "get_weather",
            description="Get weather v2",
            properties={"city": {"type": "string"}},
        )

        schema1 = ToolsSchema(standard_tools=[tool1_v1])
        schema2 = ToolsSchema(standard_tools=[tool1_v2])

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.standard_tools_added, [])
        self.assertEqual(diff.standard_tools_removed, [])
        self.assertTrue(diff.standard_tools_modified)

    def test_diff_multiple_changes(self):
        """Test diff with multiple types of changes."""
        tool_keep = self._create_function_schema("keep_tool")
        tool_remove = self._create_function_schema("remove_tool")
        tool_add = self._create_function_schema("add_tool")

        schema1 = ToolsSchema(standard_tools=[tool_keep, tool_remove])
        schema2 = ToolsSchema(standard_tools=[tool_keep, tool_add])

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.standard_tools_added, ["add_tool"])
        self.assertEqual(diff.standard_tools_removed, ["remove_tool"])

    def test_diff_empty_schemas(self):
        """Test diff of two empty schemas returns no changes."""
        schema1 = ToolsSchema(standard_tools=[])
        schema2 = ToolsSchema(standard_tools=[])

        diff = schema1.diff(schema2)
        self.assertFalse(diff.has_changes())

    def test_diff_custom_tools_changed(self):
        """Test diff detects custom tools changes."""
        tool1 = self._create_function_schema("get_weather")
        custom1 = {AdapterType.GEMINI: [{"name": "search"}]}
        custom2 = {AdapterType.GEMINI: [{"name": "search_v2"}]}

        schema1 = ToolsSchema(standard_tools=[tool1], custom_tools=custom1)
        schema2 = ToolsSchema(standard_tools=[tool1], custom_tools=custom2)

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.custom_tools_changed)
        # Standard tools unchanged
        self.assertEqual(diff.standard_tools_added, [])
        self.assertEqual(diff.standard_tools_removed, [])
        self.assertFalse(diff.standard_tools_modified)

    def test_diff_custom_tools_added(self):
        """Test diff detects custom tools being added."""
        tool1 = self._create_function_schema("get_weather")

        schema1 = ToolsSchema(standard_tools=[tool1])
        schema2 = ToolsSchema(
            standard_tools=[tool1], custom_tools={AdapterType.GEMINI: [{"name": "search"}]}
        )

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.custom_tools_changed)

    def test_diff_custom_tools_removed(self):
        """Test diff detects custom tools being removed."""
        tool1 = self._create_function_schema("get_weather")

        schema1 = ToolsSchema(
            standard_tools=[tool1], custom_tools={AdapterType.GEMINI: [{"name": "search"}]}
        )
        schema2 = ToolsSchema(standard_tools=[tool1])

        diff = schema1.diff(schema2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.custom_tools_changed)


if __name__ == "__main__":
    unittest.main()

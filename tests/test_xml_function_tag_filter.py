#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for XMLFunctionTagFilter."""

import unittest

from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter


class TestXMLFunctionTagFilter(unittest.IsolatedAsyncioTestCase):
    """Test cases for the XMLFunctionTagFilter."""

    async def test_basic_function_tag_removal(self):
        """Test removal of basic function tags without content."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = "Hello there! <function=always_move_to_main_agenda></function> How are you?"
        expected_text = "Hello there! How are you?"
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_function_tag_with_json_content(self):
        """Test removal of function tags containing JSON parameters."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = ('I can help you with that. '
                        '<function=schedule_interview>{"date": "tomorrow", "time": "3 PM", "location": "our office", "candidate_name": "Sabiha"}</function> '
                        'Let me schedule that interview for you.')
        expected_text = "I can help you with that. Let me schedule that interview for you."
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_function_tag_with_parameters(self):
        """Test removal of function tags with parameters in the tag itself."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = ('The calculation result is: '
                        '<function=safe_calculator({"expression": "two thousand + five hundred"}></function> '
                        'which equals 2500.')
        expected_text = "The calculation result is: which equals 2500."
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_multiple_function_tags(self):
        """Test removal of multiple function tags in a single text."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = ('Starting the call <function=always_move_to_main_agenda></function> '
                        'with agenda items. <function=end_call></function> Call ended. '
                        '<function=move_to_summary></function> Here is the summary.')
        expected_text = "Starting the call with agenda items. Call ended. Here is the summary."
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_nested_and_complex_function_tags(self):
        """Test handling of complex function tags with various formats."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = ('Complex example: <function=complex_function>{"nested": {"data": "value"}}</function> '
                        'and <function=simple_func/> self-closing, '
                        'plus <function=empty_func></function> empty tag.')
        expected_text = "Complex example: and self-closing, plus empty tag."
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_text_without_function_tags(self):
        """Test that text without function tags passes through unchanged."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = "This is just normal text without any function calls."
        
        result = await xml_filter.filter(original_text)
        assert result == original_text

    async def test_custom_patterns(self):
        """Test filter with custom regex patterns."""
        custom_patterns = [r'\[action:.*?\]']  # Custom pattern for different format
        xml_filter = XMLFunctionTagFilter(custom_patterns=custom_patterns)
        
        original_text = ("Standard function: <function=test></function> "
                        "and custom action: [action:do_something] here.")
        expected_text = "Standard function: and custom action: here."
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_preserve_whitespace_false(self):
        """Test filter with preserve_whitespace=False."""
        xml_filter = XMLFunctionTagFilter(preserve_whitespace=False)
        
        original_text = "Before<function=test></function>After"
        expected_text = "BeforeAfter"  # No space inserted
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_update_settings(self):
        """Test updating filter settings."""
        xml_filter = XMLFunctionTagFilter()
        
        # Test updating custom patterns
        await xml_filter.update_settings({
            "custom_patterns": [r'\[custom:.*?\]'],
            "preserve_whitespace": False
        })
        
        original_text = "Text<function=test></function>and[custom:action]here."
        expected_text = "Textandhere."  # Both patterns removed, no whitespace preservation
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_case_insensitive_matching(self):
        """Test that function tag matching is case insensitive."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = "Mixed case: <Function=Test></Function> and <FUNCTION=UPPER></FUNCTION>"
        expected_text = "Mixed case: and"
        
        result = await xml_filter.filter(original_text)
        assert result == expected_text

    async def test_multiline_function_tags(self):
        """Test filtering of function tags that span multiple lines."""
        xml_filter = XMLFunctionTagFilter()
        
        original_text = """First line.
        <function=multiline_test>
        {"param": "value",
         "other": "data"}
        </function>
        Last line."""
        expected_text = "First line. Last line."
        
        result = await xml_filter.filter(original_text)
        assert result.strip() == expected_text

    async def test_empty_and_whitespace_handling(self):
        """Test handling of edge cases like empty text and whitespace."""
        xml_filter = XMLFunctionTagFilter()
        
        test_cases = [
            ("", ""),  # Empty text
            ("   ", ""),  # Only whitespace
            ("<function=test></function>", ""),  # Only function call tag
            ("  <function=test></function>  ", ""),  # Only function call with whitespace
        ]
        
        for input_text, expected in test_cases:
            result = await xml_filter.filter(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"


if __name__ == "__main__":
    unittest.main()
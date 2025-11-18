#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import unittest

from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter


class TestMarkdownTextFilter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.filter = MarkdownTextFilter()

    async def test_basic_markdown_removal(self):
        """Test removal of basic Markdown formatting while preserving content."""
        input_text = """
            **Bold text** and *italic text*
            1. Numbered list item
            - Bullet point
            Some `inline code` here
        """

        expected_text = """
            Bold text and italic text
            1. Numbered list item
            - Bullet point
            Some inline code here
        """

        result = await self.filter.filter(input_text)
        self.assertEqual(result.strip(), expected_text.strip())

    async def test_space_preservation(self):
        """Test preservation of leading and trailing spaces (for
        word-by-word streaming in bot-tts-text).
        """
        input_text = [
            "  Leading spaces",
            "Trailing spaces  ",
            "  Both ends  ",
            "  Multiple  spaces  between  words  ",
        ]

        for text in input_text:
            result = await self.filter.filter(text)
            self.assertEqual(
                len(result), len(text), f"Space preservation failed for: '{text}'\nGot: '{result}'"
            )
            # Check if spaces are in the same positions
            for i, char in enumerate(text):
                if char == " ":
                    self.assertEqual(
                        result[i], " ", f"Space at position {i} was not preserved in: '{text}'"
                    )

    async def test_repeated_character_removal(self):
        """Test removal of repeated character sequences (5 or more)."""
        test_cases = {
            "Hello!!!!!World": "HelloWorld",  # 5 exclamations removed
            "Test####ing": "Test####ing",  # 4 hashes preserved
            "Normal text": "Normal text",  # No repeated chars
            "!!!!!": "",  # All repeated chars removed
            "Mixed!!!!!...../////": "Mixed",  # Multiple repeated sequences
            "Text^^^^test": "Text^^^^test",  # 4 carets preserved
            "Text^^^^^test": "Texttest",  # 5 carets removed
            "Dots....here": "Dots....here",  # 4 dots preserved
            "Dots.....here": "Dotshere",  # 5 dots removed
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(
                result,
                expected,
                f"Failed to handle repeated characters in: '{input_text}'\nExpected: '{expected}'\nGot: '{result}'",
            )

    async def test_numbered_list_preservation(self):
        """Test that numbered lists are preserved correctly."""
        input_text = """1. First item
        2. Second item
        3. Third item with **bold**"""

        expected = """1. First item
        2. Second item
        3. Third item with bold"""

        result = await self.filter.filter(input_text)
        self.assertEqual(
            result.strip(),
            expected.strip(),
            f"Numbered list preservation failed.\nExpected:\n{expected}\nGot:\n{result}",
        )

    async def test_html_entity_conversion(self):
        """Test conversion of HTML entities to their plain text equivalents."""
        test_cases = {
            "This &amp; that": "This & that",
            "1 &lt; 2": "1 < 2",
            "2 &gt; 1": "2 > 1",
            "Line&nbsp;break": "Line break",
            "Mixed &amp; &lt;entities&gt;": "Mixed & <entities>",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"HTML entity conversion failed for: '{input_text}'")

    async def test_asterisk_removal(self):
        """Test removal of Markdown asterisk formatting."""
        test_cases = {
            "**bold text**": "bold text",  # Double asterisks
            "*italic text*": "italic text",  # Single asterisks
            "**bold** and *italic*": "bold and italic",  # Mixed
            "multiple**bold**words": "multipleboldwords",  # No spaces
            "edge**cases***here*": "edgecaseshere",  # Adjacent asterisks
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Asterisk removal failed for: '{input_text}'")

    async def test_newline_handling(self):
        """Test handling of empty and whitespace-only lines."""
        test_cases = {
            "Line 1\n\nLine 2": "Line 1\n Line 2",  # Empty line becomes space
            "Line 1\n   \nLine 2": "Line 1\n Line 2",  # Whitespace line becomes single space
            "Text\n\n\nMore": "Text\n More",  # Multiple empty lines become spaces
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(
                result, expected, f"Newline handling failed for:\n{input_text}\nGot:\n{result}"
            )

    async def test_links_cleaning(self):
        """Test cleaning of links and URLs, i.e. https?:// is removed."""
        test_cases = {
            "Please check http://example.com": "Please check example.com",
            "Visit https://www.google.com for more": "Visit www.google.com for more",
            "No link here": "No link here",  # No link to clean
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Link cleaning failed for: '{input_text}'")

    async def test_numbered_list_marker_handling(self):
        """Test handling of numbered lists with the special §NUM§ marker."""
        test_cases = {
            "1. First\n2. Second": "1. First\n2. Second",  # Basic numbered list
            "  1. Indented": "  1. Indented",  # Indented numbered list
            "1. Item\nText\n2. Item": "1. Item\nText\n2. Item",  # Text between items
            "1.No space": "1.No space",  # Not a list item (no space)
            "12. Large number": "12. Large number",  # Multi-digit numbers
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(
                result,
                expected,
                f"Numbered list handling failed for:\n{input_text}\nGot:\n{result}",
            )

    async def test_inline_code_handling(self):
        """Test handling of inline code with backticks."""
        test_cases = {
            "`code`": "code",  # Basic inline code
            "Text `code` more": "Text code more",  # Inline code within text
            "``nested`code``": "nested`code",  # Nested backticks
            "`code1` and `code2`": "code1 and code2",  # Multiple inline codes
            "No``space``between": "Nospacebetween",  # No spaces around backticks
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Inline code handling failed for: '{input_text}'")

    async def test_simple_table_removal(self):
        """Test removal of a simple markdown table."""
        filter = MarkdownTextFilter(params=MarkdownTextFilter.InputParams(filter_tables=True))

        input_text = "| Column 1 | Column 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |"

        expected = ""

        result = await filter.filter(input_text)
        self.assertEqual(
            result.strip(),
            expected.strip(),
            f"Simple table removal failed.\nExpected:\n{expected}\nGot:\n{result}",
        )

    async def test_feature_toggles(self):
        """Test enabling/disabling specific filter features."""
        # Create a filter with all features disabled
        filter = MarkdownTextFilter(
            params=MarkdownTextFilter.InputParams(
                enable_text_filter=False,
                filter_code=False,
                filter_tables=False,
            )
        )

        # Test with text filtering disabled
        text_with_markdown = "**bold** and *italic* with `code`"
        self.assertEqual(
            await filter.filter(text_with_markdown),
            text_with_markdown,
            "Disabled filter should not modify text",
        )

        # Enable just text filtering
        await filter.update_settings({"enable_text_filter": True})
        self.assertEqual(
            await filter.filter(text_with_markdown),
            "bold and italic with code",
            "Enabled filter should remove markdown",
        )

    async def test_settings_update(self):
        """Test that filter settings can be updated at runtime."""
        filter = MarkdownTextFilter()

        # Initial state - formatting should be removed
        input_text = "**bold** and *italic*"
        self.assertEqual(await filter.filter(input_text), "bold and italic")

        # Disable text filtering
        await filter.update_settings({"enable_text_filter": False})
        self.assertEqual(
            await filter.filter(input_text), input_text, "Text filtering should be disabled"
        )

        # Re-enable text filtering
        await filter.update_settings({"enable_text_filter": True})
        self.assertEqual(
            await filter.filter(input_text),
            "bold and italic",
            "Text filtering should be re-enabled",
        )

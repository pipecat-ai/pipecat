#
# Copyright (c) 2024-2026, Daily
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
            Some `inline code` here\n\n## Subtitle
        """

        expected_text = """
            Bold text and italic text
            1. Numbered list item
            - Bullet point
            Some inline code here\nSubtitle
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

    async def test_preserve_repeated_characters(self):
        """Test preservation of repeated characters (5 or more)."""
        filter = MarkdownTextFilter(
            params=MarkdownTextFilter.InputParams(filter_repeated_sequences=False)
        )
        repeated_sequence = "55555"

        result = await filter.filter(repeated_sequence)

        self.assertEqual(
            result,
            repeated_sequence,
            f"Preserve repeated sequences failed.\nExpected:\n{repeated_sequence}\nGot:\n{result}",
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
            "Say &quot;hi&quot;": 'Say "hi"',
            "Copyright &copy; 2024": "Copyright © 2024",
            "Email <me@example.com> now": "Email me@example.com now",
            "Go to example.com/?a=1&copy=2": "Go to example.com/?a=1&copy=2",
            "AT&T": "AT&T",
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

    async def test_standalone_asterisk_preserved(self):
        """Test that an asterisk not attached to a word is kept."""
        test_cases = {
            "Press * then #": "Press * then #",
            "5 * 3 = 15": "5 * 3 = 15",
            "Use the * key": "Use the * key",
            "2*3*4 is 24": "2*3*4 is 24",
            "2**3 is 8": "2**3 is 8",
            "Type \\*literal\\* asterisks": "Type *literal* asterisks",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Standalone asterisk lost in: '{input_text}'")

    async def test_strikethrough_removal(self):
        """Test removal of struck-through text."""
        test_cases = {
            "The price was ~~ten~~ eight dollars.": "The price was eight dollars.",
            "It is ~~not ready~~ ready now.": "It is ready now.",
            "~~Old.~~ New.": " New.",
            "About ~~ 5 minutes": "About ~~ 5 minutes",
            "Roughly ~5 minutes": "Roughly ~5 minutes",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Strikethrough removal failed for: '{input_text}'")

    async def test_section_sign_preserved(self):
        """Test that a section sign in the text is kept."""
        test_cases = {
            "See § 5 of the contract.": "See § 5 of the contract.",
            "Sections §§ 3-4 apply.": "Sections §§ 3-4 apply.",
            " § 2": " § 2",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Section sign lost in: '{input_text}'")

    async def test_horizontal_rule_removal(self):
        """Test removal of horizontal rules, alone or between paragraphs."""
        test_cases = {
            "\n---\n": "",
            "\n***\n": "",
            "\n___\n": "",
            "Done.\n\n- - -\n\nNext.": "Done.\n Next.",
            "First part.\n\n---\n\nSecond part.": "First part.\n Second part.",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result.strip(), expected, f"Rule removal failed for: {input_text!r}")

    async def test_bullet_marker_removal(self):
        """Test removal of bullet markers, including streamed list items."""
        test_cases = {
            "- apples\n- pears": "apples\npears",
            "+ apples\n+ pears": "apples\npears",
            "\n- Coffee for three dollars.": "Coffee for three dollars.",
            "\n+ Red.": "Red.",
            "\n* Tea.": "Tea.",
            "Options:\n\n- Coffee.\n- Tea.": "Options:\n Coffee.\nTea.",
            "Plan:\n  - Sub step.": "Plan:\n  Sub step.",
            "- -5 degrees": "-5 degrees",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result.strip(), expected, f"Bullet removal failed for: {input_text!r}")

    async def test_blockquote_marker_removal(self):
        """Test removal of blockquote markers, including streamed quotes."""
        test_cases = {
            "> quote": "quote",
            "\n> As they say, less is more.": "As they say, less is more.",
            "Tip:\n\n> Save often.": "Tip:\n Save often.",
            ">> Nested quote.": "Nested quote.",
            "Use -> to point": "Use -> to point",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result.strip(), expected, f"Quote removal failed for: {input_text!r}")

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

    async def test_image_alt_text_kept(self):
        """Test that an image is replaced with its alt text."""
        test_cases = {
            "See ![a cat](https://example.com/cat.png) here.": "See a cat here.",
            "![Sales chart](chart.png)": "Sales chart",
            "No alt ![](x.png) here.": "No alt  here.",
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(result, expected, f"Image alt text lost in: '{input_text}'")

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

    async def test_table_removal_keeps_surrounding_text(self):
        """Test that removing a table keeps the text before and after it."""
        table = "| Item | Price |\n|------|-------|\n| Tea | 2 |\n| Coffee | 3 |"
        test_cases = {
            f"Here are the prices.\n\n{table}": "Here are the prices.",
            f"\n\n{table}\n\nLet me know what you want.": "Let me know what you want.",
            f"Before.\n{table}\nAfter.": "Before.\nAfter.",
        }

        for input_text, expected in test_cases.items():
            filter = MarkdownTextFilter(params=MarkdownTextFilter.InputParams(filter_tables=True))
            result = await filter.filter(input_text)
            self.assertEqual(result.strip(), expected, f"Table removal failed for: {input_text!r}")

    async def test_code_block_removal(self):
        """Test removal of a code block that opens and closes in one chunk."""
        test_cases = {
            "```python\nprint(1)\n```": "",
            "Here:\n```python\nprint(1)\n```\nDone.": "Here: Done.",
            "Sure! ```python\nprint(1)\n``` That prints 1.": "Sure! That prints 1.",
            "\n\n```python\nprint(1)\n```\n\nThat prints 1.": "That prints 1.",
        }

        for input_text, expected in test_cases.items():
            filter = MarkdownTextFilter(params=MarkdownTextFilter.InputParams(filter_code=True))
            result = await filter.filter(input_text)
            self.assertEqual(result, expected, f"Code block removal failed for: {input_text!r}")

    async def test_code_block_removal_across_chunks(self):
        """Test removal of a code block that spans several chunks."""
        filter = MarkdownTextFilter(params=MarkdownTextFilter.InputParams(filter_code=True))
        chunks = {
            "Here is code:": "Here is code:",
            "```python": "",
            "print(1)": "",
            "x = 1\n``` That sets x.": "That sets x.",
            "Anything else?": "Anything else?",
        }

        for input_text, expected in chunks.items():
            result = await filter.filter(input_text)
            self.assertEqual(result, expected, f"Code block removal failed for: {input_text!r}")

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

    async def test_header_stripping(self):
        """Test that markdown headers are stripped to plain text."""
        input_text = (
            "# Title\n"  # fmt: skip
            "\n"
            "## Subtitle\n"
        )
        expected = (
            "Title\n"  # fmt: skip
            "Subtitle"
        )

        result = await self.filter.filter(input_text)

        self.assertEqual(
            result.strip(),
            expected,
            f"Header stripping failed.\nInput:\n{input_text}\nExpected:\n{expected}\nGot:\n{result.strip()}",
        )

    async def test_closed_atx_header_stripping(self):
        """Test that closed ATX headers (## Title ##) strip both markers."""
        test_cases = {
            "## Subtitle ##": "Subtitle",  # Symmetric closing run
            "### Deep ###": "Deep",
            "## Done ###": "Done",  # Asymmetric closing run
            "# Title\n\n## Subtitle ##": "Title\nSubtitle",  # Closed header after a blank line
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(
                result.strip(),
                expected,
                f"Closed ATX header stripping failed for: {input_text!r}\nGot: {result!r}",
            )

    async def test_header_trailing_space_preserved(self):
        """Test that open headers keep trailing whitespace.

        Trailing spaces matter for word-by-word streaming in bot-tts-text, so
        only a closing marker's surrounding spaces are absorbed.
        """
        result = await self.filter.filter("## Trailing   ")
        self.assertEqual(
            result, "Trailing   ", f"Header trailing space not preserved.\nGot: {result!r}"
        )

    async def test_header_levels(self):
        """Test that all six ATX header levels are reduced to their text."""
        for n in range(1, 7):
            input_text = f"{'#' * n} Heading"
            result = await self.filter.filter(input_text)
            self.assertEqual(result.strip(), "Heading", f"h{n} header failed.\nGot: {result!r}")

    async def test_hash_in_content_not_treated_as_header(self):
        """Test that a '#' that isn't a leading ATX marker is left in the text."""
        test_cases = {
            "C# is great": "C# is great",  # No leading marker
            "## Issue #42 today": "Issue #42 today",  # '#' mid-content, not a closing run
            "## C# matters": "C# matters",  # Content starts with a hash word
        }

        for input_text, expected in test_cases.items():
            result = await self.filter.filter(input_text)
            self.assertEqual(
                result.strip(),
                expected,
                f"Hash-in-content handling failed for: {input_text!r}\nGot: {result!r}",
            )


if __name__ == "__main__":
    unittest.main()

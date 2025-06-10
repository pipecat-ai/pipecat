#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
from typing import Any, Mapping, Optional

from markdown import Markdown
from pydantic import BaseModel

from pipecat.utils.text.base_text_filter import BaseTextFilter


class MarkdownTextFilter(BaseTextFilter):
    """Removes Markdown formatting from text in TextFrames.

    Converts Markdown to plain text while preserving the overall structure,
    including leading and trailing spaces. Handles special cases like
    asterisks and table formatting.
    """

    class InputParams(BaseModel):
        enable_text_filter: Optional[bool] = True
        filter_code: Optional[bool] = False
        filter_tables: Optional[bool] = False

    def __init__(self, params: Optional[InputParams] = None, **kwargs):
        super().__init__(**kwargs)
        self._settings = params or MarkdownTextFilter.InputParams()
        self._in_code_block = False
        self._in_table = False
        self._interrupted = False

    async def update_settings(self, settings: Mapping[str, Any]):
        for key, value in settings.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)

    async def filter(self, text: str) -> str:
        if self._settings.enable_text_filter:
            # Remove newlines and replace with a space only when there's no text before or after
            filtered_text = re.sub(r"^\s*\n", " ", text, flags=re.MULTILINE)

            # Remove backticks from inline code, but not from code blocks
            filtered_text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", filtered_text)

            # Remove repeated sequences of 5 or more characters
            filtered_text = re.sub(r"(\S)(\1{4,})", "", filtered_text)

            # Preserve numbered list items with a unique marker, §NUM§
            filtered_text = re.sub(r"^(\d+\.)\s", r"§NUM§\1 ", filtered_text)

            # Preserve leading/trailing spaces with a unique marker, §
            # Critical for word-by-word streaming in bot-tts-text
            filtered_text = re.sub(
                r"^( +)|\s+$", lambda m: "§" * len(m.group(0)), filtered_text, flags=re.MULTILINE
            )

            # Remove space placeholders before tables, so that tables are converted to HTML
            # correctly
            filtered_text = re.sub(r"§\| ", "| ", filtered_text)

            # Convert markdown to HTML
            extension = ["tables"] if self._settings.filter_tables else []
            md = Markdown(extensions=extension)
            filtered_text = md.convert(filtered_text)

            # Remove tables
            if self._settings.filter_tables:
                filtered_text = self.remove_tables(filtered_text)

            # Remove HTML tags
            filtered_text = re.sub("<[^<]+?>", "", filtered_text)

            # Replace HTML entities
            filtered_text = filtered_text.replace("&nbsp;", " ")
            filtered_text = filtered_text.replace("&lt;", "<")
            filtered_text = filtered_text.replace("&gt;", ">")
            filtered_text = filtered_text.replace("&amp;", "&")

            # Remove double asterisks (consecutive without any exceptions)
            filtered_text = re.sub(r"\*\*", "", filtered_text)

            # Remove single asterisks at the start or end of words
            filtered_text = re.sub(r"(^|\s)\*|\*($|\s)", r"\1\2", filtered_text)

            # Remove Markdown table formatting
            filtered_text = re.sub(r"\|", "", filtered_text)
            filtered_text = re.sub(r"^\s*[-:]+\s*$", "", filtered_text, flags=re.MULTILINE)

            # Remove code blocks
            if self._settings.filter_code:
                filtered_text = self._remove_code_blocks(filtered_text)

            # Restore numbered list items
            filtered_text = filtered_text.replace("§NUM§", "")

            # Restore leading and trailing spaces
            filtered_text = re.sub("§", " ", filtered_text)

            ## Make links more readable
            filtered_text = re.sub(r"https?://", "", filtered_text)

            return filtered_text
        else:
            return text

    async def handle_interruption(self):
        self._interrupted = True
        self._in_code_block = False
        self._in_table = False

    async def reset_interruption(self):
        self._interrupted = False

    #
    # Filter code
    #

    def _remove_code_blocks(self, text: str) -> str:
        """Main method to remove code blocks from the input text.
        Handles interruptions and delegates to specific methods based on the current state.
        """
        if self._interrupted:
            self._in_code_block = False
            return text

        # Pattern to match three consecutive backticks (code block delimiter)
        code_block_pattern = r"```"
        match = re.search(code_block_pattern, text)

        if self._in_code_block:
            return self._handle_in_code_block(match, text)

        return self._handle_not_in_code_block(match, text, code_block_pattern)

    def _handle_in_code_block(self, match, text):
        """Handle text when we're currently inside a code block.
        If we find the end of the block, return text after it. Otherwise, skip the content.
        """
        if match:
            self._in_code_block = False
            end_index = match.end()
            return text[end_index:].strip()
        return ""  # Skip content inside code block

    def _handle_not_in_code_block(self, match, text, code_block_pattern):
        """Handle text when we're not currently inside a code block.
        Delegate to specific methods based on whether we find a code block delimiter.
        """
        if not match:
            return text  # No code block found, return original text

        start_index = match.start()
        if start_index == 0 or text[:start_index].isspace():
            return self._handle_start_of_code_block(text, start_index)
        return self._handle_code_block_within_text(text, code_block_pattern)

    def _handle_start_of_code_block(self, text, start_index):
        """Handle the case where we find the start of a code block.
        Return any text before the code block and set the state to inside a code block.
        """
        self._in_code_block = True
        return text[:start_index].strip()

    def _handle_code_block_within_text(self, text, code_block_pattern):
        """Handle the case where we find a code block within the text.
        If it's a complete code block, remove it and return surrounding text.
        If it's the start of a code block, return text before it and set state.
        """
        parts = re.split(code_block_pattern, text)
        if len(parts) > 2:
            return (parts[0] + " " + parts[-1]).strip()
        self._in_code_block = True
        return parts[0].strip()

    #
    # Filter tables
    #
    def remove_tables(self, text: str) -> str:
        """Remove tables from the input text, handling cases where
        both start and end tags are in the same input.
        """
        if self._interrupted:
            self._in_table = False
            return text

        # Pattern to match entire table or parts of it
        table_pattern = r"<table>.*?</table>"
        partial_table_start = r"<table>.*"
        partial_table_end = r".*</table>"

        # Remove complete tables
        text = re.sub(table_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Handle partial tables at the start
        if self._in_table:
            match = re.match(partial_table_end, text, re.DOTALL | re.IGNORECASE)
            if match:
                self._in_table = False
                return text[match.end() :].strip()
            else:
                return ""  # Still inside a table, remove all content

        # Handle partial tables at the end
        match = re.search(partial_table_start, text, re.DOTALL | re.IGNORECASE)
        if match:
            self._in_table = True
            return text[: match.start()].strip()

        return text.strip()

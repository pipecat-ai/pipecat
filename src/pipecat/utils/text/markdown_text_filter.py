#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
from typing import Any, Mapping

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
        enable_text_filter: bool = True
        filter_code: bool = False

    def __init__(self, params: InputParams = InputParams(), **kwargs):
        super().__init__(**kwargs)
        self._settings = params
        self._in_code_block = False
        self._interrupted = False

    def update_settings(self, settings: Mapping[str, Any]):
        for key, value in settings.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)

    def filter(self, text: str) -> str:
        if self._settings.enable_text_filter:
            # Replace newlines with spaces only when there's no text before or after
            text = re.sub(r"^\s*\n", " ", text, flags=re.MULTILINE)

            # Remove backticks from inline code, but not from code blocks
            text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", text)

            # Remove repeated sequences of 5 or more characters
            text = re.sub(r"(\S)(\1{4,})", "", text)

            # Preserve numbered list items with a unique marker, §NUM§
            text = re.sub(r"^(\d+\.)\s", r"§NUM§\1 ", text)

            # Preserve leading/trailing spaces with a unique marker, §
            # Critical for word-by-word streaming in bot-tts-text
            preserved_markdown = re.sub(
                r"^( +)|\s+$", lambda m: "§" * len(m.group(0)), text, flags=re.MULTILINE
            )

            # Convert markdown to HTML
            md = Markdown()
            html = md.convert(preserved_markdown)

            # Remove HTML tags
            filtered_text = re.sub("<[^<]+?>", "", html)

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
                filtered_text = self.remove_code_blocks(filtered_text)

            # Restore numbered list items
            filtered_text = filtered_text.replace("§NUM§", "")

            # Restore leading and trailing spaces
            filtered_text = re.sub("§", " ", filtered_text)

            return filtered_text
        else:
            return text

    def handle_interruption(self):
        self._interrupted = True
        self._in_code_block = False

    def reset_interruption(self):
        self._interrupted = False

    def remove_code_blocks(self, text: str) -> str:
        """
        Main method to remove code blocks from the input text.
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
        """
        Handle text when we're currently inside a code block.
        If we find the end of the block, return text after it. Otherwise, skip the content.
        """
        if match:
            self._in_code_block = False
            end_index = match.end()
            return text[end_index:].strip()
        return " "  # Skip content inside code block

    def _handle_not_in_code_block(self, match, text, code_block_pattern):
        """
        Handle text when we're not currently inside a code block.
        Delegate to specific methods based on whether we find a code block delimiter.
        """
        if not match:
            return text  # No code block found, return original text

        start_index = match.start()
        if start_index == 0 or text[:start_index].isspace():
            return self._handle_start_of_code_block(text, start_index)

        return self._handle_code_block_within_text(text, code_block_pattern)

    def _handle_start_of_code_block(self, text, start_index):
        """
        Handle the case where we find the start of a code block.
        Return any text before the code block and set the state to inside a code block.
        """
        self._in_code_block = True
        return text[:start_index].strip()

    def _handle_code_block_within_text(self, text, code_block_pattern):
        """
        Handle the case where we find a code block within the text.
        If it's a complete code block, remove it and return surrounding text.
        If it's the start of a code block, return text before it and set state.
        """
        parts = re.split(code_block_pattern, text)
        if len(parts) > 2:
            return (parts[0] + " " + parts[-1]).strip()
        self._in_code_block = True
        return parts[0].strip()

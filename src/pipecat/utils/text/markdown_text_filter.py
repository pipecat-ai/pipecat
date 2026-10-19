#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Markdown text filter for removing Markdown formatting from text.

This module provides a text filter that converts Markdown content to plain text
while preserving structure and handling special cases like code blocks and tables.
"""

import html
import re
from collections.abc import Mapping
from typing import Any

from markdown import Markdown
from pydantic import BaseModel

from pipecat.utils.text.base_text_filter import BaseTextFilter

# A Markdown table: a header row, a delimiter row (| --- | :-: |) and any body
# rows, each line starting with a pipe.
_MARKDOWN_TABLE = re.compile(
    r"^[ \t]*\|.*\n"
    r"[ \t]*\|[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*(?:\n|$)"
    r"(?:[ \t]*\|.*(?:\n|$))*",
    re.MULTILINE,
)

# A code fence: ``` at the start of a line, as in CommonMark. A ``` inside a
# sentence ("type ``` to start a code block") is text.
_CODE_FENCE = re.compile(r"^[ \t]*```", re.MULTILINE)

# An HTML tag whose attributes either have values or are boolean HTML
# attributes (<details open>). Only tags named in _HTML_TAGS are stripped, so
# text that md.convert passes through as raw HTML ("x<y and y>z",
# "a<b and c>d") is kept.
_HTML_TAG = re.compile(
    r"</?([A-Za-z][A-Za-z0-9]*)"
    r"(?:\s+(?:[A-Za-z_:][-\w:.]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+)"
    r"|(?i:allowfullscreen|async|autofocus|autoplay|checked|controls|default|defer"
    r"|disabled|hidden|inert|loop|multiple|muted|nomodule|novalidate|open|readonly"
    r"|required|reversed|selected)(?=[\s/>])))*"
    r"\s*/?>"
)

# The tags md.convert writes, plus common ones in an LLM's raw HTML.
_HTML_TAGS = frozenset(
    "a abbr b blockquote br caption center cite code col colgroup dd del details div dl dt "
    "em figcaption figure font h1 h2 h3 h4 h5 h6 hr i img ins kbd li mark ol p pre q s "
    "small span strike strong sub summary sup table tbody td tfoot th thead tr u ul".split()
)

# Placeholders that hold leading/trailing spaces and numbered list markers
# through md.convert. Private-use characters, which real text doesn't contain,
# so a "§" in the text ("see § 5") is kept.
_SPACE = "\ue000"
_NUM = "\ue001"
# Placeholder for an asterisk that is content, not emphasis.
_STAR = "\ue002"


class MarkdownTextFilter(BaseTextFilter):
    """Text filter that removes Markdown formatting from text content.

    Converts Markdown to plain text while preserving the overall structure,
    including leading and trailing spaces. Handles special cases like
    asterisks and table formatting. Supports selective filtering of code
    blocks and tables based on configuration.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Markdown text filtering.

        Parameters:
            enable_text_filter: Whether to apply Markdown filtering. Defaults to True.
            filter_code: Whether to remove code blocks from the text. Defaults to False.
            filter_tables: Whether to remove table content from the text. Defaults to False.
            filter_repeated_sequences: Whether to remove repeated sequences from the text. Defaults to True.
        """

        enable_text_filter: bool | None = True
        filter_code: bool | None = False
        filter_tables: bool | None = False
        filter_repeated_sequences: bool | None = True

    def __init__(self, params: InputParams | None = None, **kwargs):
        """Initialize the Markdown text filter.

        Args:
            params: Configuration parameters for filtering behavior.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._settings = params or MarkdownTextFilter.InputParams()
        self._in_code_block = False
        self._in_table = False
        self._interrupted = False

    async def update_settings(self, settings: Mapping[str, Any]):
        """Update the filter's configuration settings.

        Args:
            settings: Dictionary of setting names to values for configuration.
        """
        for key, value in settings.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)

    async def filter(self, text: str) -> str:
        """Apply Markdown filtering transformations to the input text.

        Args:
            text: The input text containing Markdown formatting to be filtered.

        Returns:
            The filtered text with Markdown formatting removed or converted.
        """
        if self._settings.enable_text_filter:
            filtered_text = text

            # Remove code blocks while their ``` fences are still in the text:
            # md.convert turns a fence into inline code and the HTML tag strip
            # then removes it.
            if self._settings.filter_code:
                filtered_text = self._remove_code_blocks(filtered_text)

            # Remove complete tables before the newline collapse below, which
            # would join a table to the text around it.
            if self._settings.filter_tables:
                filtered_text = _MARKDOWN_TABLE.sub("", filtered_text)

            # Hold escaped asterisks (\*) and asterisks between digits (2*3,
            # 2**3) in a placeholder: they're content, but md.convert and the
            # asterisk removal below would read them as emphasis.
            filtered_text = re.sub(
                r"\\\*|(?<=\d)\*+(?=\d)",
                lambda m: _STAR * m.group(0).count("*"),
                filtered_text,
            )

            # Replace images with their alt text, as links are replaced with
            # their text. md.convert would keep the alt text only as an <img>
            # attribute, which the HTML tag strip removes.
            filtered_text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", filtered_text)

            # Remove horizontal rules (---, ***, ___) before the newline collapse
            # below, which would indent them so md.convert no longer sees a rule.
            filtered_text = re.sub(
                r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", filtered_text, flags=re.MULTILINE
            )

            # Strip bullet (-, +, *) and blockquote (>) markers for the same
            # reason. A streamed list item or quote starts its chunk on a new
            # line, which the collapse would indent, leaving the marker to be
            # spoken. Any indent counts, since nested list items are indented.
            filtered_text = re.sub(
                r"^([ \t]*)(?:[-+*]|>+)[ \t]+(?=\S)", r"\1", filtered_text, flags=re.MULTILINE
            )

            # Strip markdown headers to plain text before the newline collapse
            # below. md.convert can't recognize a header once the collapse
            # injects a leading space, so we normalize headers ourselves.
            # First, drop blank lines before a header (keeping the marker) so
            # the collapse doesn't turn them into a leading space.
            filtered_text = re.sub(r"\n\s*\n(#{1,6}[ \t])", r"\n\1", filtered_text)
            # Then strip the leading marker and any "closed" trailing marker
            # (e.g. ## Title ##), leaving just the header text. The trailing
            # marker's surrounding spaces are absorbed only when the marker is
            # present, so genuine trailing whitespace (needed for word-by-word
            # streaming) is preserved on open headers.
            filtered_text = re.sub(
                r"^#{1,6}[ \t]+(.*?)(?:[ \t]+#+[ \t]*)?$",
                r"\1",
                filtered_text,
                flags=re.MULTILINE,
            )

            # Remove newlines and replace with a space only when there's no text before or after
            filtered_text = re.sub(r"^\s*\n", " ", filtered_text, flags=re.MULTILINE)

            # Remove backticks from inline code, but not from code blocks
            filtered_text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", filtered_text)

            # Remove repeated sequences of 5 or more characters
            if self._settings.filter_repeated_sequences:
                filtered_text = re.sub(r"(\S)(\1{4,})", "", filtered_text)

            # Preserve numbered list items with a unique marker
            filtered_text = re.sub(r"^(\d+\.)\s", rf"{_NUM}\1 ", filtered_text)

            # Preserve leading/trailing spaces with a unique marker
            # Critical for word-by-word streaming in bot-tts-text
            filtered_text = re.sub(
                r"^( +)|\s+$", lambda m: _SPACE * len(m.group(0)), filtered_text, flags=re.MULTILINE
            )

            # Remove space placeholders before tables, so that tables are converted to HTML
            # correctly
            filtered_text = filtered_text.replace(f"{_SPACE}| ", "| ")

            # Convert markdown to HTML
            extension = ["tables"] if self._settings.filter_tables else []
            md = Markdown(extensions=extension)
            filtered_text = md.convert(filtered_text)

            # Remove tables
            if self._settings.filter_tables:
                filtered_text = self.remove_tables(filtered_text)

            # Remove HTML tags
            filtered_text = _HTML_TAG.sub(
                lambda m: "" if m.group(1).lower() in _HTML_TAGS else m.group(0), filtered_text
            )

            # Replace HTML entities, including the numeric ones md.convert writes
            # an email autolink (<me@example.com>) as
            filtered_text = filtered_text.replace("&nbsp;", " ")
            filtered_text = html.unescape(filtered_text)

            # Remove double asterisks (consecutive without any exceptions)
            filtered_text = re.sub(r"\*\*", "", filtered_text)

            # Remove single asterisks at the start or end of words. A standalone
            # asterisk ("Press * then #", "5 * 3") is content and stays.
            filtered_text = re.sub(r"(^|\s)\*(?=\S)|(?<=\S)\*($|\s)", r"\1\2", filtered_text)

            # Remove struck-through text with its markers, which md.convert
            # doesn't handle: it's text the reply takes back, so it isn't spoken
            filtered_text = re.sub(r"[ \t]?~~(?=\S).+?(?<=\S)~~", "", filtered_text)

            # Remove Markdown table formatting
            filtered_text = re.sub(r"\|", "", filtered_text)
            filtered_text = re.sub(r"^\s*[-:]+\s*$", "", filtered_text, flags=re.MULTILINE)

            # Restore content asterisks
            filtered_text = filtered_text.replace(_STAR, "*")

            # Restore numbered list items
            filtered_text = filtered_text.replace(_NUM, "")

            # Restore leading and trailing spaces
            filtered_text = filtered_text.replace(_SPACE, " ")

            ## Make links more readable
            filtered_text = re.sub(r"https?://", "", filtered_text)

            return filtered_text
        else:
            return text

    async def handle_interruption(self):
        """Handle interruption events in the processing pipeline.

        Resets the filter state and clears any tracking variables for
        code blocks and tables.
        """
        self._interrupted = True
        self._in_code_block = False
        self._in_table = False

    async def reset_interruption(self):
        """Reset the filter state after an interruption has been handled.

        Clears the interrupted flag to restore normal operation.
        """
        self._interrupted = False

    #
    # Filter code
    #

    def _remove_code_blocks(self, text: str) -> str:
        """Remove code blocks from the input text.

        A code block can open and close within one text chunk or span several,
        so whether the text starts inside a block carries over between calls.
        """
        if self._interrupted:
            self._in_code_block = False
            return text

        parts = _CODE_FENCE.split(text)
        if len(parts) == 1:
            return "" if self._in_code_block else text

        # The parts between fences alternate between outside and inside a code
        # block, starting from where the previous chunk left off.
        kept = []
        for part in parts:
            if not self._in_code_block and part.strip():
                kept.append(part.strip())
            self._in_code_block = not self._in_code_block
        # The last part isn't followed by a fence, so undo its toggle.
        self._in_code_block = not self._in_code_block

        return " ".join(kept)

    #
    # Filter tables
    #
    def remove_tables(self, text: str) -> str:
        """Remove HTML tables from the input text.

        Handles cases where both start and end tags are in the same input,
        as well as tables that span multiple text chunks.

        Args:
            text: The text containing HTML tables to remove.

        Returns:
            The text with tables removed.
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

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

    def __init__(self, params: InputParams = InputParams(), **kwargs):
        super().__init__(**kwargs)
        self._settings = params

    def update_settings(self, settings: Mapping[str, Any]):
        for key, value in settings.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)

    def filter(self, text: str) -> str:
        if self._settings.enable_text_filter:
            # Replace newlines with spaces only when there's no text before or after
            text = re.sub(r"^\s*\n", " ", text, flags=re.MULTILINE)

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

            # Restore numbered list items
            filtered_text = filtered_text.replace("§NUM§", "")

            # Restore leading and trailing spaces
            filtered_text = re.sub("§", " ", filtered_text)

            return filtered_text
        else:
            return text

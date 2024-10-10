#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re

from markdown import Markdown

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MarkdownRemovalProcessor(FrameProcessor):
    """Removes Markdown formatting from text in TextFrames.

    Converts Markdown to plain text while preserving the overall structure,
    including leading and trailing spaces. Handles special cases like
    asterisks and table formatting.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._md = Markdown()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            cleaned_text = self._remove_markdown(frame.text)
            await self.push_frame(TextFrame(text=cleaned_text))
        else:
            await self.push_frame(frame, direction)

    def _remove_markdown(self, markdown_string: str) -> str:
        # Replace newlines with spaces to handle cases with leading newlines
        markdown_string = markdown_string.replace("\n", " ")

        # Preserve numbered list items with a unique marker, §NUM§
        markdown_string = re.sub(r"^(\d+\.)\s", r"§NUM§\1 ", markdown_string)

        # Preserve leading/trailing spaces with a unique marker, §
        # Critical for word-by-word streaming in bot-tts-text
        preserved_markdown = re.sub(
            r"^( +)|\s+$", lambda m: "§" * len(m.group(0)), markdown_string, flags=re.MULTILINE
        )

        # Convert markdown to HTML
        md = Markdown()
        html = md.convert(preserved_markdown)

        # Remove HTML tags
        text = re.sub("<[^<]+?>", "", html)

        # Replace HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")

        # Remove leading/trailing asterisks
        # Necessary for bot-tts-text, as they appear as literal asterisks
        text = re.sub(r"^\*{1,2}|\*{1,2}$", "", text)

        # Remove Markdown table formatting
        text = re.sub(r"\|", "", text)
        text = re.sub(r"^\s*[-:]+\s*$", "", text, flags=re.MULTILINE)

        # Restore numbered list items
        text = text.replace("§NUM§", "")

        # Restore leading and trailing spaces
        text = re.sub("§", " ", text)

        return text

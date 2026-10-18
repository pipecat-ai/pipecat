#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Split a leading tagged thought, such as ``<think>...</think>``, off streamed LLM text.

Some models write their reasoning into the answer text, wrapped in tags, ahead
of the answer itself. :class:`LeadingThoughtSplitter` separates that reasoning
from the text that follows so a service can emit it as thoughts rather than as
text to speak.
"""

from enum import Enum


class LeadingThoughtEvent(Enum):
    """The kind of a piece of text produced by :class:`LeadingThoughtSplitter`.

    Parameters:
        TEXT: Text outside the thought.
        THOUGHT_START: The thought began. Its text is empty.
        THOUGHT_TEXT: Text inside the thought.
        THOUGHT_END: The thought ended. Its text is empty.
    """

    TEXT = "text"
    THOUGHT_START = "thought_start"
    THOUGHT_TEXT = "thought_text"
    THOUGHT_END = "thought_end"


class _State(Enum):
    DETECTING = "detecting"
    IN_THOUGHT = "in_thought"
    AFTER_THOUGHT = "after_thought"
    CONTENT = "content"


class LeadingThoughtSplitter:
    """Split a tagged thought at the start of streamed text from the rest of it.

    Only a thought that opens the text is recognized, optionally after
    whitespace; tags anywhere later are left in the text. Tags may be split
    across chunks, so the start of the text is held back until it is clear
    whether it opens a thought, and inside a thought, a possible start of the
    closing tag is held back. Text that doesn't open with a thought passes
    through from the first chunk that rules one out, and whitespace between a
    thought and the text after it is dropped.

    Call :meth:`flush` when the text ends, or :meth:`reset` when it is cut
    short. Either readies the splitter for the next text.

    Example::

        splitter = LeadingThoughtSplitter("<think>", "</think>")
        for chunk in ["<thi", "nk>Hmm.</th", "ink> ", "Hello!"]:
            for kind, text in splitter.split(chunk):
                ...
        for kind, text in splitter.flush():
            ...
    """

    def __init__(self, open_tag: str, close_tag: str):
        """Initialize the splitter.

        Args:
            open_tag: The tag that opens a thought, e.g. ``"<think>"``.
            close_tag: The tag that closes a thought, e.g. ``"</think>"``.
        """
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._state = _State.DETECTING
        self._buffer = ""

    def split(self, text: str) -> list[tuple[LeadingThoughtEvent, str]]:
        """Split the next chunk of text.

        Args:
            text: The next chunk of streamed text.

        Returns:
            The ``(kind, text)`` pieces this chunk completes, in order.
        """
        if self._state == _State.CONTENT:
            return [(LeadingThoughtEvent.TEXT, text)]

        events: list[tuple[LeadingThoughtEvent, str]] = []

        if self._state == _State.DETECTING:
            self._buffer += text
            head = self._buffer.lstrip()
            if len(head) < len(self._open_tag) and self._open_tag.startswith(head):
                return events
            if not head.startswith(self._open_tag):
                passthrough, self._buffer = self._buffer, ""
                self._state = _State.CONTENT
                return [(LeadingThoughtEvent.TEXT, passthrough)]
            events.append((LeadingThoughtEvent.THOUGHT_START, ""))
            self._buffer = ""
            self._state = _State.IN_THOUGHT
            text = head[len(self._open_tag) :]

        if self._state == _State.IN_THOUGHT:
            self._buffer += text
            idx = self._buffer.find(self._close_tag)
            if idx == -1:
                # Hold back what could be the start of the closing tag.
                safe_end = len(self._buffer) - len(self._close_tag) + 1
                if safe_end > 0:
                    events.append((LeadingThoughtEvent.THOUGHT_TEXT, self._buffer[:safe_end]))
                    self._buffer = self._buffer[safe_end:]
                return events
            if thought := self._buffer[:idx]:
                events.append((LeadingThoughtEvent.THOUGHT_TEXT, thought))
            events.append((LeadingThoughtEvent.THOUGHT_END, ""))
            text = self._buffer[idx + len(self._close_tag) :]
            self._buffer = ""
            self._state = _State.AFTER_THOUGHT

        # After the thought: drop whitespace until the text resumes.
        if text := text.lstrip():
            self._state = _State.CONTENT
            events.append((LeadingThoughtEvent.TEXT, text))
        return events

    def flush(self) -> list[tuple[LeadingThoughtEvent, str]]:
        """End the text, releasing what was held back and closing an open thought.

        Returns:
            The remaining ``(kind, text)`` pieces, in order.
        """
        return self._end(release=True)

    def reset(self) -> list[tuple[LeadingThoughtEvent, str]]:
        """End text that was cut short, e.g. by an interruption or an error.

        Text still held back is discarded.

        Returns:
            ``THOUGHT_END`` if a thought was open, so the caller can close it;
            otherwise nothing.
        """
        return self._end(release=False)

    def _end(self, *, release: bool) -> list[tuple[LeadingThoughtEvent, str]]:
        events: list[tuple[LeadingThoughtEvent, str]] = []
        if self._state == _State.IN_THOUGHT:
            if self._buffer and release:
                events.append((LeadingThoughtEvent.THOUGHT_TEXT, self._buffer))
            events.append((LeadingThoughtEvent.THOUGHT_END, ""))
        elif self._state == _State.DETECTING and self._buffer and release:
            events.append((LeadingThoughtEvent.TEXT, self._buffer))
        self._buffer = ""
        self._state = _State.DETECTING
        return events

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parse tagged thoughts, such as ``<think>...</think>``, out of streamed LLM text.

Some models write their reasoning into the answer text, wrapped in tags, ahead
of the answer itself. :class:`TaggedLLMTextParser` separates that reasoning
from the text that follows so a service can emit it as thoughts rather than as
text to speak.
"""

from dataclasses import dataclass
from enum import Enum


class TaggedLLMTextParsedOutputChunkKind(Enum):
    """The kind of a :class:`TaggedLLMTextParsedOutputChunk`.

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


@dataclass
class TaggedLLMTextParsedOutputChunk:
    """A categorized piece of text produced by :class:`TaggedLLMTextParser`.

    Parameters:
        kind: What the piece is.
        text: The piece's text; empty for ``THOUGHT_START`` and ``THOUGHT_END``.
    """

    kind: TaggedLLMTextParsedOutputChunkKind
    text: str = ""


_Kind = TaggedLLMTextParsedOutputChunkKind


def _chunk(kind: _Kind, text: str = "") -> TaggedLLMTextParsedOutputChunk:
    return TaggedLLMTextParsedOutputChunk(kind, text)


class _State(Enum):
    DETECTING = "detecting"
    IN_THOUGHT = "in_thought"
    AFTER_THOUGHT = "after_thought"
    CONTENT = "content"


class TaggedLLMTextParser:
    """Parse a tagged thought out of streamed LLM text.

    Feed the text chunk by chunk to :meth:`parse_next_chunk`, which returns the
    categorized pieces each chunk completes: answer text, or a thought's start,
    text and end.

    A thought is recognized only at the start of the text, optionally after
    whitespace; tags later in the text are passed on as text. Tags may be split
    across chunks, so the start of the text is held back until it is clear
    whether it opens a thought, and inside a thought, a possible start of the
    closing tag is held back. Text that doesn't open with a thought passes
    through from the first chunk that rules one out, and whitespace between a
    thought and the text after it is dropped.

    Call :meth:`flush` when the text ends, or :meth:`reset` when it is cut
    short. Either readies the parser for the next text.

    Example::

        parser = TaggedLLMTextParser("<think>", "</think>")
        for text in ["<thi", "nk>Hmm.</th", "ink> ", "Hello!"]:
            for chunk in parser.parse_next_chunk(text):
                ...
        for chunk in parser.flush():
            ...
    """

    def __init__(self, thought_open_tag: str, thought_close_tag: str):
        """Initialize the parser.

        Args:
            thought_open_tag: The tag that opens a thought, e.g. ``"<think>"``.
            thought_close_tag: The tag that closes a thought, e.g. ``"</think>"``.
        """
        self._open_tag = thought_open_tag
        self._close_tag = thought_close_tag
        self._state = _State.DETECTING
        self._buffer = ""

    def parse_next_chunk(self, text: str) -> list[TaggedLLMTextParsedOutputChunk]:
        """Parse the next chunk of streamed text.

        Args:
            text: The next chunk of streamed text.

        Returns:
            The pieces this chunk completes, in order.
        """
        if self._state == _State.CONTENT:
            return [_chunk(_Kind.TEXT, text)]

        output: list[TaggedLLMTextParsedOutputChunk] = []

        if self._state == _State.DETECTING:
            self._buffer += text
            head = self._buffer.lstrip()
            if len(head) < len(self._open_tag) and self._open_tag.startswith(head):
                return output
            if not head.startswith(self._open_tag):
                passthrough, self._buffer = self._buffer, ""
                self._state = _State.CONTENT
                return [_chunk(_Kind.TEXT, passthrough)]
            output.append(_chunk(_Kind.THOUGHT_START))
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
                    output.append(_chunk(_Kind.THOUGHT_TEXT, self._buffer[:safe_end]))
                    self._buffer = self._buffer[safe_end:]
                return output
            if thought := self._buffer[:idx]:
                output.append(_chunk(_Kind.THOUGHT_TEXT, thought))
            output.append(_chunk(_Kind.THOUGHT_END))
            text = self._buffer[idx + len(self._close_tag) :]
            self._buffer = ""
            self._state = _State.AFTER_THOUGHT

        # After the thought: drop whitespace until the text resumes.
        if text := text.lstrip():
            self._state = _State.CONTENT
            output.append(_chunk(_Kind.TEXT, text))
        return output

    def flush(self) -> list[TaggedLLMTextParsedOutputChunk]:
        """End the text, releasing what was held back and closing an open thought.

        Returns:
            The remaining pieces, in order.
        """
        return self._end(release=True)

    def reset(self) -> list[TaggedLLMTextParsedOutputChunk]:
        """End text that was cut short, e.g. by an interruption or an error.

        Text still held back is discarded.

        Returns:
            A ``THOUGHT_END`` piece if a thought was open, so the caller can
            close it; otherwise nothing.
        """
        return self._end(release=False)

    def _end(self, *, release: bool) -> list[TaggedLLMTextParsedOutputChunk]:
        output: list[TaggedLLMTextParsedOutputChunk] = []
        if self._state == _State.IN_THOUGHT:
            if self._buffer and release:
                output.append(_chunk(_Kind.THOUGHT_TEXT, self._buffer))
            output.append(_chunk(_Kind.THOUGHT_END))
        elif self._state == _State.DETECTING and self._buffer and release:
            output.append(_chunk(_Kind.TEXT, self._buffer))
        self._buffer = ""
        self._state = _State.DETECTING
        return output

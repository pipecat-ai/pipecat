#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The frames a unit of text becomes once its expression tags are read.

Part of expressive conversation (alpha). :func:`segment_frames` turns the
segments an :class:`~pipecat.expressive.parser.ExpressionParser` emits into
the pieces a TTS service speaks as one request: each a frame, which is a
segment of its own for clients and a part of the context, and the text to send
to the TTS for it.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from pipecat.expressive.parser import Event, Pause, Segment, Text
from pipecat.frames.frames import AggregatedTextFrame, TextType

# The text types of what a TTS performs rather than says: events and pauses.
EXPRESSION_TEXT_TYPES = (TextType.EXPRESSION_EVENT, TextType.EXPRESSION_PAUSE)


def segment_frames(
    segments: list[Segment], text_type: TextType | str, *, trim: bool
) -> list[tuple[AggregatedTextFrame, str]]:
    """Build the frame for each segment, with the text to send to the TTS for it.

    Words become frames of ``text_type`` that carry their modes, and the text
    to speak keeps the spaces between them. Events and pauses become frames of
    their own text types, with no text to send until the TTS service renders
    them.

    Args:
        segments: The segments of one unit of text, such as a sentence.
        text_type: The text type of the unit's words.
        trim: Whether to trim the unit as a sentence is trimmed: the frames'
            text, and the text to speak at the ends of the unit.

    Returns:
        Each segment's frame, with the text to send to the TTS for it.
    """
    pieces: list[tuple[AggregatedTextFrame, str]] = []
    for segment in segments:
        if isinstance(segment, Text):
            text, raw = (
                (segment.text.strip(), segment.raw.strip()) if trim else (segment.text, segment.raw)
            )
            frame = AggregatedTextFrame(text, text_type, raw_text=raw, modes=list(segment.modes))
            pieces.append((frame, segment.text))
        else:
            pieces.append((_event_frame(segment), ""))

    words = [
        i for i, (frame, _) in enumerate(pieces) if frame.text_type not in EXPRESSION_TEXT_TYPES
    ]
    if trim and words:
        first, last = words[0], words[-1]
        pieces[first] = (pieces[first][0], pieces[first][1].lstrip())
        pieces[last] = (pieces[last][0], pieces[last][1].rstrip())
    return pieces


def _event_frame(segment: Event | Pause) -> AggregatedTextFrame:
    if isinstance(segment, Pause):
        return AggregatedTextFrame(
            str(segment.duration_ms), TextType.EXPRESSION_PAUSE, raw_text=segment.raw
        )
    return AggregatedTextFrame(segment.name, TextType.EXPRESSION_EVENT, raw_text=segment.raw)

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Perception: how the user said what they said.

Part of expressive conversation, which lets an agent hear how something was
said and answer in kind. A perception producer, such as an STT service or a
processor after it, reads each user turn and sets what it perceives as the
transcription's ``perception``: signals, such as a tone with the likeliest
labels and how sure it is of each, or a measure such as speech rate. It
declares the signals it reports with :class:`PerceptionCapabilities`, and
broadcasts them at start in a
:class:`~pipecat.frames.frames.PerceptionMetadataFrame`, so the LLM can be told
how to read them.

**Alpha.** These types are likely to change between releases.
"""

from dataclasses import dataclass, field


@dataclass
class PerceptionSignal:
    """One reading of how the user spoke.

    A signal is read either as labels or as a value.

    Parameters:
        name: What is read, such as ``"tone"`` or ``"speech_rate"``.
        labels: The likeliest labels, likeliest first, each with how sure the
            reading is, from 0 to 1, such as ``[("frustrated", 0.86),
            ("neutral", 0.10)]``. Empty for a signal read as a value.
        value: The measure, for a signal read as a value, such as words per
            second. None for a signal read as labels.
    """

    name: str
    labels: list[tuple[str, float]] = field(default_factory=list)
    value: float | None = None


@dataclass
class PerceptionTerm:
    """One signal a perception producer reports.

    Parameters:
        name: The signal's name, as in :attr:`PerceptionSignal.name`.
        description: A short line on how to read it, such as its unit or
            range, or empty.
        labels: The labels it can report, for a signal read as labels, or
            empty for a signal read as a value.
    """

    name: str
    description: str = ""
    labels: list[str] = field(default_factory=list)


@dataclass
class PerceptionCapabilities:
    """The signals a perception producer reports.

    Parameters:
        signals: Each signal the producer can report.
    """

    signals: list[PerceptionTerm] = field(default_factory=list)


def format_perception(signals: list[PerceptionSignal]) -> str:
    """Format a reading as written at the head of the user's message to the LLM.

    Labels keep their confidences, so the LLM can tell a clear reading from a
    guess: ``[heard: tone frustrated 0.86, neutral 0.10; speech_rate 4.1]``.

    Args:
        signals: The signals read.

    Returns:
        The formatted reading.
    """

    def signal_text(signal: PerceptionSignal) -> str:
        if signal.labels:
            labels = ", ".join(f"{label} {confidence:.2f}" for label, confidence in signal.labels)
            return f"{signal.name} {labels}"
        return f"{signal.name} {signal.value:g}"

    return f"[heard: {'; '.join(signal_text(signal) for signal in signals)}]"

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The labels Pipecat Tone's tagger chooses, and the expression tags they're written as.

The tagger labels with ``TAGGER_DELIVERY`` and ``TAGGER_SOUNDS``. A call's label space holds
those its TTS service performs and its policy offers, each with the name the service
performs it by, and a sentence's labels are written as tags in those names.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from pipecat.expressive.expression import (
    NEUTRAL_DELIVERY,
    TAGGER_DELIVERY,
    TAGGER_SOUNDS,
    ExpressionCapabilities,
    ExpressionTerm,
    TaggerDelivery,
    TaggerSound,
)
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy


@dataclass(frozen=True)
class LabelSpace:
    """The labels a call can use, each with the name its TTS service performs it by.

    Parameters:
        deliveries: Delivery label to the name to write, in ``TAGGER_DELIVERY`` order.
        sounds: Sound label to the name to write, in ``TAGGER_SOUNDS`` order.
        emphasis: Whether the service stresses ``^^words^^``.
    """

    deliveries: dict[str, str]
    sounds: dict[str, str]
    emphasis: bool

    @classmethod
    def resolve(cls, capabilities: ExpressionCapabilities, policy: ExpressionPolicy):
        """The labels the service performs and the policy offers.

        A breath is offered only with ``ExpressionGuidance.BREATHE`` in the policy.
        """
        sounds: list[TaggerSound] = [s for s in TAGGER_SOUNDS if s != "breath" or _breathes(policy)]
        vocabulary = capabilities.tagger_vocabulary
        return cls(
            deliveries=_resolve(
                TAGGER_DELIVERY,
                capabilities.delivery,
                capabilities.free_text_delivery,
                vocabulary,
                policy.delivery,
            ),
            sounds=_resolve(
                sounds,
                capabilities.sounds,
                capabilities.free_text_sounds,
                vocabulary,
                policy.sounds,
            ),
            emphasis=capabilities.emphasis,
        )


@dataclass(frozen=True)
class Span:
    """A delivery for some words, ``<<delivery: words>>``, by offsets in the sentence.

    Parameters:
        start: The offset of the first word's first character.
        end: The offset just past the last word.
        delivery: The words' delivery, a label from ``TAGGER_DELIVERY``.
    """

    start: int
    end: int
    delivery: str


@dataclass
class SentenceLabels:
    """What the tagger chose for one sentence, in its labels; usually nothing.

    Parameters:
        delivery: The sentence's delivery, or None for the voice's baseline.
        spans: Words in a delivery of their own, mainly slow details.
        emphasis: Words to stress, as ``(start, end)`` offsets.
        sound_before: A sound before the sentence, or None.
        sound_after: A sound after the sentence, or None.
    """

    delivery: str | None = None
    spans: list[Span] = field(default_factory=list)
    emphasis: list[tuple[int, int]] = field(default_factory=list)
    sound_before: str | None = None
    sound_after: str | None = None


class ReplyRenderer:
    """Writes one reply's sentences as tags, a delivery only where it changes."""

    def __init__(self, space: LabelSpace):
        """Initialize the renderer.

        Args:
            space: The labels the call can use.
        """
        self._space = space
        self._delivery: str | None = None

    def render(self, sentence: str, labels: SentenceLabels) -> str:
        """Write the next sentence of the reply with its labels as tags.

        Labels the space doesn't hold, or that would overlap or cut a word, are
        dropped, so the tags always speak exactly the sentence's words.
        """
        text = sentence.strip()
        if not text:
            return text
        kept = _keep(text, labels, self._space)
        names = self._space.deliveries

        # (offset, order, tag): at one offset a close comes before an open.
        inserts: list[tuple[int, int, str]] = []
        for span in kept.spans:
            inserts += [(span.start, 1, f"<<{names[span.delivery]}: "), (span.end, 0, ">>")]
        for start, end in kept.emphasis:
            inserts += [(start, 1, "^^"), (end, 0, "^^")]
        for at, _, tag in sorted(inserts, key=lambda i: (i[0], i[1]), reverse=True):
            text = text[:at] + tag + text[at:]

        if kept.sound_before:
            text = f"[[{self._space.sounds[kept.sound_before]}]] {text}"
        if kept.sound_after:
            text = f"{text} [[{self._space.sounds[kept.sound_after]}]]"
        delivery = names[kept.delivery] if kept.delivery else None
        if delivery != self._delivery:
            text = f"<<{delivery or NEUTRAL_DELIVERY}>> {text}"
            self._delivery = delivery
        return text


def _breathes(policy: ExpressionPolicy) -> bool:
    return ExpressionGuidance.BREATHE in policy.guidance


def _resolve(
    labels: Iterable[TaggerDelivery | TaggerSound],
    terms: list[ExpressionTerm],
    free_text: bool,
    vocabulary: dict[TaggerDelivery | TaggerSound, list[str]],
    offered: list[str] | None,
) -> dict[str, str]:
    """Each label's first name the service performs and the policy offers.

    The service's names for a label are tried best first, then, for a service that
    accepts free text, the label itself. A label with no such name is left out.
    """
    declared = {term.name for term in terms}
    resolved = {}
    for label in labels:
        names = vocabulary.get(label, [label])
        for name in [*names, label] if free_text else names:
            if (free_text or name in declared) and (offered is None or name in offered):
                resolved[label] = name
                break
    return resolved


def _keep(sentence: str, labels: SentenceLabels, space: LabelSpace) -> SentenceLabels:
    """The labels the space holds, on whole words and not overlapping."""
    taken: list[tuple[int, int]] = []
    spans = []
    for span in sorted(labels.spans, key=lambda s: s.start):
        if (
            span.delivery in space.deliveries
            and _words(sentence, span.start, span.end)
            and not _overlaps(taken, span.start, span.end)
        ):
            taken.append((span.start, span.end))
            spans.append(span)
    emphasis = []
    if space.emphasis:
        for start, end in sorted(labels.emphasis):
            if _words(sentence, start, end) and not _overlaps(taken, start, end):
                taken.append((start, end))
                emphasis.append((start, end))
    return SentenceLabels(
        delivery=labels.delivery if labels.delivery in space.deliveries else None,
        spans=spans,
        emphasis=emphasis,
        sound_before=labels.sound_before if labels.sound_before in space.sounds else None,
        sound_after=labels.sound_after if labels.sound_after in space.sounds else None,
    )


def _words(text: str, start: int, end: int) -> bool:
    return (
        0 <= start < end <= len(text)
        and not text[start].isspace()
        and not text[end - 1].isspace()
        and _boundary(text, start)
        and _boundary(text, end)
    )


def _boundary(text: str, at: int) -> bool:
    return at in (0, len(text)) or not (text[at - 1].isalnum() and text[at].isalnum())


def _overlaps(taken: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < e and s < end for s, e in taken)

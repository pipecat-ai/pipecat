#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The labels Pipecat Tone's tagger chooses, and the expression tags they're written as.

The tagger labels with ``CORE_DELIVERY`` and ``CORE_SOUNDS``, universal names, and draws
a breath as one of :data:`BREATH_KINDS`. A call's label space holds those its TTS service
performs and its policy offers, each with the name to write: its universal name, or a
developer's own, which the TTS service then performs by its own name.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import cast

from pipecat.expressive.expression import (
    CORE_DELIVERY,
    CORE_SOUNDS,
    NEUTRAL_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
)
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy

# The kinds the tagger draws a breath as, each a universal sound.
BREATH_KINDS: tuple[ExpressionSound, ...] = (
    ExpressionSound.LIGHT_BREATH,
    ExpressionSound.DEEP_BREATH,
    ExpressionSound.EXHALE,
)

# The services the model was trained with, by the names it reads for them. It reads
# "other" for any other.
TRAINED_SERVICES = ("cartesia", "elevenlabs", "fish", "minimax")
OTHER_SERVICE = "other"


@dataclass(frozen=True)
class LabelSpace:
    """The labels a call can use, each with the name to write it as.

    Parameters:
        deliveries: Delivery label to the name to write, in ``CORE_DELIVERY`` order.
        sounds: Sound label to the name to write, in ``CORE_SOUNDS`` order.
        emphasis: Whether the service stresses ``^^words^^``.
        tts: The service, by the name the model was trained to read for it.
        breaths: Kind of breath to the name to write, in ``BREATH_KINDS``
            order, where a breath is offered. The model chooses only a breath,
            and the tagger draws its kind.
    """

    deliveries: dict[str, str]
    sounds: dict[str, str]
    emphasis: bool
    tts: str = OTHER_SERVICE
    breaths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        capabilities: ExpressionCapabilities,
        policy: ExpressionPolicy,
        service_name: str = "",
        vocabulary: dict[ExpressionDelivery | ExpressionSound, list[str]] | None = None,
    ):
        """The labels the service performs and the policy offers.

        A label is written as the first of the developer's names for it, then its
        own, that the service performs and the policy offers. A breath, and its kinds,
        are offered only with ``ExpressionGuidance.BREATHE`` in the policy. A breath is
        offered where the service performs only some of its kinds, written as the first
        of them for a model that draws no kinds.

        Args:
            capabilities: What the service performs.
            policy: The pipeline's expression policy.
            service_name: The service's name, such as ``"ElevenLabsTTSService#0"``,
                which says which of the services the model was trained with it is.
                A service given a name of its own, such as ``name="tts"``, is read
                as one the model wasn't trained with.
            vocabulary: Names to try first for a label, before its own.
        """
        names = cast(dict[str, list[str]], vocabulary or {})
        sounds = [s for s in CORE_SOUNDS if s != ExpressionSound.BREATH or _breathes(policy)]
        resolved = _resolve(sounds, capabilities.sound_name, names, policy.sounds)
        # Each kind of breath is a sound of its own.
        breaths = (
            _resolve(BREATH_KINDS, capabilities.sound_name, names, policy.sounds)
            if _breathes(policy)
            else {}
        )
        # A breath is last in CORE_SOUNDS, so adding it keeps the order.
        if breaths and ExpressionSound.BREATH not in resolved:
            resolved[ExpressionSound.BREATH] = next(iter(breaths.values()))
        return cls(
            deliveries=_resolve(CORE_DELIVERY, capabilities.delivery_name, names, policy.delivery),
            sounds=resolved,
            emphasis=capabilities.emphasis,
            tts=_trained_service(service_name),
            breaths=breaths,
        )

    def sound_name(self, label: str) -> str | None:
        """The name to write for a sound or a kind of breath, if the space holds it."""
        return self.sounds.get(label) or self.breaths.get(label)


@dataclass(frozen=True)
class Span:
    """A delivery for some words, ``<<delivery: words>>``, by offsets in the sentence.

    Parameters:
        start: The offset of the first word's first character.
        end: The offset just past the last word.
        delivery: The words' delivery, a label from ``CORE_DELIVERY``.
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
            text = f"[[{self._space.sound_name(kept.sound_before)}]] {text}"
        if kept.sound_after:
            text = f"{text} [[{self._space.sound_name(kept.sound_after)}]]"
        delivery = names[kept.delivery] if kept.delivery else None
        if delivery != self._delivery:
            text = f"<<{delivery or NEUTRAL_DELIVERY}>> {text}"
            self._delivery = delivery
        return text


def _trained_service(service_name: str) -> str:
    name = service_name.lower()
    return next((service for service in TRAINED_SERVICES if service in name), OTHER_SERVICE)


def _breathes(policy: ExpressionPolicy) -> bool:
    return ExpressionGuidance.BREATHE in policy.guidance


def _resolve(
    labels: Iterable[ExpressionDelivery | ExpressionSound],
    performed_name: Callable[[str], str | None],
    vocabulary: dict[str, list[str]],
    offered: list[str] | None,
) -> dict[str, str]:
    """Each label's first name the service performs and the policy offers.

    The developer's names for a label are tried best first, then the label itself.
    A label with no such name is left out.
    """
    resolved = {}
    for label in labels:
        for name in [*vocabulary.get(label, []), label]:
            if performed_name(name) is not None and (offered is None or name in offered):
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
        sound_before=labels.sound_before if space.sound_name(labels.sound_before or "") else None,
        sound_after=labels.sound_after if space.sound_name(labels.sound_after or "") else None,
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

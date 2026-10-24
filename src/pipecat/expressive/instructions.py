#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The instruction that teaches an LLM to take part in expressive conversation.

Part of expressive conversation (alpha). The instruction has three parts: how
to read what was perceived of how the user spoke, built from a perception
producer's :class:`~pipecat.expressive.perception.PerceptionCapabilities`; the
tags the active TTS service can perform, built from its
:class:`~pipecat.expressive.expression.ExpressionCapabilities`; and the
developer's guidance on how to answer, from an
:class:`~pipecat.expressive.policy.ExpressionPolicy`.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from pipecat.expressive.expression import SLOW_DELIVERY, ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.perception import (
    PerceptionCapabilities,
    PerceptionSignal,
    PerceptionTerm,
    format_perception,
)
from pipecat.expressive.policy import ExpressionPolicy


def expression_instructions(
    capabilities: ExpressionCapabilities | None,
    policy: ExpressionPolicy | None = None,
    *,
    perception: PerceptionCapabilities | None = None,
) -> str | None:
    """Build the instruction for an LLM in an expressive conversation.

    Args:
        capabilities: What the TTS service that speaks the LLM's replies can
            perform, or None for nothing.
        policy: Which tags to offer, and guidance on how to answer.
        perception: The signals perceived of how the user spoke, or None when
            nothing is perceived.

    Returns:
        The instruction, or None when nothing is perceived and the TTS service
        can perform none of the tags offered.
    """
    policy = policy or ExpressionPolicy()
    parts = [
        part
        for part in (_perception_section(perception), _tags_section(capabilities, policy))
        if part
    ]
    if not parts:
        return None
    if policy.guidance:
        parts.append("\n".join(policy.guidance))
    return "\n\n".join(parts)


def _perception_section(perception: PerceptionCapabilities | None) -> str | None:
    if not perception or not perception.signals:
        return None
    such_as = ""
    example = next((term for term in perception.signals if len(term.labels) >= 2), None)
    if example:
        labels = [(example.labels[0], 0.86), (example.labels[1], 0.10)]
        such_as = f" such as {format_perception([PerceptionSignal(example.name, labels)])},"
    signals = "\n".join(_signal_line(term) for term in perception.signals)
    return (
        f"A user message may start with how the user sounded,{such_as} each signal with its "
        "likeliest labels and how sure the reading is, from 0 to 1, or with a measure. The "
        "reading can be wrong, so weigh it against their words. A reading below 0.5 is a "
        f"guess: don't act on it. Never write it yourself. The signals are:\n{signals}"
    )


def _signal_line(term: PerceptionTerm) -> str:
    if not term.labels:
        return f"- {term.name}: {(term.description or 'a measure').rstrip('.')}."
    line = f"- {term.name}: {', '.join(term.labels)}."
    return f"{line} {term.description}" if term.description else line


def _tags_section(
    capabilities: ExpressionCapabilities | None, policy: ExpressionPolicy
) -> str | None:
    if capabilities is None:
        return None
    tags = []
    delivery = _offered(capabilities.delivery, policy.delivery, capabilities.free_text_delivery)
    if delivery:
        tags.append(
            "- <<delivery>> at the start of a sentence speaks it, and the sentences after it, "
            "with that delivery, until the next delivery tag. <<neutral>> returns to your "
            "normal voice. A delivery is one word. "
            + _choices("deliveries", delivery, capabilities.free_text_delivery, "<<{}>>")
        )
        # A real span shows that its delivery comes before the colon and the
        # words to speak after it. Written with placeholders, a span reads as a
        # label and its value, as in <<delivery: sympathetic>>.
        example = _span_example(delivery)
        tags.append(
            f"- <<{example}: A B 7>> speaks just the words after the colon, A B 7, with the "
            f"delivery before it, {example}, on top of the delivery around them, and then goes "
            "back to it. Keep it to a few words."
        )
    sounds = _offered(capabilities.sounds, policy.sounds, capabilities.free_text_sounds)
    if sounds:
        tags.append(
            "- [[sound]] makes a sound at that point. "
            + _choices("sounds", sounds, capabilities.free_text_sounds, "[[{}]]")
        )
    if capabilities.emphasis:
        tags.append("- ^^word^^ stresses a word.")
    if capabilities.pause_range_ms:
        tags.append("- [[pause 500]] pauses for about that many milliseconds.")
    if not tags:
        return None
    return "You can direct how your words are spoken with tags:\n" + "\n".join(tags)


def unperformable_tags(capabilities: ExpressionCapabilities, policy: ExpressionPolicy) -> list[str]:
    """The tags a policy offers that a TTS service can't perform.

    Args:
        capabilities: What the TTS service can perform.
        policy: The policy to check.

    Returns:
        Each such tag as it would be written, such as ``"<<relieved>>"``.
    """
    tags = [
        f"<<{name}>>"
        for name in policy.delivery or []
        if not _performs(name, capabilities.delivery, capabilities.free_text_delivery)
    ]
    tags += [
        f"[[{name}]]"
        for name in policy.sounds or []
        if not _performs(name, capabilities.sounds, capabilities.free_text_sounds)
    ]
    return tags


def _performs(name: str, declared: list[ExpressionTerm], free_text: bool) -> bool:
    return free_text or any(term.name == name for term in declared)


def _offered(
    declared: list[ExpressionTerm], names: list[str] | None, free_text: bool
) -> list[ExpressionTerm]:
    """The terms to offer: those declared, narrowed to the policy's names."""
    if names is None:
        return declared
    by_name = {term.name: term for term in declared}
    if free_text:
        return [by_name.get(name, ExpressionTerm(name)) for name in names]
    return [by_name[name] for name in names if name in by_name]


def _span_example(delivery: list[ExpressionTerm]) -> str:
    """The delivery a span is shown with: slow, which details are said in, if offered."""
    names = [term.name for term in delivery]
    return SLOW_DELIVERY if SLOW_DELIVERY in names else names[0]


def _choices(kind: str, terms: list[ExpressionTerm], free_text: bool, tag: str) -> str:
    """The terms offered, each written as its tag, as in ``<<calm>>`` or ``[[laughs]]``."""

    def term_text(term: ExpressionTerm) -> str:
        name = tag.format(term.name)
        return f"{name} ({term.description})" if term.description else name

    listed = ", ".join(term_text(term) for term in terms)
    if free_text:
        return f"For example: {listed}. Others work too."
    return (
        f"Use only these {kind}, written exactly as listed: {listed}. "
        "Any other word in a tag is ignored, so if none fits, write no tag."
    )

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The instruction that teaches an LLM to take part in expressive conversation.

Part of expressive conversation (alpha). The instruction has three parts: how
to read what was perceived of how the user spoke, built from a perception
producer's :class:`~pipecat.expressive.perception.PerceptionCapabilities`; the
tags the active TTS service can perform, by their universal names, built from its
:class:`~pipecat.expressive.expression.ExpressionCapabilities`; and the
developer's guidance on how to answer, from an
:class:`~pipecat.expressive.policy.ExpressionPolicy`.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from collections.abc import Callable, Sequence

from pipecat.expressive.expression import (
    CORE_DELIVERY,
    CORE_SOUNDS,
    NEUTRAL_DELIVERY,
    SLOW_DELIVERY,
    ExpressionCapabilities,
    ExpressionTerm,
)
from pipecat.expressive.perception import (
    PerceptionCapabilities,
    PerceptionSignal,
    PerceptionTerm,
    format_perception,
)
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy


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
    tags = _tags_section(capabilities, policy)
    parts = [part for part in (_perception_section(perception), tags) if part]
    if not parts:
        return None
    # The presets are about tags, so an LLM taught none isn't given them.
    guidance = [
        line for line in policy.guidance if tags or not isinstance(line, ExpressionGuidance)
    ]
    if guidance:
        parts.append("\n".join(guidance))
    return "\n\n".join(parts)


def expression_policy_warning(
    capabilities: ExpressionCapabilities | None, policy: ExpressionPolicy | None
) -> str | None:
    """Tell what an expression policy asks for that a TTS service can't perform.

    Args:
        capabilities: What the TTS service can perform, or None for nothing.
        policy: Which tags to offer, and guidance on how to answer.

    Returns:
        The warning, or None when there's no policy or the service can
        perform every tag it offers.
    """
    if policy is None:
        return None
    if capabilities is None or _tags_section(capabilities, policy) is None:
        return (
            "the TTS service performs none of the expression policy's tags, so they "
            "and its guidance presets aren't taught"
        )
    if left_out := _left_out(capabilities, policy):
        return (
            f"the TTS service can't perform {', '.join(left_out)} from the expression "
            "policy, so they aren't offered"
        )
    return None


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
        f"A user message may start with how the user sounded, in parentheses,{such_as} each "
        "signal with its likeliest labels and how sure the reading is, from 0 to 1, or with a "
        "measure. The reading can be wrong, so weigh it against their words. A reading below "
        f"0.5 is a guess: don't act on it. Never write it yourself. The signals are:\n{signals}"
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
    delivery = [
        term
        for term in _offered(
            policy.delivery, CORE_DELIVERY, capabilities.delivery_name, capabilities.delivery
        )
        # Taught on its own, as the return to the voice's baseline.
        if term.name != NEUTRAL_DELIVERY
    ]
    if delivery:
        tags.append(
            "- <<delivery>> at the start of a sentence speaks it, and the sentences after it, "
            "with that delivery, until the next delivery tag. <<neutral>> returns to your "
            "normal voice. " + _choices("deliveries", delivery, "<<{}>>")
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
    # A breath is offered by default only to a policy that asks for breaths.
    core_sounds = [
        s for s in CORE_SOUNDS if s != "breath" or ExpressionGuidance.BREATHE in policy.guidance
    ]
    sounds = _offered(policy.sounds, core_sounds, capabilities.sound_name, capabilities.sounds)
    if sounds:
        tags.append(
            "- [[sound]] makes a sound at that point. " + _choices("sounds", sounds, "[[{}]]")
        )
    if capabilities.emphasis:
        tags.append("- ^^word^^ stresses a word.")
    if capabilities.pause_range_ms:
        tags.append("- [[pause 500]] pauses for about that many milliseconds.")
    if not tags:
        return None
    return "You can direct how your words are spoken with tags:\n" + "\n".join(tags)


def _left_out(capabilities: ExpressionCapabilities, policy: ExpressionPolicy) -> list[str]:
    """The policy's tags the service can't perform, each as it would be written."""
    tags = [
        f"<<{name}>>"
        for name in policy.delivery or []
        # Every service returns to its baseline: neutral is taught on its own.
        if name != NEUTRAL_DELIVERY and capabilities.delivery_name(name) is None
    ]
    tags += [f"[[{name}]]" for name in policy.sounds or [] if capabilities.sound_name(name) is None]
    return tags


def _offered(
    names: list[str] | None,
    core: Sequence[str],
    performed_name: Callable[[str], str | None],
    declared: list[ExpressionTerm],
) -> list[ExpressionTerm]:
    """The names to offer: the policy's, or the core ones, that the service performs.

    Each is offered with the description the service gives the name it performs it by.
    """
    descriptions = {term.name: term.description for term in declared}
    offered = []
    for name in core if names is None else names:
        performed = performed_name(name)
        if performed is not None:
            offered.append(ExpressionTerm(name, descriptions.get(performed, "")))
    return offered


def _span_example(delivery: list[ExpressionTerm]) -> str:
    """The delivery a span is shown with: slow, for details, unless only others are listed."""
    names = [term.name for term in delivery]
    return names[0] if names and SLOW_DELIVERY not in names else SLOW_DELIVERY


def _choices(kind: str, terms: list[ExpressionTerm], tag: str) -> str:
    """The terms offered, each written as its tag, as in ``<<calm>>`` or ``[[laugh]]``."""

    def term_text(term: ExpressionTerm) -> str:
        name = tag.format(term.name)
        return f"{name} ({term.description})" if term.description else name

    listed = ", ".join(term_text(term) for term in terms)
    return (
        f"Use only these {kind}, written exactly as listed: {listed}. "
        "Any other word in a tag is ignored, so if none fits, write no tag."
    )

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The instruction that teaches an LLM to write expression tags.

Part of expressive conversation (alpha). The instruction has two parts: the
protocol, which describes the tags the active TTS service can perform, built
from its :class:`~pipecat.expressive.expression.ExpressionCapabilities`; and
the developer's guidance on when to use them, from an
:class:`~pipecat.expressive.policy.ExpressionPolicy`.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from pipecat.expressive.expression import (
    NEUTRAL_DELIVERY,
    SLOW_DELIVERY,
    ExpressionCapabilities,
    ExpressionTerm,
)
from pipecat.expressive.policy import ExpressionPolicy


def expression_instructions(
    capabilities: ExpressionCapabilities | None, policy: ExpressionPolicy | None = None
) -> str | None:
    """Build the instruction for an LLM whose replies a TTS service speaks.

    Args:
        capabilities: What the TTS service can perform, or None for nothing.
        policy: Which tags to offer, and guidance on when to use them.

    Returns:
        The instruction, or None when the service can perform none of the
        tags offered.
    """
    if capabilities is None:
        return None
    policy = policy or ExpressionPolicy()

    tags = _tags(capabilities, policy)
    if not tags:
        return None

    parts = ["You can direct how your words are spoken with tags:\n" + "\n".join(tags)]
    if policy.guidance:
        parts.append("\n".join(policy.guidance))
    return "\n\n".join(parts)


def expression_policy_warning(
    capabilities: ExpressionCapabilities | None, policy: ExpressionPolicy | None
) -> str | None:
    """Tell what an expression policy asks for that a TTS service can't perform.

    Args:
        capabilities: What the TTS service can perform, or None for nothing.
        policy: Which tags to offer, and guidance on when to use them.

    Returns:
        The warning, or None when there's no policy or the service can
        perform every tag it offers.
    """
    if policy is None:
        return None
    if capabilities is None or not _tags(capabilities, policy):
        return (
            "the TTS service performs none of the expression policy's tags, so the "
            "policy and its guidance aren't taught"
        )
    if left_out := _left_out(capabilities, policy):
        return (
            f"the TTS service can't perform {', '.join(left_out)} from the expression "
            "policy, so they aren't offered"
        )
    return None


def _tags(capabilities: ExpressionCapabilities, policy: ExpressionPolicy) -> list[str]:
    """The instruction's lines for the tags offered, one per kind of tag."""
    tags = []
    delivery = [
        term
        for term in _offered(
            capabilities.delivery, policy.delivery, capabilities.free_text_delivery
        )
        # Taught on its own, as the return to the voice's baseline.
        if term.name != NEUTRAL_DELIVERY
    ]
    # A service that accepts any delivery is taught the tag even with no
    # examples to list, unless the policy names none.
    if delivery or (capabilities.free_text_delivery and policy.delivery is None):
        tags.append(
            "- <<delivery>> at the start of a sentence speaks it, and the sentences after it, "
            "with that delivery, until the next delivery tag. <<neutral>> returns to your "
            "normal voice. A delivery is a word or short phrase. "
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
    if sounds or (capabilities.free_text_sounds and policy.sounds is None):
        tags.append(
            "- [[sound]] makes a sound at that point. "
            + _choices("sounds", sounds, capabilities.free_text_sounds, "[[{}]]")
        )
    if capabilities.emphasis:
        tags.append("- ^^word^^ stresses a word.")
    if capabilities.pause_range_ms:
        tags.append("- [[pause 500]] pauses for about that many milliseconds.")
    return tags


def _left_out(capabilities: ExpressionCapabilities, policy: ExpressionPolicy) -> list[str]:
    """The policy's tags the instruction leaves out, each as it would be written."""
    tags = []
    for names, declared, free_text, tag in (
        (policy.delivery, capabilities.delivery, capabilities.free_text_delivery, "<<{}>>"),
        (policy.sounds, capabilities.sounds, capabilities.free_text_sounds, "[[{}]]"),
    ):
        offered = {term.name for term in _offered(declared, names, free_text)}
        tags += [tag.format(name) for name in names or [] if name not in offered]
    # Every service returns to its baseline: neutral is taught on its own.
    return [tag for tag in tags if tag != f"<<{NEUTRAL_DELIVERY}>>"]


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
    """The delivery a span is shown with: slow, for details, unless only others are listed."""
    names = [term.name for term in delivery]
    return names[0] if names and SLOW_DELIVERY not in names else SLOW_DELIVERY


def _choices(kind: str, terms: list[ExpressionTerm], free_text: bool, tag: str) -> str:
    """The terms offered, each written as its tag, as in ``<<calm>>`` or ``[[laughs]]``."""

    def term_text(term: ExpressionTerm) -> str:
        name = tag.format(term.name)
        return f"{name} ({term.description})" if term.description else name

    listed = ", ".join(term_text(term) for term in terms)
    if free_text and not listed:
        return "Write whatever fits."
    if free_text:
        return f"For example: {listed}. Others work too."
    return (
        f"Use only these {kind}, written exactly as listed: {listed}. "
        "Any other word in a tag is ignored, so if none fits, write no tag."
    )

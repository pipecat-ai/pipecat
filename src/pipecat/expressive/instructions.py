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

from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
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

    tags = []
    delivery = _offered(capabilities.delivery, policy.delivery, capabilities.free_text_delivery)
    if delivery:
        tags.append(
            "- <<delivery>> at the start of a sentence speaks it, and the sentences after it, "
            "with that delivery, until the next delivery tag. <<neutral>> returns to your "
            "normal voice. A delivery is one word. "
            + _choices("deliveries", delivery, capabilities.free_text_delivery)
        )
    sounds = _offered(capabilities.sounds, policy.sounds, capabilities.free_text_sounds)
    if sounds:
        tags.append(
            "- [[sound]] makes a sound at that point. "
            + _choices("sounds", sounds, capabilities.free_text_sounds, brackets=True)
        )
    if capabilities.emphasis:
        tags.append("- ^^word^^ stresses a word.")
    if capabilities.pause_range_ms:
        tags.append("- [[pause 500]] pauses for about that many milliseconds.")
    if not tags:
        return None

    parts = ["You can direct how your words are spoken with tags:\n" + "\n".join(tags)]
    if policy.guidance:
        parts.append("\n".join(policy.guidance))
    return "\n\n".join(parts)


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


def _choices(
    kind: str, terms: list[ExpressionTerm], free_text: bool, *, brackets: bool = False
) -> str:
    def term_text(term: ExpressionTerm) -> str:
        name = f"[[{term.name}]]" if brackets else term.name
        return f"{name} ({term.description})" if term.description else name

    listed = ", ".join(term_text(term) for term in terms)
    if free_text:
        return f"For example: {listed}. Others work too."
    return (
        f"Use only these {kind}, written exactly as listed: {listed}. "
        "Any other word in a tag is ignored, so if none fits, write no tag."
    )

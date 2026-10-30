#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Named text transforms a :class:`~pipecat.textflow.setup.TextFlowSetup` can refer to.

A setup names transforms rather than holding callables so it can be written as
YAML and edited in the page. The registry maps each name to a factory that
builds the transform from the options the setup gives it. The built-in
transforms from :mod:`pipecat.utils.text.transforms` are registered, plus a
few small ones that make what a stage does easy to see.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pipecat.utils.text.transforms import (
    VoiceFormatter,
    email_to_speech,
    expand_currency,
    expand_numbers,
    expand_percentages,
    expand_phone_numbers,
    expand_units,
    normalize_acronyms,
    normalize_dates,
    replace_text,
    strip_markdown,
)
from pipecat.utils.text.transforms.text_transform import TextTransform


@dataclass
class TransformOption:
    """One option a transform factory accepts.

    Parameters:
        name: The option's key in :attr:`TransformSpec.options`.
        kind: ``"bool"``, ``"int"``, ``"str"`` or ``"pairs"`` (a list of two-string lists).
        default: The value used when the setup leaves the option out.
        description: What the option does.
    """

    name: str
    kind: str
    default: Any
    description: str


@dataclass
class TransformEntry:
    """A registered transform.

    Parameters:
        name: The name a setup uses.
        description: What the transform does.
        factory: Builds the transform from an options mapping.
        options: The options the factory accepts.
    """

    name: str
    description: str
    factory: Callable[[dict[str, Any]], TextTransform]
    options: list[TransformOption] = field(default_factory=list)


_REGISTRY: dict[str, TransformEntry] = {}


def register_transform(
    name: str,
    factory: Callable[[dict[str, Any]], TextTransform],
    *,
    description: str = "",
    options: list[TransformOption] | None = None,
) -> None:
    """Register a transform under ``name`` so setups can refer to it.

    Args:
        name: The name a :class:`~pipecat.textflow.setup.TransformSpec` uses.
        factory: Builds the transform from the spec's options.
        description: What the transform does, shown in the page.
        options: The options the factory accepts, shown as controls in the page.
    """
    _REGISTRY[name] = TransformEntry(
        name=name, description=description, factory=factory, options=options or []
    )


def registered_transforms() -> list[TransformEntry]:
    """All registered transforms, in registration order."""
    return list(_REGISTRY.values())


def build_transform(name: str, options: dict[str, Any] | None = None) -> TextTransform:
    """Build the transform registered as ``name`` with ``options``.

    Raises:
        KeyError: If no transform is registered under ``name``.
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown transform {name!r}. Registered transforms: {known}")
    entry = _REGISTRY[name]
    resolved = {option.name: option.default for option in entry.options}
    resolved.update(options or {})
    return entry.factory(resolved)


def _plain(transform: TextTransform, description: str) -> None:
    register_transform(
        getattr(transform, "__name__", str(transform)),
        lambda _options, t=transform: t,
        description=description,
    )


_plain(strip_markdown, "Remove markdown markup: **bold** → bold, headings, lists, links.")
_plain(expand_currency, "$42.50 → forty-two dollars and fifty cents.")
_plain(expand_percentages, "12.5% → twelve point five percent.")
_plain(expand_units, "5 km → 5 kilometers.")
_plain(normalize_dates, "2024-01-15 → January 15th, two thousand and twenty-four.")
_plain(email_to_speech, "a@b.com → a at b dot com.")
_plain(expand_phone_numbers, "555-123-4567 → 5 5 5 1 2 3 4 5 6 7.")
_plain(normalize_acronyms, "IBM → I B M.")


def _expand_numbers(options: dict[str, Any]) -> TextTransform:
    cutoff = options.get("digit_cutoff")
    return expand_numbers(digit_cutoff=int(cutoff) if cutoff is not None else None)


register_transform(
    "expand_numbers",
    _expand_numbers,
    description="1994 → 1 9 9 4, for numbers above the cutoff.",
    options=[
        TransformOption("digit_cutoff", "int", 2025, "Numbers above this are read digit by digit."),
    ],
)


def _voice_formatter(options: dict[str, Any]) -> TextTransform:
    return VoiceFormatter(**options)


register_transform(
    "voice_formatter",
    _voice_formatter,
    description="The VoiceFormatter bundle of the built-in transforms, each one switchable.",
    options=[
        TransformOption("strip_markdown", "bool", True, "Remove markdown markup."),
        TransformOption("expand_phone_numbers", "bool", True, "Read phone numbers digit by digit."),
        TransformOption("normalize_acronyms", "bool", True, "Space out acronyms."),
        TransformOption("expand_currency", "bool", True, "Spell out currency amounts."),
        TransformOption("expand_numbers", "bool", False, "Read numbers digit by digit."),
        TransformOption("expand_percentages", "bool", True, "Spell out percentages."),
        TransformOption("expand_units", "bool", True, "Expand unit abbreviations."),
        TransformOption("email_to_speech", "bool", True, "Read email addresses aloud."),
        TransformOption("normalize_dates", "bool", True, "Spell out ISO dates."),
    ],
)


def _replace(options: dict[str, Any]) -> TextTransform:
    pairs = [(str(p), str(r)) for p, r in options.get("replacements") or []]
    return replace_text(pairs)


register_transform(
    "replace_text",
    _replace,
    description="Find-and-replace rules, each pattern a regular expression.",
    options=[
        TransformOption(
            "replacements", "pairs", [], "Rules as [pattern, replacement] pairs, applied in order."
        )
    ],
)


def _wrap(options: dict[str, Any]) -> TextTransform:
    prefix = str(options.get("prefix", ""))
    suffix = str(options.get("suffix", ""))

    async def wrap(text: str, text_type: str) -> str:
        return f"{prefix}{text}{suffix}"

    return wrap


register_transform(
    "wrap",
    _wrap,
    description="Put markup around the segment, such as a provider's <spell> tags.",
    options=[
        TransformOption("prefix", "str", "<spell>", "Text placed before the segment."),
        TransformOption("suffix", "str", "</spell>", "Text placed after the segment."),
    ],
)


def _redact(options: dict[str, Any]) -> TextTransform:
    keep = int(options.get("keep_last", 4))
    mask = str(options.get("mask", "X"))

    async def redact(text: str, text_type: str) -> str:
        head, tail = (text[:-keep], text[-keep:]) if keep else (text, "")
        return re.sub(r"[A-Za-z0-9]", mask, head) + tail

    return redact


register_transform(
    "redact",
    _redact,
    description="Mask letters and digits except the last few: 1234-5678 → XXXX-5678.",
    options=[
        TransformOption("keep_last", "int", 4, "How many trailing characters stay visible."),
        TransformOption("mask", "str", "X", "The character that replaces the rest."),
    ],
)


def _upper(_options: dict[str, Any]) -> TextTransform:
    async def upper(text: str, text_type: str) -> str:
        return text.upper()

    return upper


register_transform("upper", _upper, description="Uppercase the segment, to make a stage visible.")


def _strip_emoji(_options: dict[str, Any]) -> TextTransform:
    return replace_text([(r"[\U0001F300-\U0001FAFF☀-➿]️?", "")])


register_transform("strip_emoji", _strip_emoji, description="Remove emoji from the segment.")


def _drop(_options: dict[str, Any]) -> TextTransform:
    async def drop(text: str, text_type: str) -> str:
        return ""

    return drop


register_transform(
    "drop",
    _drop,
    description="Return nothing, so the stage drops the segment instead of passing it on.",
)


def _fail(_options: dict[str, Any]) -> TextTransform:
    async def fail(text: str, text_type: str) -> str:
        raise RuntimeError("transform failed on purpose")

    return fail


register_transform(
    "fail", _fail, description="Raise an error, to show how a failing transform is reported."
)

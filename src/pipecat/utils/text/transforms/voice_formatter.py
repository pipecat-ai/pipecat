#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configurable voice formatting bundle for TTS preprocessing."""

from pipecat.frames.frames import AggregationType


class VoiceFormatter:
    r"""Configurable bundle that applies a pipeline of voice-formatting transforms.

    Each option enables or disables one transform. Transforms are applied in a
    deliberate order: structural cleanup first, language expansions second, user
    replacements last.

    Example::

        formatter = VoiceFormatter(
            strip_markdown=True,
            expand_currency=True,
            number_digit_cutoff=2025,
            custom_replacements=[(r"\bDr\.", "Doctor")],
        )
        tts = CartesiaTTSService(
            text_transforms=[("*", formatter)],
        )
    """

    def __init__(
        self,
        *,
        strip_markdown: bool = True,
        expand_phone_numbers: bool = True,
        normalize_acronyms: bool = True,
        expand_currency: bool = True,
        expand_numbers: bool = False,
        number_digit_cutoff: int | None = None,
        expand_percentages: bool = True,
        expand_units: bool = True,
        email_to_speech: bool = True,
        normalize_dates: bool = True,
        custom_replacements: list[tuple[str, str]] | None = None,
    ):
        """Initialize the voice formatter.

        Args:
            strip_markdown: Strip Markdown formatting symbols (bold, italic, headers,
                code spans). Enabled by default.
            expand_phone_numbers: Space out phone number digits for individual
                pronunciation. Enabled by default.
            normalize_acronyms: Space out uppercase acronyms (e.g. ``"API"`` →
                ``"A P I"``). Enabled by default.
            expand_currency: Expand currency amounts to spoken form (e.g. ``"$42.50"``
                → ``"forty two dollars and fifty cents"``). Requires ``num2words``.
            expand_numbers: Expand numeric digits to spoken words. Disabled by default
                since it can affect numbers that are better read as digits. Requires
                ``num2words``.
            number_digit_cutoff: Numbers above this value are read digit-by-digit
                instead of as a quantity. Defaults to ``None`` (expand all numbers
                as words). Only used when ``expand_numbers=True``.
            expand_percentages: Expand percentage expressions (e.g. ``"50%"`` →
                ``"fifty percent"``). Requires ``num2words``.
            expand_units: Expand unit abbreviations (e.g. ``"5km"`` →
                ``"5 kilometers"``). Enabled by default.
            email_to_speech: Transform email addresses to spoken form. Enabled by
                default.
            normalize_dates: Expand date expressions to spoken form. Requires
                ``num2words``.
            custom_replacements: List of ``(regex_pattern, replacement)`` pairs applied
                after all other transforms.
        """
        self._transforms = []

        if strip_markdown:
            from pipecat.utils.text.transforms.strip_markdown import (
                strip_markdown as _strip_markdown,
            )

            self._transforms.append(_strip_markdown)

        # email_to_speech must run before expand_phone_numbers (phone regex matches
        # digit-only domains) and before normalize_acronyms (all-caps local parts get
        # letter-spaced, breaking the email pattern).
        if email_to_speech:
            from pipecat.utils.text.transforms.email import email_to_speech as _email_to_speech

            self._transforms.append(_email_to_speech)

        if expand_phone_numbers:
            from pipecat.utils.text.transforms.phone import (
                expand_phone_numbers as _expand_phone_numbers,
            )

            self._transforms.append(_expand_phone_numbers)

        if normalize_dates:
            from pipecat.utils.text.transforms.dates import normalize_dates as _normalize_dates

            self._transforms.append(_normalize_dates)

        if expand_currency:
            from pipecat.utils.text.transforms.currency import expand_currency as _expand_currency

            self._transforms.append(_expand_currency)

        if expand_percentages:
            from pipecat.utils.text.transforms.percentages import (
                expand_percentages as _expand_percentages,
            )

            self._transforms.append(_expand_percentages)

        # expand_units must run before normalize_acronyms: uppercase unit abbreviations
        # like "MB" or "MPH" would be letter-spaced first and then not recognized.
        if expand_units:
            from pipecat.utils.text.transforms.units import expand_units as _expand_units

            self._transforms.append(_expand_units)

        if normalize_acronyms:
            from pipecat.utils.text.transforms.acronyms import (
                normalize_acronyms as _normalize_acronyms,
            )

            self._transforms.append(_normalize_acronyms)

        if expand_numbers:
            from pipecat.utils.text.transforms.numbers import expand_numbers as _expand_numbers

            self._transforms.append(_expand_numbers(digit_cutoff=number_digit_cutoff))

        if custom_replacements:
            from pipecat.utils.text.transforms.replacements import replace_text

            self._transforms.append(replace_text(custom_replacements))

    async def __call__(self, text: str, aggregation_type: str | AggregationType) -> str:
        """Apply all configured transforms in order.

        Args:
            text: Input text to transform.
            aggregation_type: Aggregation type passed through to each transform.

        Returns:
            Transformed text ready for TTS synthesis.
        """
        for transform in self._transforms:
            text = await transform(text, aggregation_type)
        return text

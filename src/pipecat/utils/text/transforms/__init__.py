#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Built-in text transformers for TTS voice formatting.

Transformers are async callables with signature::

    async def transform(text: str, aggregation_type: str) -> str

They are registered with a TTS service via ``text_transforms``::

    tts = CartesiaTTSService(
        text_transforms=[("*", strip_markdown), ("*", expand_currency)],
    )

Or use :class:`VoiceFormatter` for a single configurable bundle::

    tts = CartesiaTTSService(
        text_transforms=[("*", VoiceFormatter(expand_currency=True))],
    )
"""

from pipecat.utils.text.transforms.acronyms import normalize_acronyms
from pipecat.utils.text.transforms.currency import expand_currency
from pipecat.utils.text.transforms.dates import normalize_dates
from pipecat.utils.text.transforms.email import email_to_speech
from pipecat.utils.text.transforms.numbers import expand_numbers
from pipecat.utils.text.transforms.percentages import expand_percentages
from pipecat.utils.text.transforms.phone import expand_phone_numbers
from pipecat.utils.text.transforms.replacements import replace_text
from pipecat.utils.text.transforms.strip_markdown import strip_markdown
from pipecat.utils.text.transforms.units import expand_units
from pipecat.utils.text.transforms.voice_formatter import VoiceFormatter

__all__ = [
    "normalize_acronyms",
    "expand_currency",
    "normalize_dates",
    "email_to_speech",
    "expand_numbers",
    "expand_percentages",
    "expand_phone_numbers",
    "replace_text",
    "strip_markdown",
    "expand_units",
    "VoiceFormatter",
]

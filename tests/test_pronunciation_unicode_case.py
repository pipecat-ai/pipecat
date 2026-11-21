#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.utils.text.transforms.pronunciations import pronunciation_transform


@pytest.mark.asyncio
@pytest.mark.parametrize("word", ["iris", "IRIS", "İris", "ıris", "sun", "ſun", "Kelvin", "Kelvin"])
async def test_pronunciation_case_insensitive_matches_keep_their_ipa(word):
    transform = pronunciation_transform(
        {"iris": "aɪrɪs", "sun": "sʌn", "kelvin": "kɛlvɪn"},
        lambda text, ipa: f"<{ipa}>{text}</{ipa}>",
    )
    ipa = "sʌn" if word in ("sun", "ſun") else "kɛlvɪn" if word in ("Kelvin", "Kelvin") else "aɪrɪs"
    assert await transform(word, "*") == f"<{ipa}>{word}</{ipa}>"

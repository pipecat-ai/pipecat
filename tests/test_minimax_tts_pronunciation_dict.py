#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMax TTS pronunciation dictionaries."""

import pytest

from tests.test_minimax_tts_text_normalization import _request


@pytest.mark.asyncio
@pytest.mark.parametrize("tone", [None, ["omg/oh my god", "Pipecat/(ˈpaɪpkæt)"]])
async def test_request_sends_the_pronunciation_dict_only_when_set(tone):
    request = await _request(pronunciation_dict=tone)
    assert request.get("pronunciation_dict") == ({"tone": tone} if tone else None)

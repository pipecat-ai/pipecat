#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Google HTTP TTS's SSML construction."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pipecat.services.google.tts", exc_type=ImportError)

from pipecat.services.google.tts import GoogleHttpTTSService  # noqa: E402


def test_text_is_escaped_inside_ssml():
    with patch.object(GoogleHttpTTSService, "_create_client", return_value=MagicMock()):
        service = GoogleHttpTTSService(
            settings=GoogleHttpTTSService.Settings(voice="en-US-Neural2-A")
        )

    ssml = service._construct_ssml("Q&A: is 3 < 5 > 2?")

    assert "Q&amp;A: is 3 &lt; 5 &gt; 2?" in ssml

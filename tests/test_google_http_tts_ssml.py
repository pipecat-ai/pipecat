#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SSML document built by GoogleHttpTTSService."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.cloud.texttospeech_v1")

from pipecat.services.google.tts import GoogleHttpTTSService  # noqa: E402


def _service(**settings) -> GoogleHttpTTSService:
    with patch.object(GoogleHttpTTSService, "_create_client", return_value=MagicMock()):
        return GoogleHttpTTSService(
            settings=GoogleHttpTTSService.Settings(voice="en-US-Standard-A", **settings)
        )


def test_reserved_characters_are_escaped():
    ssml = _service()._construct_ssml("Q&A about AT&T: 5 < 6 > 4")

    assert "Q&amp;A about AT&amp;T: 5 &lt; 6 &gt; 4" in ssml
    # The document stays well-formed, so Google does not reject it.
    root = ET.fromstring(ssml)
    assert "".join(root.itertext()) == "Q&A about AT&T: 5 < 6 > 4"


def test_reserved_characters_are_escaped_inside_prosody_and_emphasis():
    ssml = _service(rate="slow", emphasis="strong")._construct_ssml("Tom & Jerry")

    assert "Tom &amp; Jerry" in ssml
    ET.fromstring(ssml)

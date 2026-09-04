#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the xAI (Grok) Realtime voice setting."""

from pipecat.services.xai.realtime import events


def test_default_voice_matches_xai_default():
    """xAI's server-side default is "eve"; the session mirrors it."""
    assert events.SessionProperties().voice == "eve"


def test_default_voice_reaches_the_wire():
    """The default voice is serialized into the session update sent to xAI."""
    session = events.SessionUpdateEvent(session=events.SessionProperties()).model_dump(
        exclude_none=True
    )["session"]

    assert session["voice"] == "eve"


def test_voice_is_a_plain_string():
    """Built-in IDs and Custom Voices API IDs are both just strings."""
    assert events.SessionProperties(voice="my-custom-voice-1").voice == "my-custom-voice-1"

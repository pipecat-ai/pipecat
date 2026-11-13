#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the given / not-given helpers in pipecat.utils.types."""

import pytest

from pipecat.utils.types import NOT_GIVEN, assert_given, require_given


def test_assert_given_returns_the_value_including_none():
    assert assert_given("voice") == "voice"
    assert assert_given(None) is None


def test_assert_given_rejects_not_given():
    with pytest.raises(RuntimeError):
        assert_given(NOT_GIVEN)


def test_require_given_returns_a_value():
    assert require_given("en_US-lessac-medium", "Piper TTS voice") == "en_US-lessac-medium"


def test_require_given_rejects_not_given():
    with pytest.raises(RuntimeError):
        require_given(NOT_GIVEN, "Piper TTS voice")


@pytest.mark.parametrize("value", [None, ""], ids=["none", "blank"])
def test_require_given_rejects_missing_and_blank_with_the_given_name(value):
    with pytest.raises(ValueError, match="^Piper TTS voice must be specified$"):
        require_given(value, "Piper TTS voice")


def test_require_given_only_treats_the_empty_string_as_blank():
    assert require_given(0, "speed") == 0
    assert require_given(False, "flag") is False

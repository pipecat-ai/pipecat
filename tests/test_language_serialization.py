#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Language enum wire serialization.

These tests guard against serialization regressions where ``str(Language.X)``
is used to build wire values (query params, payloads). ``str()`` on a
``StrEnum`` member depends on the enum's ``__str__`` semantics: the stdlib
``StrEnum`` (Python 3.11+) returns the value (``"en"``), while a plain
``class StrEnum(str, Enum)`` fallback inherits ``Enum.__str__`` and returns
the member name (``"Language.EN"``). On the pipecat 0.0.10x line running
Python 3.10, this caused DeepgramSTTService to send ``language=Language.EN``
in the WebSocket query string, resulting in a 400 on every connect.

Wire serialization should always use ``.value`` so it never depends on
``__str__`` semantics.
"""

from enum import Enum

import pytest

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transcriptions.language import Language


@pytest.fixture
def legacy_strenum_str():
    """Simulate the pre-3.11 vendored StrEnum fallback.

    The fallback (``class StrEnum(str, Enum)`` with only ``__new__``) inherits
    ``Enum.__str__``, so ``str(Language.EN)`` returns ``"Language.EN"`` instead
    of ``"en"``. Restores the original ``__str__`` after the test.
    """
    original = Language.__str__
    Language.__str__ = Enum.__str__
    yield
    Language.__str__ = original


def test_deepgram_connect_kwargs_language_uses_enum_value():
    """The Deepgram connect kwargs must contain the enum value, not the member name."""
    service = DeepgramSTTService(api_key="test-key")
    kwargs = service._build_connect_kwargs()
    assert kwargs["language"] == "en"


def test_deepgram_connect_kwargs_language_is_str_independent(legacy_strenum_str):
    """Language serialization must not depend on StrEnum.__str__ semantics.

    With ``__str__`` behaving like the pre-3.11 vendored fallback,
    ``str(Language.EN)`` returns ``"Language.EN"``. The wire value must still
    be ``"en"`` because serialization goes through ``.value``.
    """
    assert str(Language.EN) == "Language.EN"  # sanity-check the simulation
    service = DeepgramSTTService(api_key="test-key")
    kwargs = service._build_connect_kwargs()
    assert kwargs["language"] == "en"


def test_deepgram_connect_kwargs_language_accepts_plain_string():
    """Plain string language values must pass through unchanged."""
    service = DeepgramSTTService(
        api_key="test-key",
        settings=DeepgramSTTService.Settings(language="en-US"),
    )
    kwargs = service._build_connect_kwargs()
    assert kwargs["language"] == "en-US"

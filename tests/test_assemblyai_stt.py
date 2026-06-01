#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the AssemblyAI streaming STT service connection parameters."""

from urllib.parse import parse_qs, urlparse

import pytest

from pipecat.services.assemblyai.stt import AssemblyAISTTService


def _query(service: AssemblyAISTTService) -> dict[str, list[str]]:
    """Build the WebSocket URL and return its parsed query parameters."""
    return parse_qs(urlparse(service._build_ws_url()).query)


def test_continuous_partials_defaults_to_true_for_u3_rt_pro():
    # u3-rt-pro is the default model; continuous_partials should be on by default.
    service = AssemblyAISTTService(api_key="test-key")
    assert _query(service)["continuous_partials"] == ["true"]


def test_continuous_partials_can_be_disabled():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(continuous_partials=False),
    )
    assert _query(service)["continuous_partials"] == ["false"]


def test_continuous_partials_omitted_for_universal_streaming():
    # continuous_partials is a U3Pro-only parameter and must not be sent otherwise.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-streaming-english"),
    )
    assert "continuous_partials" not in _query(service)


def test_interruption_delay_omitted_by_default():
    # Unset means "use the server default" — the param should not be sent.
    service = AssemblyAISTTService(api_key="test-key")
    assert "interruption_delay" not in _query(service)


def test_interruption_delay_sent_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(interruption_delay=300),
    )
    assert _query(service)["interruption_delay"] == ["300"]


def test_interruption_delay_omitted_for_universal_streaming():
    # interruption_delay is a U3Pro-only parameter.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english", interruption_delay=300
        ),
    )
    assert "interruption_delay" not in _query(service)


@pytest.mark.parametrize("value", [0, 1000])
def test_interruption_delay_boundaries_allowed(value):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(interruption_delay=value),
    )
    assert _query(service)["interruption_delay"] == [str(value)]


@pytest.mark.parametrize("value", [-1, 1001])
def test_interruption_delay_out_of_range_raises(value):
    with pytest.raises(ValueError):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(interruption_delay=value),
        )

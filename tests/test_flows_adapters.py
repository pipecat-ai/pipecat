#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the LLM adapter.

This module tests the LLMAdapter class used by the flow manager for
conversation-summary generation and formatting.

Tests:
    - Summary message formatting
"""

import pytest

from pipecat.flows.adapters import LLMAdapter


@pytest.fixture
def adapter():
    return LLMAdapter()


def test_format_summary_message(adapter):
    """Test summary message formatting."""
    message = adapter.format_summary_message("Test summary")
    assert message == {
        "role": "developer",
        "content": "Here's a summary of the conversation:\nTest summary",
    }

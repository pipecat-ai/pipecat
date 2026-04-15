#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pytest fixtures for Genesys AudioHook serializer tests.

These fixtures provide sample AudioHook protocol messages for testing
the GenesysAudioHookSerializer. They are scoped to this directory only.
"""

import pytest


@pytest.fixture
def sample_open_message():
    """Sample AudioHook open message from Genesys."""
    return {
        "version": "2",
        "type": "open",
        "seq": 1,
        "id": "test-session-123",
        "parameters": {
            "conversationId": "conv-456",
            "participant": {
                "ani": "+1234567890",
                "dnis": "+0987654321",
            },
            "media": [
                {
                    "type": "audio",
                    "format": "PCMU",
                    "channels": ["external"],
                    "rate": 8000,
                }
            ],
        },
    }


@pytest.fixture
def sample_open_message_with_input_variables():
    """Sample AudioHook open message with custom inputVariables from Genesys."""
    return {
        "version": "2",
        "type": "open",
        "seq": 1,
        "id": "test-session-123",
        "parameters": {
            "conversationId": "conv-456",
            "participant": {
                "ani": "+1234567890",
                "dnis": "+0987654321",
            },
            "media": [
                {
                    "type": "audio",
                    "format": "PCMU",
                    "channels": ["external"],
                    "rate": 8000,
                }
            ],
            "inputVariables": {
                "customer_id": "cust-789",
                "queue_name": "billing",
                "priority": "high",
                "language": "es-ES",
            },
        },
    }


@pytest.fixture
def sample_ping_message():
    """Sample AudioHook ping message."""
    return {
        "version": "2",
        "type": "ping",
        "seq": 5,
        "id": "test-session-123",
        "position": "PT10.5S",
    }


@pytest.fixture
def sample_close_message():
    """Sample AudioHook close message from Genesys."""
    return {
        "version": "2",
        "type": "close",
        "seq": 10,
        "id": "test-session-123",
        "position": "PT30.0S",
        "parameters": {
            "reason": "disconnect",
        },
    }


@pytest.fixture
def sample_pause_message():
    """Sample AudioHook pause message."""
    return {
        "version": "2",
        "type": "pause",
        "seq": 7,
        "id": "test-session-123",
        "position": "PT15.0S",
        "parameters": {
            "reason": "hold",
        },
    }


@pytest.fixture
def sample_update_message():
    """Sample AudioHook update message."""
    return {
        "version": "2",
        "type": "update",
        "seq": 8,
        "id": "test-session-123",
        "position": "PT20.0S",
        "parameters": {
            "participant": {
                "ani": "+1234567890",
                "dnis": "+0987654321",
                "name": "John Doe",
            },
        },
    }


@pytest.fixture
def sample_error_message():
    """Sample AudioHook error message."""
    return {
        "version": "2",
        "type": "error",
        "seq": 9,
        "id": "test-session-123",
        "parameters": {
            "code": 500,
            "message": "Internal server error",
        },
    }


@pytest.fixture
def sample_dtmf_message():
    """Sample AudioHook DTMF message."""
    return {
        "version": "2",
        "type": "dtmf",
        "seq": 6,
        "id": "test-session-123",
        "position": "PT12.0S",
        "parameters": {
            "digit": "5",
        },
    }


@pytest.fixture
def sample_dtmf_star_message():
    """Sample AudioHook DTMF message with star key."""
    return {
        "version": "2",
        "type": "dtmf",
        "seq": 6,
        "id": "test-session-123",
        "position": "PT12.0S",
        "parameters": {
            "digit": "*",
        },
    }


@pytest.fixture
def sample_dtmf_hash_message():
    """Sample AudioHook DTMF message with hash key."""
    return {
        "version": "2",
        "type": "dtmf",
        "seq": 6,
        "id": "test-session-123",
        "position": "PT12.0S",
        "parameters": {
            "digit": "#",
        },
    }

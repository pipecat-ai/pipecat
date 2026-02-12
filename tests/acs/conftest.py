#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mocked_call_connection_id():
    return "conn-test-123"


@pytest.fixture
def mocked_call_connection_client():
    conn = MagicMock()
    conn.hang_up = AsyncMock()
    return conn


@pytest.fixture
def mocked_call_automation_client(mocked_call_connection_client):
    client = MagicMock()
    client.get_call_connection.return_value = mocked_call_connection_client
    return client

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipecat.serializers.vonage import VonageFrameSerializer


class _FakeResponse:
    status = 204

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def text(self):
        raise AssertionError("204 Vonage hangup responses should not be treated as errors")


class _FakeClientSession:
    def __init__(self):
        self.put_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    def put(self, endpoint, *, headers, json):
        self.put_calls.append((endpoint, headers, json))
        return _FakeResponse()


@pytest.mark.asyncio
async def test_vonage_hangup_treats_204_as_success():
    session = _FakeClientSession()
    fake_aiohttp = SimpleNamespace(ClientSession=lambda: session)
    fake_jwt = SimpleNamespace(encode=lambda claims, private_key, algorithm: "token")

    serializer = VonageFrameSerializer(
        call_uuid="call-123",
        application_id="app-123",
        private_key="private-key",
    )

    with patch.dict(sys.modules, {"aiohttp": fake_aiohttp, "jwt": fake_jwt}):
        await serializer._hang_up_call()

    assert session.put_calls == [
        (
            "https://api.nexmo.com/v1/calls/call-123",
            {"Authorization": "Bearer token", "Content-Type": "application/json"},
            {"action": "hangup"},
        )
    ]

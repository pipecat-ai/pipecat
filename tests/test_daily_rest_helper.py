#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Daily REST helper."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.transports.daily.utils import (
    DailyMeetingTokenParams,
    DailyMeetingTokenProperties,
    DailyRESTHelper,
)

ROOM_URL = "https://example.daily.co/test-room"


def _make_session(token: str = "test-token") -> MagicMock:
    """Return a fake aiohttp session whose post() yields a 200 token response."""
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"token": token})
    response.text = AsyncMock(return_value="")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.post.return_value = response
    return session


def _posted_properties(session: MagicMock) -> dict:
    """Return the `properties` object from the JSON body posted to /meeting-tokens."""
    session.post.assert_called_once()
    url = session.post.call_args.args[0]
    assert url.endswith("/meeting-tokens")
    return session.post.call_args.kwargs["json"]["properties"]


@pytest.mark.asyncio
async def test_get_token_default_leaves_eject_at_token_exp_unset():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)

    token = await helper.get_token(ROOM_URL)

    assert token == "test-token"
    properties = _posted_properties(session)
    assert "eject_at_token_exp" not in properties
    assert properties["room_name"] == "test-room"
    assert properties["is_owner"] is True
    assert "exp" in properties


@pytest.mark.asyncio
async def test_get_token_eject_true_is_sent():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)

    await helper.get_token(ROOM_URL, eject_at_token_exp=True)

    properties = _posted_properties(session)
    assert properties["eject_at_token_exp"] is True


@pytest.mark.asyncio
async def test_get_token_eject_false_is_sent():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)

    await helper.get_token(ROOM_URL, eject_at_token_exp=False)

    properties = _posted_properties(session)
    assert properties["eject_at_token_exp"] is False


@pytest.mark.asyncio
async def test_get_token_keeps_eject_from_params_when_arg_unset():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)
    params = DailyMeetingTokenParams(
        properties=DailyMeetingTokenProperties(eject_at_token_exp=True, user_name="bot")
    )

    await helper.get_token(ROOM_URL, params=params)

    properties = _posted_properties(session)
    assert properties["eject_at_token_exp"] is True
    assert properties["user_name"] == "bot"
    assert properties["room_name"] == "test-room"
    assert properties["is_owner"] is True
    assert "exp" in properties


@pytest.mark.asyncio
async def test_get_token_default_sends_owner_and_one_hour_exp():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)

    before = time.time()
    await helper.get_token(ROOM_URL)
    after = time.time()

    properties = _posted_properties(session)
    assert properties["is_owner"] is True
    assert int(before + 3600) <= properties["exp"] <= int(after + 3600)


@pytest.mark.asyncio
async def test_get_token_keeps_exp_from_params_when_expiry_time_unset():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)
    params = DailyMeetingTokenParams(properties=DailyMeetingTokenProperties(exp=1234567890))

    await helper.get_token(ROOM_URL, params=params)

    assert _posted_properties(session)["exp"] == 1234567890


@pytest.mark.asyncio
async def test_get_token_expiry_time_arg_overrides_params_exp():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)
    params = DailyMeetingTokenParams(properties=DailyMeetingTokenProperties(exp=1234567890))

    before = time.time()
    await helper.get_token(ROOM_URL, expiry_time=60, params=params)
    after = time.time()

    exp = _posted_properties(session)["exp"]
    assert exp != 1234567890
    assert int(before + 60) <= exp <= int(after + 60)


@pytest.mark.asyncio
async def test_get_token_keeps_is_owner_from_params_when_owner_unset():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)
    params = DailyMeetingTokenParams(properties=DailyMeetingTokenProperties(is_owner=False))

    await helper.get_token(ROOM_URL, params=params)

    assert _posted_properties(session)["is_owner"] is False


@pytest.mark.asyncio
async def test_get_token_owner_arg_overrides_params_is_owner():
    session = _make_session()
    helper = DailyRESTHelper(daily_api_key="key", aiohttp_session=session)
    params = DailyMeetingTokenParams(properties=DailyMeetingTokenProperties(is_owner=True))

    await helper.get_token(ROOM_URL, owner=False, params=params)

    assert _posted_properties(session)["is_owner"] is False

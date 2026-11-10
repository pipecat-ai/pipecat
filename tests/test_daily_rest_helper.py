#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Daily REST helper."""

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

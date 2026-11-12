#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.services.google.image import GoogleImageGenService


def _service_with_mock_client(mock_genai_client_class):
    mock_client = MagicMock()
    mock_client.aio.aclose = AsyncMock()
    mock_genai_client_class.return_value = mock_client
    return GoogleImageGenService(api_key="test-api-key"), mock_client


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_cleanup_closes_genai_session(mock_genai_client_class):
    service, mock_client = _service_with_mock_client(mock_genai_client_class)

    await service.cleanup()

    mock_client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_genai_session_is_closed_once_across_repeated_cleanup(mock_genai_client_class):
    service, mock_client = _service_with_mock_client(mock_genai_client_class)

    await service.cleanup()
    await service.cleanup()

    mock_client.aio.aclose.assert_awaited_once()

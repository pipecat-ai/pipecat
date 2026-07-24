import asyncio
from unittest.mock import AsyncMock

import pytest

from pipecat.services.deepgram.tts import DeepgramTTSService


@pytest.mark.asyncio
async def test_disconnect_closes_websocket_before_canceling_receive_task():
    service = DeepgramTTSService(api_key="test")
    receive_task = object()
    service._receive_task = receive_task
    service._disconnect_websocket = AsyncMock()
    service.cancel_task = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await service._disconnect()

    service._disconnect_websocket.assert_awaited_once()
    service.cancel_task.assert_awaited_once_with(receive_task)
    assert service._receive_task is None
    assert service._disconnecting is True

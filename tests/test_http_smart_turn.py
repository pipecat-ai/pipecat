#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for HttpSmartTurnAnalyzer.

The prediction runs on the analyzer's model thread, which has no event loop, so
the HTTP request has to be handed back to the loop that owns the aiohttp session.
"""

import numpy as np
import pytest

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.http_smart_turn import HttpSmartTurnAnalyzer


class _FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeSession:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None):
        self.requests = 0
        self._response = response
        self._error = error

    def post(self, *args, **kwargs):
        self.requests += 1
        if self._error:
            raise self._error
        return self._response


def _analyzer(session: _FakeSession) -> HttpSmartTurnAnalyzer:
    analyzer = HttpSmartTurnAnalyzer(
        url="http://localhost/predict",
        aiohttp_session=session,
        sample_rate=16000,
        params=SmartTurnParams(stop_secs=1.0),
    )
    analyzer.append_audio(np.ones(1600, dtype=np.int16).tobytes(), is_speech=True)
    return analyzer


@pytest.mark.asyncio
async def test_request_is_sent_from_model_thread():
    session = _FakeSession(_FakeResponse(200, {"prediction": 1, "probability": 0.9}))
    analyzer = _analyzer(session)

    state, metrics = await analyzer.analyze_end_of_turn()
    await analyzer.cleanup()

    assert session.requests == 1
    assert state == EndOfTurnState.COMPLETE
    assert metrics is not None and metrics.is_complete


@pytest.mark.asyncio
async def test_timeout_completes_turn():
    session = _FakeSession(error=TimeoutError())
    analyzer = _analyzer(session)

    state, _ = await analyzer.analyze_end_of_turn()
    await analyzer.cleanup()

    assert state == EndOfTurnState.COMPLETE

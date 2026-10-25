#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how AWSPollyTTSService opens and closes its Polly client."""

import aiobotocore.session
import pytest

from pipecat.frames.frames import CancelFrame, EndFrame
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.test_aws_credentials import _renewing_credentials, no_configured_keys  # noqa: F401


class _Stream:
    async def read(self) -> bytes:
        return b"\x00\x00" * 1600


class _Polly:
    async def synthesize_speech(self, **params):
        return {"AudioStream": _Stream()}


class _ClientContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        self._session.opened += 1
        return _Polly()

    async def __aexit__(self, *exc):
        self._session.closed += 1
        return False


class _Session:
    """Counts the Polly clients opened and closed through it."""

    def __init__(self):
        self.opened = 0
        self.closed = 0

    def create_client(self, *args, **kwargs):
        return _ClientContext(self)


def _make_service() -> tuple[AWSPollyTTSService, _Session]:
    service = AWSPollyTTSService(api_key="secret", aws_access_key_id="key", region="us-east-1")
    service._sample_rate = 16000
    service._task_manager = TaskManager()
    session = _Session()
    service._aws_session = session
    return service, session


async def _speak(service: AWSPollyTTSService, text: str):
    return [frame async for frame in service.run_tts(text, "ctx")]


@pytest.mark.asyncio
async def test_sentences_share_one_polly_client():
    """Every sentence goes through the same Polly client."""
    service, session = _make_service()

    for text in ("One.", "Two.", "Three."):
        await _speak(service, text)

    assert session.opened == 1
    assert session.closed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()], ids=["stop", "cancel"])
async def test_stopping_closes_the_polly_client(frame):
    """Stopping or cancelling the service closes its client; speaking again opens a new one."""
    service, session = _make_service()
    await _speak(service, "One.")

    if isinstance(frame, EndFrame):
        await service.stop(frame)
    else:
        await service.cancel(frame)
    assert session.closed == 1

    await _speak(service, "Two.")
    assert session.opened == 2


@pytest.mark.asyncio
async def test_shared_client_signs_with_renewed_credentials(no_configured_keys):  # noqa: F811
    """Without configured keys, the shared client still renews its credentials."""
    service = AWSPollyTTSService(region="us-east-1")
    service._aws_session = aiobotocore.session.get_session()
    service._aws_session._credentials = _renewing_credentials()

    client = await service._get_client()
    signing_keys = [
        (await client._request_signer._credentials.get_frozen_credentials()).access_key
        for _ in range(2)
    ]
    await service._close_client()

    assert signing_keys[0] != signing_keys[1]

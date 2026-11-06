#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime streaming settings, optional dependency and client lifecycle."""

import subprocess
import sys
from dataclasses import fields
from unittest.mock import Mock

import pytest
from rimelabs_sdk import RimeAuthenticationError

from pipecat.frames.frames import (
    ErrorFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.rime import stt as rime
from pipecat.services.rime.stt import RimeSTTService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.types import is_given
from tests.rime_stt_helpers import FakeRime, audio


def test_settings_are_complete_copied_and_construction_is_lazy(monkeypatch):
    constructor = Mock(side_effect=AssertionError("construction must not create a client"))
    monkeypatch.setattr(rime, "Rime", constructor)
    terms = [" Pipecat ", "Rime", "Rime"]
    settings = RimeSTTService.Settings(language=Language.EN_US, context_terms=terms)
    service = RimeSTTService(api_key="test", settings=settings)
    terms.append("later")
    assert service._settings.context_terms == (" Pipecat ", "Rime", "Rime")
    assert service._settings.language == "en-US"
    assert service._settings.model is None
    assert not constructor.called
    assert all(is_given(getattr(service._settings, f.name)) for f in fields(service._settings))
    assert all(
        not is_given(getattr(RimeSTTService.Settings(), f.name))
        for f in fields(settings)
        if f.name != "extra"
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
@pytest.mark.parametrize("parameter", ["request_timeout", "shutdown_timeout"])
def test_invalid_timeout(timeout, parameter):
    with pytest.raises(ValueError, match=parameter):
        RimeSTTService(api_key="test", **{parameter: timeout})


def test_client_and_api_key_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either client or api_key"):
        RimeSTTService(client=FakeRime(), api_key="test")


@pytest.mark.asyncio
async def test_injected_client_survives_shutdown_and_repeated_cleanup():
    client = FakeRime(automatic=True)
    service = RimeSTTService(client=client)
    await run_test(
        service,
        frames_to_send=[VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame()],
    )
    await service.cleanup()
    assert not client.closed


@pytest.mark.asyncio
async def test_settings_update_is_atomic():
    service = RimeSTTService(api_key="test")
    with pytest.raises(ValueError, match="mode"):
        await service._update_settings(service.Settings(language="es-MX", mode="invalid"))
    assert service._settings.language == "en"
    await service._update_settings(service.Settings(language="es-MX"))
    assert service._settings.language == "es-MX"
    with pytest.raises(ValueError, match="model"):
        await service._update_settings(service.Settings(model="unknown"))
    assert service._settings.model is None


@pytest.mark.asyncio
async def test_partial_startup_reports_error_and_cleans_up(monkeypatch):
    monkeypatch.setattr(
        rime, "Rime", Mock(side_effect=RimeAuthenticationError("missing credential"))
    )
    service = RimeSTTService(api_key="")
    _, upstream = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            audio(),
            VADUserStoppedSpeakingFrame(),
        ],
    )
    errors = [frame for frame in upstream if isinstance(frame, ErrorFrame)]
    assert len(errors) == 1
    assert not service.is_usable
    await service.cleanup()


def test_missing_sdk_installation_hint():
    code = """
import sys
sys.modules['rimelabs_sdk'] = None
import pipecat.services.rime
try:
    import pipecat.services.rime.stt
except ImportError as error:
    assert 'pipecat-ai[rime-stt]' in str(error)
else:
    raise AssertionError('missing SDK must fail with an installation hint')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

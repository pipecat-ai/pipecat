#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime STT settings, optional dependency and client lifecycle."""

import subprocess
import sys
from unittest.mock import Mock

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import ErrorFrame, STTUpdateSettingsFrame
from pipecat.services.rime import stt as rime
from pipecat.services.rime.stt import RimeSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from tests.rime_stt_helpers import FakeRime, harness, send


def test_default_settings_and_lazy_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("construction must not create a client"))
    monkeypatch.setattr(rime, "Rime", constructor)
    service = RimeSTTService(api_key="test")
    settings = service._settings
    assert settings.model is None
    assert settings.language == "en"
    assert settings.mode == "written"
    assert tuple(settings.context_terms) == ()
    assert settings.output_contract == "revised_hypotheses"
    assert not constructor.called


def test_invalid_construction_settings_raise():
    with pytest.raises(ValueError, match="model"):
        RimeSTTService(api_key="test", settings=RimeSTTService.Settings(model="unknown"))
    with pytest.raises(ValueError, match="mode"):
        RimeSTTService(api_key="test", settings=RimeSTTService.Settings(mode="invalid"))


@pytest.mark.asyncio
async def test_language_is_stored_as_its_tag():
    service = RimeSTTService(api_key="test", settings=RimeSTTService.Settings(language=Language.ES))
    assert type(service._settings.language) is str and service._settings.language == "es"
    await service._update_settings(service.Settings(language=Language.EN_US))
    assert type(service._settings.language) is str and service._settings.language == "en-US"
    await service._update_settings(service.Settings(language="xx-YY"))
    assert service._settings.language == "xx-YY"


@pytest.mark.parametrize(
    "terms, match",
    [
        ([""], "nonempty"),
        (["Rime", "  "], "nonempty"),
        (["call <|endoftext|> now"], "control markers"),
        (["x" * 1025], "1,024"),
        (["x" * 512, "y" * 511], "1,024"),
    ],
)
def test_invalid_context_terms(terms, match):
    with pytest.raises(ValueError, match=match):
        RimeSTTService(api_key="test", settings=RimeSTTService.Settings(context_terms=terms))


def test_context_terms_at_the_byte_limit():
    terms = ["x" * 511, "y" * 511]
    service = RimeSTTService(api_key="test", settings=RimeSTTService.Settings(context_terms=terms))
    assert list(service._settings.context_terms) == terms


@pytest.mark.asyncio
async def test_model_and_extra_updates_are_ignored():
    service = RimeSTTService(api_key="test")
    delta = service.Settings(language="es", model="unknown")
    delta.extra = {"speed": 2}
    assert await service._update_settings(delta) == {"language": "en"}
    assert service._settings.language == "es"
    assert service._settings.model is None
    assert service._settings.extra == {}


@pytest.mark.asyncio
async def test_invalid_settings_update_is_rejected_whole(monkeypatch):
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: FakeRime())
    service = RimeSTTService(api_key="test")
    async with harness(service) as (_, upstream):
        await send(
            service,
            STTUpdateSettingsFrame(delta=service.Settings(language="es-MX", mode="invalid")),
        )
        errors = await upstream.wait_for(ErrorFrame)
        assert errors[0].category is ErrorCategory.APPLICATION and "mode" in errors[0].error
        assert service._settings.language == "en"
        assert service.is_usable
        await send(service, STTUpdateSettingsFrame(delta=service.Settings(language="es-MX")))
        assert service._settings.language == "es-MX"


@pytest.mark.asyncio
async def test_client_is_created_on_start_and_closed_on_cleanup(monkeypatch):
    client = FakeRime()
    created = []
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: created.append(kwargs) or client)
    service = RimeSTTService(api_key="secret")
    async with harness(service):
        assert created == [{"api_key": "secret"}]
    assert client.closed


def test_missing_sdk_installation_hint():
    code = """
import sys
sys.modules['rimelabs_sdk'] = None
import pipecat.services.rime
try:
    import pipecat.services.rime.stt
except ImportError as error:
    assert 'Missing module' in str(error)
else:
    raise AssertionError('missing SDK must fail with an installation hint')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

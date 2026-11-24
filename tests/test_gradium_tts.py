#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""GradiumTTSService: the setup message selects the configured model."""

import pytest
from loguru import logger

from pipecat.services.gradium.tts import GradiumTTSService


@pytest.fixture
def warnings():
    """The warning messages logged during the test."""
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    yield records
    logger.remove(sink)


def test_setup_message_selects_the_default_model():
    service = GradiumTTSService(api_key="test-key")

    assert service._build_setup_msg("ctx")["model_name"] == "default"


def test_setup_message_selects_the_configured_model():
    service = GradiumTTSService(
        api_key="test-key",
        settings=GradiumTTSService.Settings(model="gradium-tts-beta-202611"),
    )

    assert service._build_setup_msg("ctx")["model_name"] == "gradium-tts-beta-202611"


@pytest.mark.asyncio
async def test_a_model_update_applies_to_the_next_context(warnings):
    service = GradiumTTSService(api_key="test-key")

    await service._update_settings(GradiumTTSService.Settings(model="gradium-tts-beta-202611"))

    assert service._build_setup_msg("ctx")["model_name"] == "gradium-tts-beta-202611"
    assert not any("not currently supported" in w for w in warnings)

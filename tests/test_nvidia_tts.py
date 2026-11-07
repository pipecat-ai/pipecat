#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for NvidiaTTSService setup."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("riva.client")

from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


@pytest.mark.asyncio
async def test_setup_keeps_the_event_loop_running_while_the_server_answers(monkeypatch):
    service = NvidiaTTSService(api_key="test-key")
    answered = threading.Event()

    def get_synthesis_config(_request):
        answered.wait(timeout=5)

    stub = SimpleNamespace(GetRivaSynthesisConfig=get_synthesis_config)
    monkeypatch.setattr(service, "_service", SimpleNamespace(stub=stub))

    setup = asyncio.create_task(service.setup(frame_processor_setup(TaskManager())))
    try:
        await asyncio.sleep(0.1)
        assert not setup.done()

        answered.set()
        await setup
    finally:
        answered.set()
        await service.cleanup()

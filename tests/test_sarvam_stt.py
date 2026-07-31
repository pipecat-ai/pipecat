#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

pytest.importorskip("sarvamai")

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


def test_sarvam_vad_signals_recommend_external_strategies():
    """With ``vad_signals`` on, Sarvam's boundaries are what drive turns."""
    service = SarvamSTTService(
        api_key="test-key",
        settings=SarvamSTTService.Settings(model="saaras:v3", vad_signals=True),
    )
    strategies = service.service_metadata_frame().user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)


def test_sarvam_without_vad_signals_recommends_no_strategies():
    """Without them Sarvam proposes no turns, so the defaults stand."""
    service = SarvamSTTService(api_key="test-key")
    assert service.service_metadata_frame().user_turn_strategies is None

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

pytest.importorskip("sarvamai")

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


def test_sarvam_vad_signals_recommend_external_strategies():
    """With `vad_signals` on, Sarvam's boundaries are what drive turns."""
    service = SarvamSTTService(
        api_key="test-key",
        settings=SarvamSTTService.Settings(model="saaras:v3", vad_signals=True),
    )
    strategies = service.service_metadata_frame().user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)


def test_saaras_v4_is_accepted_with_language_and_mode():
    service = SarvamSTTService(
        api_key="test-key",
        mode="translate",
        settings=SarvamSTTService.Settings(
            model="saaras:v4",
            language=Language.HI_IN,
        ),
    )

    assert service._settings.model == "saaras:v4"
    assert service._settings.language == Language.HI_IN
    assert service._get_language_string() == "hi-IN"
    assert service._mode == "translate"


def test_saaras_v4_rejects_prompt():
    with pytest.raises(ValueError, match="does not support prompt"):
        SarvamSTTService(
            api_key="test-key",
            settings=SarvamSTTService.Settings(
                model="saaras:v4",
                prompt="medical vocabulary",
            ),
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "positive_speech_threshold",
        "negative_speech_threshold",
        "min_speech_frames",
        "first_turn_min_speech_frames",
        "negative_frames_count",
        "negative_frames_window",
        "start_speech_volume_threshold",
        "interrupt_min_speech_frames",
        "pre_speech_pad_frames",
        "num_initial_ignored_frames",
    ],
)
def test_saaras_v4_rejects_fine_grained_vad_parameters(parameter):
    with pytest.raises(ValueError, match=f"does not support {parameter}"):
        SarvamSTTService(
            api_key="test-key",
            settings=SarvamSTTService.Settings(
                model="saaras:v4",
                **{parameter: 1},
            ),
        )


def test_sarvam_without_vad_signals_recommends_no_strategies():
    """Without them Sarvam proposes no turns, so the defaults stand."""
    service = SarvamSTTService(api_key="test-key")
    assert service.service_metadata_frame().user_turn_strategies is None

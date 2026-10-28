#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the names TTS services perform a tagger's labels by."""

import pytest

from pipecat.adapters.tts.services.cartesia_adapter import CartesiaExpressionAdapter
from pipecat.adapters.tts.services.elevenlabs_adapter import ElevenLabsAudioTagExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import FishAudioS2ExpressionAdapter
from pipecat.adapters.tts.services.minimax_adapter import MiniMaxExpressionAdapter
from pipecat.expressive.expression import (
    TAGGER_DELIVERY,
    TAGGER_SOUNDS,
    ExpressionCapabilities,
    ExpressionTerm,
)
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.processors.tone.tagger._labels import LabelSpace

ADAPTERS = [
    CartesiaExpressionAdapter,
    ElevenLabsAudioTagExpressionAdapter,
    FishAudioS2ExpressionAdapter,
    MiniMaxExpressionAdapter,
]


def tagger_name(
    capabilities: ExpressionCapabilities, label: str, offered: list[str] | None = None
) -> str | None:
    """The name the tagger writes a label as, with a policy offering ``offered``."""
    policy = ExpressionPolicy(
        guidance=[ExpressionGuidance.BREATHE], delivery=offered, sounds=offered
    )
    space = LabelSpace.resolve(capabilities, policy)
    return {**space.deliveries, **space.sounds}.get(label)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_name_in_an_adapters_vocabulary_is_one_it_performs(adapter):
    capabilities = adapter().capabilities()

    assert set(capabilities.tagger_vocabulary) <= {*TAGGER_DELIVERY, *TAGGER_SOUNDS}
    for label, names in capabilities.tagger_vocabulary.items():
        for name in names:
            assert tagger_name(capabilities, label, offered=[name]) == name


def test_the_best_name_the_policy_offers_is_chosen():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("calm"), ExpressionTerm("reassuring")],
        tagger_vocabulary={"calm": ["calm", "reassuring"]},
    )

    assert tagger_name(capabilities, "calm") == "calm"
    assert tagger_name(capabilities, "calm", offered=["reassuring", "happy"]) == "reassuring"
    assert tagger_name(capabilities, "calm", offered=["happy"]) is None


def test_a_label_left_out_is_performed_by_its_own_name_when_the_service_declares_it():
    capabilities = ExpressionCapabilities(
        sounds=[ExpressionTerm("laughter")],
        delivery=[ExpressionTerm("calm")],
        tagger_vocabulary={"laugh": ["laughter"]},
    )

    assert tagger_name(capabilities, "calm") == "calm"
    assert tagger_name(capabilities, "laugh") == "laughter"
    assert tagger_name(capabilities, "sad") is None
    assert tagger_name(capabilities, "sigh") is None


def test_a_service_that_takes_free_text_also_takes_the_label_last():
    capabilities = ExpressionCapabilities(
        free_text_sounds=True, free_text_delivery=True, tagger_vocabulary={"warm": ["warmly"]}
    )

    assert tagger_name(capabilities, "grateful") == "grateful"
    assert tagger_name(capabilities, "warm") == "warmly"
    assert tagger_name(capabilities, "warm", offered=["warm"]) == "warm"


def test_a_name_the_service_doesnt_perform_is_passed_over():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("happy")], tagger_vocabulary={"warm": ["content", "happy"]}
    )

    assert tagger_name(capabilities, "warm") == "happy"

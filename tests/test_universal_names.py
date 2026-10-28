#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the universal expression names, and the names TTS services perform them by."""

import pytest

from pipecat.adapters.tts.services.cartesia_adapter import CartesiaExpressionAdapter
from pipecat.adapters.tts.services.elevenlabs_adapter import ElevenLabsAudioTagExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import FishAudioS2ExpressionAdapter
from pipecat.adapters.tts.services.minimax_adapter import MiniMaxExpressionAdapter
from pipecat.expressive.expression import (
    CORE_DELIVERY,
    CORE_SOUNDS,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
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

# Sounds services declare that aren't a voice, so they have no universal name.
NOT_A_VOICE = {
    "applause",
    "clapping",
    "gunshot",
    "explosion",
    "fart",
    "audience laughing",
    "background laughter",
    "crowd laughing",
}


def tagger_name(
    capabilities: ExpressionCapabilities, label: str, offered: list[str] | None = None
) -> str | None:
    """The name the tagger writes a label as, with a policy offering ``offered``."""
    policy = ExpressionPolicy(
        guidance=[ExpressionGuidance.BREATHE], delivery=offered, sounds=offered
    )
    space = LabelSpace.resolve(capabilities, policy)
    return {**space.deliveries, **space.sounds, **space.breaths}.get(label)


def test_a_name_is_either_a_delivery_or_a_sound():
    assert not set(ExpressionDelivery) & set(ExpressionSound)


def test_the_core_names_are_universal():
    assert set(CORE_DELIVERY) <= set(ExpressionDelivery)
    assert set(CORE_SOUNDS) <= set(ExpressionSound)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_a_service_names_universal_names_by_names_it_performs(adapter):
    capabilities = adapter().capabilities()

    assert set(capabilities.service_names) <= {*ExpressionDelivery, *ExpressionSound}
    for universal, names in capabilities.service_names.items():
        if universal in ExpressionDelivery:
            free_text, declared = capabilities.free_text_delivery, capabilities.delivery
            performed = capabilities.delivery_name(universal)
        else:
            free_text, declared = capabilities.free_text_sounds, capabilities.sounds
            performed = capabilities.sound_name(universal)
        assert performed in names
        if not free_text:
            assert set(names) & {term.name for term in declared}


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_voice_a_service_declares_is_reached_by_a_universal_name(adapter):
    capabilities = adapter().capabilities()
    reached = {name for names in capabilities.service_names.values() for name in names}
    universal = {*ExpressionDelivery, *ExpressionSound}

    for term in [*capabilities.delivery, *capabilities.sounds]:
        assert term.name in universal | reached | NOT_A_VOICE, term.name


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_core_name_the_tagger_writes_is_one_the_service_performs(adapter):
    capabilities = adapter().capabilities()

    for name in CORE_DELIVERY:
        assert tagger_name(capabilities, name) in (capabilities.delivery_name(name), None)
    for name in CORE_SOUNDS:
        assert tagger_name(capabilities, name) in (capabilities.sound_name(name), None)


@pytest.mark.parametrize(
    "adapter,names",
    [
        (CartesiaExpressionAdapter, {}),
        (
            ElevenLabsAudioTagExpressionAdapter,
            {
                "light breath": "quick breath",
                "deep breath": "deep breath",
                "exhale": "exhales softly",
            },
        ),
        (
            FishAudioS2ExpressionAdapter,
            {"light breath": "inhaling", "deep breath": "deep breath", "exhale": "exhaling"},
        ),
        (
            MiniMaxExpressionAdapter,
            {"light breath": "breath", "deep breath": "inhale", "exhale": "exhale"},
        ),
    ],
)
def test_each_kind_of_breath_has_a_name_where_the_service_breathes(adapter, names):
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE])

    assert LabelSpace.resolve(adapter().capabilities(), policy).breaths == names


@pytest.mark.parametrize(
    "offered,breaths",
    [
        (["deep breath"], {"deep breath": "deep breath"}),
        (["sighs", "inhales deeply"], {"deep breath": "inhales deeply"}),
        (["short breath", "exhales"], {"light breath": "short breath", "exhale": "exhales"}),
    ],
)
def test_a_policy_that_names_only_a_kind_still_breathes(offered, breaths):
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE], sounds=offered)

    space = LabelSpace.resolve(ElevenLabsAudioTagExpressionAdapter().capabilities(), policy)

    assert space.breaths == breaths
    assert space.sounds["breath"] == next(iter(breaths.values()))


def test_the_best_name_the_policy_offers_is_chosen():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("calm"), ExpressionTerm("reassuring")],
        service_names={ExpressionDelivery.CALM: ["calm", "reassuring"]},
    )

    assert tagger_name(capabilities, "calm") == "calm"
    assert tagger_name(capabilities, "calm", offered=["reassuring", "happy"]) == "reassuring"
    assert tagger_name(capabilities, "calm", offered=["happy"]) is None


def test_a_label_left_out_is_performed_by_its_own_name_when_the_service_declares_it():
    capabilities = ExpressionCapabilities(
        sounds=[ExpressionTerm("laughter")],
        delivery=[ExpressionTerm("calm")],
        service_names={ExpressionSound.LAUGH: ["laughter"]},
    )

    assert tagger_name(capabilities, "calm") == "calm"
    assert tagger_name(capabilities, "laugh") == "laughter"
    assert tagger_name(capabilities, "sad") is None
    assert tagger_name(capabilities, "sigh") is None


def test_a_service_that_takes_free_text_also_takes_the_label_last():
    capabilities = ExpressionCapabilities(
        free_text_sounds=True,
        free_text_delivery=True,
        service_names={ExpressionDelivery.WARM: ["warmly"]},
    )

    assert tagger_name(capabilities, "grateful") == "grateful"
    assert tagger_name(capabilities, "warm") == "warmly"
    assert tagger_name(capabilities, "warm", offered=["warm"]) == "warm"


def test_a_name_the_service_doesnt_perform_is_passed_over():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("happy")],
        service_names={ExpressionDelivery.WARM: ["content", "happy"]},
    )

    assert tagger_name(capabilities, "warm") == "happy"

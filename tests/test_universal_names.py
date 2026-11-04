#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the universal expression names, and the names TTS services perform them by."""

import pytest

from pipecat.adapters.tts.services.azure_adapter import AzureHDExpressionAdapter
from pipecat.adapters.tts.services.cartesia_adapter import CartesiaExpressionAdapter
from pipecat.adapters.tts.services.elevenlabs_adapter import ElevenLabsAudioTagExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import FishAudioS2ExpressionAdapter
from pipecat.adapters.tts.services.gemini_adapter import Gemini38ExpressionAdapter
from pipecat.adapters.tts.services.inworld_adapter import (
    InworldExpressionAdapter,
    InworldFlashExpressionAdapter,
)
from pipecat.adapters.tts.services.minimax_adapter import MiniMaxExpressionAdapter
from pipecat.adapters.tts.services.mistral_adapter import MistralExpressionAdapter
from pipecat.adapters.tts.services.soniox_adapter import SonioxExpressionAdapter
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
    AzureHDExpressionAdapter,
    CartesiaExpressionAdapter,
    ElevenLabsAudioTagExpressionAdapter,
    FishAudioS2ExpressionAdapter,
    Gemini38ExpressionAdapter,
    InworldExpressionAdapter,
    InworldFlashExpressionAdapter,
    MiniMaxExpressionAdapter,
    MistralExpressionAdapter,
    SonioxExpressionAdapter,
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

    assert set(capabilities.service_vocabulary) <= {*ExpressionDelivery, *ExpressionSound}
    for universal, names in capabilities.service_vocabulary.items():
        if universal in ExpressionDelivery:
            free_text, declared = capabilities.allows_free_text_delivery, capabilities.delivery
            performed = capabilities.delivery_name(universal)
        else:
            free_text, declared = capabilities.allows_free_text_sounds, capabilities.sounds
            performed = capabilities.sound_name(universal)
        assert performed in names
        if not free_text:
            assert set(names) & {term.name for term in declared}


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_voice_a_service_declares_is_reached_by_a_universal_name(adapter):
    capabilities = adapter().capabilities()
    reached = {name for names in capabilities.service_vocabulary.values() for name in names}
    universal = {*ExpressionDelivery, *ExpressionSound}

    for term in [*capabilities.delivery, *capabilities.sounds]:
        assert term.name in universal | reached | NOT_A_VOICE, term.name


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_the_tagger_writes_each_core_name_the_service_performs_as_itself(adapter):
    capabilities = adapter().capabilities()

    for name in CORE_DELIVERY:
        assert tagger_name(capabilities, name) == (
            name if capabilities.delivery_name(name) else None
        )
    for name in CORE_SOUNDS:
        assert tagger_name(capabilities, name) == (name if capabilities.sound_name(name) else None)


@pytest.mark.parametrize(
    "adapter,kinds",
    [
        (AzureHDExpressionAdapter, ["light breath"]),
        (CartesiaExpressionAdapter, []),
        (ElevenLabsAudioTagExpressionAdapter, ["light breath", "deep breath", "exhale"]),
        (FishAudioS2ExpressionAdapter, ["light breath", "deep breath", "exhale"]),
        (Gemini38ExpressionAdapter, ["light breath", "deep breath", "exhale"]),
        (InworldExpressionAdapter, ["light breath"]),
        (InworldFlashExpressionAdapter, ["light breath"]),
        (MiniMaxExpressionAdapter, ["light breath", "deep breath", "exhale"]),
        (MistralExpressionAdapter, ["light breath"]),
        (SonioxExpressionAdapter, ["exhale"]),
    ],
)
def test_each_kind_of_breath_the_service_performs_is_offered(adapter, kinds):
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE])

    space = LabelSpace.resolve(adapter().capabilities(), policy)

    assert space.breaths == {kind: kind for kind in kinds}


@pytest.mark.parametrize(
    "offered,kinds",
    [
        (["deep breath"], ["deep breath"]),
        (["sigh", "deep breath"], ["deep breath"]),
        (["light breath", "exhale"], ["light breath", "exhale"]),
    ],
)
def test_a_policy_that_names_only_a_kind_still_breathes(offered, kinds):
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE], sounds=offered)

    space = LabelSpace.resolve(ElevenLabsAudioTagExpressionAdapter().capabilities(), policy)

    assert space.breaths == {kind: kind for kind in kinds}
    assert space.sounds["breath"] == kinds[0]


def test_a_label_is_written_as_itself_where_the_service_performs_it():
    capabilities = ExpressionCapabilities(
        sounds=[ExpressionTerm("laughter")],
        delivery=[ExpressionTerm("happy")],
        service_vocabulary={
            ExpressionDelivery.WARM: ["content", "happy"],
            ExpressionSound.LAUGH: ["laughter"],
        },
    )

    assert tagger_name(capabilities, "warm") == "warm"
    assert tagger_name(capabilities, "laugh") == "laugh"
    assert tagger_name(capabilities, "sad") is None
    assert tagger_name(capabilities, "sigh") is None


def test_a_developers_name_for_a_label_is_written_where_the_policy_offers_it():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("calm"), ExpressionTerm("reassuring")]
    )
    vocabulary = {"calm": ["reassuring"]}

    def written(offered):
        policy = ExpressionPolicy(delivery=offered)
        return LabelSpace.resolve(capabilities, policy, vocabulary=vocabulary).deliveries.get(
            "calm"
        )

    assert written(None) == "reassuring"
    assert written(["calm"]) == "calm"
    assert written(["happy"]) is None

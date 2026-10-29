#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the instruction that teaches an LLM to write expression tags."""

import pytest
from pydantic import ValidationError

from pipecat.expressive.expression import (
    SPEED_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)
from pipecat.expressive.instructions import expression_instructions, expression_policy_warning
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionTerm
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy

FIXED = ExpressionCapabilities(
    sounds=[ExpressionTerm("laughter")],
    delivery=[ExpressionTerm("calm"), ExpressionTerm("happy"), ExpressionTerm("sad")],
    pause_range_ms=(100, 5000),
    service_vocabulary={ExpressionDelivery.WARM: ["happy"], ExpressionSound.LAUGH: ["laughter"]},
)

FREE_TEXT = ExpressionCapabilities(
    sounds=[ExpressionTerm("laughs"), ExpressionTerm("woo", "an excited whoop")],
    delivery=[ExpressionTerm("excited"), ExpressionTerm("whispers", "speaking in a whisper")],
    allows_free_text_sounds=True,
    allows_free_text_delivery=True,
    emphasis=True,
    service_vocabulary={ExpressionDelivery.WHISPERING: ["whispers"]},
)


def test_nothing_is_taught_for_a_service_that_performs_nothing():
    assert expression_instructions(None) is None
    assert expression_instructions(ExpressionCapabilities()) is None


def test_a_service_is_offered_the_core_universal_names_it_performs():
    instructions = expression_instructions(FIXED)

    assert (
        "Use only these deliveries, written exactly as listed: <<calm>>, <<warm>>, <<sad>>."
        in instructions
    )
    assert "Use only these sounds, written exactly as listed: [[laugh]]." in instructions


def test_a_service_that_performs_delivery_is_taught_spans():
    instructions = expression_instructions(FIXED)

    assert "- <<calm: A B 7>> speaks just the words after the colon, A B 7" in instructions
    assert "speaks just the words after the colon" not in expression_instructions(
        ExpressionCapabilities(
            sounds=[ExpressionTerm("laughter")],
            service_vocabulary={ExpressionSound.LAUGH: ["laughter"]},
        )
    )


def test_a_free_text_service_is_offered_universal_names_as_the_only_choices():
    instructions = expression_instructions(FREE_TEXT)

    assert (
        "Use only these deliveries, written exactly as listed: <<calm>>, <<sympathetic>>, "
        "<<apologetic>>, <<warm>>, <<excited>>, <<surprised>>, <<sad>>, <<grateful>>, "
        "<<curious>>, <<confident>>, <<slow>>, <<fast>>."
    ) in instructions
    assert (
        "Use only these sounds, written exactly as listed: [[laugh]], [[chuckle]], [[sigh]]."
        in instructions
    )
    assert "Others work too" not in instructions


def test_a_breath_is_offered_only_to_a_policy_that_asks_for_breaths():
    breathes = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE])

    assert "[[breath]]" not in expression_instructions(FREE_TEXT)
    assert "[[sigh]], [[breath]]." in expression_instructions(FREE_TEXT, breathes)


def test_only_the_controls_a_service_performs_are_taught():
    fixed = expression_instructions(FIXED)
    free_text = expression_instructions(FREE_TEXT)

    assert "^^word^^" not in fixed
    assert "[[pause 500]]" in fixed
    assert "^^word^^ stresses a word." in free_text
    assert "[[pause" not in free_text


def test_a_policy_narrows_a_fixed_list_in_its_own_order():
    policy = ExpressionPolicy(delivery=["sad", "relieved", "calm"], sounds=[])

    instructions = expression_instructions(FIXED, policy)

    assert "exactly as listed: <<sad>>, <<calm>>." in instructions
    assert "[[sound]]" not in instructions


def test_a_policy_names_any_delivery_for_a_free_text_service():
    policy = ExpressionPolicy(delivery=["whispering", "relieved"])

    instructions = expression_instructions(FREE_TEXT, policy)

    # A universal name is described as the service describes its own name for it.
    assert (
        "exactly as listed: <<whispering>> (speaking in a whisper), <<relieved>>." in instructions
    )


def test_nothing_is_taught_when_a_policy_offers_nothing_performable():
    capabilities = ExpressionCapabilities(delivery=[ExpressionTerm("calm")])
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE], delivery=["relieved"])

    assert expression_instructions(capabilities, policy) is None


def test_guidance_follows_the_tags_in_order():
    policy = ExpressionPolicy(
        guidance=[ExpressionGuidance.MATCH_STYLE, "Never laugh while discussing billing."]
    )

    instructions = expression_instructions(FIXED, policy)

    assert instructions.endswith(
        f"\n\n{ExpressionGuidance.MATCH_STYLE}\nNever laugh while discussing billing."
    )


def test_neutral_in_a_policy_is_taught_on_its_own_line():
    instructions = expression_instructions(FIXED, ExpressionPolicy(delivery=["neutral", "calm"]))

    assert "exactly as listed: <<calm>>." in instructions
    assert "<<neutral>> returns to your normal voice." in instructions


def test_a_service_that_performs_speed_is_offered_slow_with_when_to_use_it():
    capabilities = ExpressionCapabilities(delivery=[ExpressionTerm("calm"), *SPEED_DELIVERY])
    policy = ExpressionPolicy(
        guidance=[ExpressionGuidance.SLOW_FOR_DETAILS], delivery=["calm", "slow"]
    )

    instructions = expression_instructions(capabilities, policy)

    assert (
        "<<slow>> (to say something carefully, such as a number, a code or an address)"
        in instructions
    )
    # A span is shown slowing a detail, as details are said.
    assert "- <<slow: A B 7>> speaks just the words after the colon" in instructions
    assert instructions.endswith(ExpressionGuidance.SLOW_FOR_DETAILS)


def test_a_free_text_service_with_no_examples_is_offered_the_core_names():
    capabilities = ExpressionCapabilities(
        allows_free_text_sounds=True, allows_free_text_delivery=True
    )

    instructions = expression_instructions(capabilities)

    assert "written exactly as listed: <<calm>>, <<sympathetic>>," in instructions
    assert "- <<slow: A B 7>> speaks just the words after the colon" in instructions
    assert "written exactly as listed: [[laugh]], [[chuckle]], [[sigh]]." in instructions


def test_a_free_text_service_is_taught_none_of_a_kind_the_policy_names_none_of():
    capabilities = ExpressionCapabilities(
        allows_free_text_sounds=True, allows_free_text_delivery=True, emphasis=True
    )

    instructions = expression_instructions(capabilities, ExpressionPolicy(delivery=[], sounds=[]))

    assert (
        instructions
        == "You can direct how your words are spoken with tags:\n- ^^word^^ stresses a word."
    )


def test_a_policy_keeps_presets_apart_from_lines_with_the_same_text():
    policy = ExpressionPolicy(
        guidance=[ExpressionGuidance.MATCH_STYLE, str(ExpressionGuidance.MATCH_STYLE)]
    )

    preset, line = policy.guidance
    assert isinstance(preset, ExpressionGuidance)
    assert not isinstance(line, ExpressionGuidance)
    assert preset == line


@pytest.mark.parametrize("field", ["guidance", "delivery", "sounds"])
def test_a_policy_list_given_as_a_bare_string_is_rejected(field):
    # A bare string would otherwise be offered letter by letter.
    with pytest.raises(ValidationError):
        ExpressionPolicy(**{field: "calm"})


PERCEPTION = PerceptionCapabilities(
    signals=[
        PerceptionTerm("tone", labels=["positive", "neutral", "sad", "frustrated"]),
        PerceptionTerm("speech_rate", "Words per second."),
    ]
)


def test_perception_is_taught_with_an_example_and_its_signals():
    instructions = expression_instructions(None, perception=PERCEPTION)

    assert "in parentheses, such as ([tone positive=0.86, neutral=0.10]), each" in instructions
    assert "A reading below 0.5 is a guess: don't act on it." in instructions
    assert instructions.endswith(
        "- tone: positive, neutral, sad, frustrated.\n- speech_rate: Words per second."
    )


def test_perception_comes_before_the_tags_and_the_guidance_last():
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE])

    instructions = expression_instructions(FIXED, policy, perception=PERCEPTION)

    reading = instructions.index("([tone")
    tags = instructions.index("You can direct how your words are spoken")
    assert reading < tags < instructions.index(ExpressionGuidance.MATCH_STYLE)


def test_only_the_developer_guidance_is_kept_for_perception_without_tags():
    policy = ExpressionPolicy(
        guidance=[
            ExpressionGuidance.MATCH_STYLE,
            "If the user sounds frustrated, apologize once.",
            ExpressionGuidance.SLOW_FOR_DETAILS,
        ]
    )

    instructions = expression_instructions(None, policy, perception=PERCEPTION)

    assert instructions.endswith("\n\nIf the user sounds frustrated, apologize once.")
    assert ExpressionGuidance.MATCH_STYLE not in instructions
    assert "<<slow" not in instructions


def test_no_warning_without_a_policy_or_when_every_tag_is_performable():
    assert expression_policy_warning(FIXED, None) is None
    assert expression_policy_warning(None, None) is None
    assert expression_policy_warning(FIXED, ExpressionPolicy(delivery=["calm"])) is None
    # A free-text service performs any delivery or sound it's given.
    assert expression_policy_warning(FREE_TEXT, ExpressionPolicy(delivery=["relieved"])) is None


def test_a_warning_names_the_policy_tags_the_service_cant_perform():
    policy = ExpressionPolicy(delivery=["neutral", "calm", "relieved"], sounds=["sighs"])

    warning = expression_policy_warning(FIXED, policy)

    # Every service returns to its baseline, so neutral is never left out.
    assert warning == (
        "the TTS service can't perform <<relieved>>, [[sighs]] from the expression "
        "policy, so they aren't offered"
    )


def test_a_warning_says_when_the_service_performs_none_of_the_policy():
    capabilities = ExpressionCapabilities(delivery=[ExpressionTerm("calm")])
    expected = (
        "the TTS service performs none of the expression policy's tags, so they "
        "and its guidance presets aren't taught"
    )

    assert expression_policy_warning(capabilities, ExpressionPolicy(delivery=["relieved"])) == (
        expected
    )
    assert expression_policy_warning(None, ExpressionPolicy()) == expected


def test_a_developer_line_with_a_presets_text_is_kept_for_perception_without_tags():
    line = str(ExpressionGuidance.MATCH_STYLE)

    instructions = expression_instructions(
        None, ExpressionPolicy(guidance=[line]), perception=PERCEPTION
    )

    assert instructions.endswith(f"\n\n{line}")

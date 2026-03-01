#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Unit tests for Onairos personalization services."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.services.onairos import (
    OnairosPersonaInjector,
    OnairosUserData,
)
from pipecat.services.onairos.persona import MBTI_TYPES, _extract_score
from pipecat.tests.utils import run_test


class TestOnairosUserData(unittest.TestCase):
    """Tests for OnairosUserData model."""

    def test_default_values(self):
        """Test that OnairosUserData has correct defaults."""
        data = OnairosUserData()
        self.assertEqual(data.positive_traits, {})
        self.assertEqual(data.traits_to_improve, {})
        self.assertEqual(data.user_summary, "")
        self.assertEqual(data.top_traits_explanation, "")
        self.assertEqual(data.archetype, "")
        self.assertEqual(data.nudges, [])
        self.assertEqual(data.mbti, {})
        self.assertEqual(data.raw_data, {})

    def test_with_simple_scores(self):
        """Test OnairosUserData with simple numeric trait scores."""
        data = OnairosUserData(
            positive_traits={"AI Enthusiasm": 80, "Coffee Lover": 95},
            traits_to_improve={"Social Media Engagement": 35},
            user_summary="You enjoy technology and coffee.",
            archetype="Strategic Explorer",
            mbti={"INFJ": 0.627, "INTJ": 0.585},
        )
        self.assertEqual(data.positive_traits["AI Enthusiasm"], 80)
        self.assertEqual(data.traits_to_improve["Social Media Engagement"], 35)
        self.assertEqual(data.user_summary, "You enjoy technology and coffee.")
        self.assertEqual(data.archetype, "Strategic Explorer")
        self.assertEqual(data.mbti["INFJ"], 0.627)

    def test_with_detailed_trait_objects(self):
        """Test OnairosUserData with {score, emoji, evidence} trait objects."""
        data = OnairosUserData(
            positive_traits={
                "Stoic Wisdom Interest": {
                    "score": 80,
                    "emoji": "🏛️",
                    "evidence": "Frequently engages with philosophy content",
                }
            },
            nudges=[{"text": "Try journaling a decision you're mulling over"}],
        )
        trait = data.positive_traits["Stoic Wisdom Interest"]
        self.assertEqual(trait["score"], 80)
        self.assertEqual(trait["emoji"], "🏛️")
        self.assertEqual(len(data.nudges), 1)


class TestExtractScore(unittest.TestCase):
    """Tests for the _extract_score helper."""

    def test_extract_from_int(self):
        self.assertEqual(_extract_score(80), 80.0)

    def test_extract_from_float(self):
        self.assertEqual(_extract_score(75.5), 75.5)

    def test_extract_from_dict(self):
        self.assertEqual(_extract_score({"score": 90, "emoji": "🔥"}), 90.0)

    def test_extract_from_invalid(self):
        self.assertIsNone(_extract_score("not a score"))
        self.assertIsNone(_extract_score(None))
        self.assertIsNone(_extract_score({}))


class TestOnairosPersonaInjector(unittest.IsolatedAsyncioTestCase):
    """Tests for OnairosPersonaInjector service."""

    def test_initialization_without_credentials(self):
        """Test initialization without API credentials."""
        persona = OnairosPersonaInjector(user_id="test_user")
        self.assertEqual(persona.user_id, "test_user")
        self.assertFalse(persona.has_credentials)
        self.assertFalse(persona.has_data)

    def test_initialization_with_credentials(self):
        """Test initialization with API credentials."""
        persona = OnairosPersonaInjector(
            user_id="test_user",
            api_url="https://api.example.com",
            access_token="test_token",
        )
        self.assertTrue(persona.has_credentials)

    def test_initialization_with_user_data(self):
        """Test initialization with pre-loaded user data."""
        user_data = OnairosUserData(
            positive_traits={"Test Trait": 75},
            archetype="Explorer",
            mbti={"INFJ": 0.5},
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)
        self.assertTrue(persona.has_data)
        self.assertEqual(persona.user_data.positive_traits["Test Trait"], 75)
        self.assertEqual(persona.user_data.archetype, "Explorer")

    def test_set_api_credentials(self):
        """Test setting API credentials after initialization."""
        persona = OnairosPersonaInjector(user_id="test_user")
        self.assertFalse(persona.has_credentials)

        persona.set_api_credentials(
            api_url="https://api.example.com", access_token="test_token"
        )
        self.assertTrue(persona.has_credentials)

    def test_set_user_data(self):
        """Test setting user data directly."""
        persona = OnairosPersonaInjector(user_id="test_user")
        self.assertFalse(persona.has_data)

        user_data = OnairosUserData(positive_traits={"Test": 50})
        persona.set_user_data(user_data)
        self.assertTrue(persona.has_data)

    def test_set_user_data_from_dict_traits_only(self):
        """Test setting user data from a /traits-only response format."""
        persona = OnairosPersonaInjector(user_id="test_user")

        persona.set_user_data_from_dict(
            {
                "traits": {
                    "positive_traits": {"Trait1": 80, "Trait2": 60},
                    "traits_to_improve": {"Weakness1": 30},
                },
            }
        )

        self.assertTrue(persona.has_data)
        self.assertEqual(persona.user_data.positive_traits["Trait1"], 80)
        self.assertEqual(persona.user_data.traits_to_improve["Weakness1"], 30)

    def test_set_user_data_from_dict_combined_inference(self):
        """Test setting user data from a /combined-inference response format."""
        persona = OnairosPersonaInjector(user_id="test_user")

        output_scores = [0.5 + i * 0.01 for i in range(16)]
        persona.set_user_data_from_dict(
            {
                "InferenceResult": {"output": output_scores},
                "traits": {
                    "positive_traits": {"Analytical": 85},
                    "traits_to_improve": {"Patience": 40},
                },
            }
        )

        self.assertTrue(persona.has_data)
        self.assertEqual(persona.user_data.positive_traits["Analytical"], 85)
        self.assertEqual(len(persona.user_data.mbti), 16)
        self.assertAlmostEqual(persona.user_data.mbti["INTJ"], 0.5)

    def test_format_augmentation_with_all_data(self):
        """Test that augmentation is formatted correctly with all data types."""
        user_data = OnairosUserData(
            positive_traits={"AI Interest": 90, "Music Lover": 75},
            traits_to_improve={"Public Speaking": 35},
            user_summary="You are a tech enthusiast who loves hiking.",
            archetype="Strategic Explorer",
            mbti={"INFJ": 0.8, "INTJ": 0.7, "ENFJ": 0.6},
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        augmentation = persona._format_augmentation()

        self.assertIn("Positive Traits of User:", augmentation)
        self.assertIn("AI Interest", augmentation)
        self.assertIn("Areas to Improve:", augmentation)
        self.assertIn("Public Speaking", augmentation)
        self.assertIn("User Summary:", augmentation)
        self.assertIn("tech enthusiast", augmentation)
        self.assertIn("Archetype: Strategic Explorer", augmentation)
        self.assertIn("MBTI Alignment (Personalities User Likes):", augmentation)
        self.assertIn("INFJ", augmentation)
        self.assertIn("Critical Instruction:", augmentation)

    def test_format_augmentation_empty_data(self):
        """Test augmentation with empty user data."""
        persona = OnairosPersonaInjector(user_id="test_user")
        augmentation = persona._format_augmentation()
        self.assertEqual(augmentation, "")

    def test_format_augmentation_partial_data(self):
        """Test augmentation with only some data types."""
        user_data = OnairosUserData(
            positive_traits={"Test": 50},
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        augmentation = persona._format_augmentation()

        self.assertIn("Positive Traits of User:", augmentation)
        self.assertNotIn("Areas to Improve:", augmentation)
        self.assertNotIn("User Summary:", augmentation)
        self.assertNotIn("Archetype:", augmentation)
        self.assertNotIn("MBTI", augmentation)

    def test_format_augmentation_with_detailed_traits(self):
        """Test augmentation extracts scores from {score, emoji, evidence} objects."""
        user_data = OnairosUserData(
            positive_traits={
                "Stoic Wisdom": {"score": 80, "emoji": "🏛️", "evidence": "reads philosophy"},
                "AI Interest": {"score": 75, "emoji": "🤖", "evidence": "follows AI news"},
            },
        )
        persona = OnairosPersonaInjector(
            user_id="test_user",
            user_data=user_data,
            params=OnairosPersonaInjector.InputParams(
                include_traits_to_improve=False,
                include_user_summary=False,
                include_archetype=False,
                include_mbti=False,
                critical_instruction="",
            ),
        )

        augmentation = persona._format_augmentation()

        self.assertIn("Stoic Wisdom: 80", augmentation)
        self.assertIn("AI Interest: 75", augmentation)

    def test_top_mbti_limit(self):
        """Test that MBTI is limited to top N entries."""
        user_data = OnairosUserData(
            mbti={
                "INFJ": 0.9,
                "INTJ": 0.8,
                "ENFJ": 0.7,
                "INFP": 0.6,
                "ENTP": 0.5,
                "ISFJ": 0.4,
                "ISTJ": 0.3,
            }
        )
        persona = OnairosPersonaInjector(
            user_id="test_user",
            user_data=user_data,
            params=OnairosPersonaInjector.InputParams(
                include_personality_traits=False,
                include_traits_to_improve=False,
                include_user_summary=False,
                include_archetype=False,
                include_mbti=True,
                top_mbti_count=3,
                critical_instruction="",
            ),
        )

        augmentation = persona._format_augmentation()

        self.assertIn("INFJ", augmentation)
        self.assertIn("INTJ", augmentation)
        self.assertIn("ENFJ", augmentation)
        self.assertNotIn("ISFJ", augmentation)

    def test_mbti_score_mapping(self):
        """Test that MBTI inference output array is correctly mapped to type names."""
        scores = [0.5 + i * 0.01 for i in range(16)]
        mbti = dict(zip(MBTI_TYPES, scores))

        self.assertEqual(len(mbti), 16)
        self.assertAlmostEqual(mbti["INTJ"], 0.50)
        self.assertAlmostEqual(mbti["INTP"], 0.51)
        self.assertAlmostEqual(mbti["ESFP"], 0.65)

    async def test_process_frame_injects_context(self):
        """Test that process_frame injects Onairos context."""
        user_data = OnairosUserData(
            positive_traits={"Test Trait": 80},
            archetype="Explorer",
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        context = LLMContext([{"role": "system", "content": "Base prompt"}])
        frame = OpenAILLMContextFrame(context=context)

        frames_to_send = [frame]
        expected_down_frames = [OpenAILLMContextFrame]

        (received_down, _) = await run_test(
            persona,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        messages = received_down[0].context.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertIn("Positive Traits of User:", messages[1]["content"])

    async def test_context_only_injected_once(self):
        """Test that context is not injected multiple times."""
        user_data = OnairosUserData(positive_traits={"Test": 50})
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        context = LLMContext([{"role": "system", "content": "Base"}])

        frame1 = OpenAILLMContextFrame(context=context)
        await persona.process_frame(frame1, None)

        frame2 = OpenAILLMContextFrame(context=context)
        await persona.process_frame(frame2, None)

        messages = context.get_messages()
        self.assertEqual(len(messages), 2)


class TestOnairosAPIIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for Onairos API integration (mocked)."""

    async def test_fetch_combined_inference_response(self):
        """Test parsing a /combined-inference response with traits and MBTI."""
        mbti_scores = [0.584, 0.500, 0.550, 0.520,
                       0.627, 0.511, 0.580, 0.490,
                       0.460, 0.580, 0.470, 0.450,
                       0.430, 0.410, 0.400, 0.390]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "InferenceResult": {"output": mbti_scores},
                "traits": {
                    "positive_traits": {"Stoic Wisdom": 80, "AI Interest": 75},
                    "traits_to_improve": {"Social Media": 35},
                },
            }
        )

        async_cm_response = MagicMock()
        async_cm_response.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=async_cm_response)
        mock_session.closed = False

        persona = OnairosPersonaInjector(
            user_id="test_user",
            api_url="https://api2.onairos.uk/combined-inference",
            access_token="test_token",
        )
        persona._http_session = mock_session

        result = await persona.fetch_user_data()

        self.assertIsNotNone(result)
        self.assertEqual(result.positive_traits["Stoic Wisdom"], 80)
        self.assertEqual(result.traits_to_improve["Social Media"], 35)
        self.assertEqual(len(result.mbti), 16)
        self.assertAlmostEqual(result.mbti["INTJ"], 0.584)
        self.assertAlmostEqual(result.mbti["INFJ"], 0.627)

    async def test_fetch_traits_only_response(self):
        """Test parsing a /traits-only response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "success": True,
                "traits": {
                    "positive_traits": {"Analytical": 90},
                    "traits_to_improve": {"Patience": 40},
                },
            }
        )

        async_cm_response = MagicMock()
        async_cm_response.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=async_cm_response)
        mock_session.closed = False

        persona = OnairosPersonaInjector(
            user_id="test_user",
            api_url="https://api2.onairos.uk/traits-only",
            access_token="test_token",
        )
        persona._http_session = mock_session

        result = await persona.fetch_user_data()

        self.assertIsNotNone(result)
        self.assertEqual(result.positive_traits["Analytical"], 90)
        self.assertEqual(result.traits_to_improve["Patience"], 40)
        self.assertEqual(result.mbti, {})

    async def test_fetch_without_credentials_returns_none(self):
        """Test that fetch returns None without credentials."""
        persona = OnairosPersonaInjector(user_id="test_user")
        result = await persona.fetch_user_data()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

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
    OnairosContextAggregator,
    OnairosMemoryService,
    OnairosPersonaInjector,
    OnairosUserData,
)
from pipecat.tests.utils import run_test


class TestOnairosUserData(unittest.TestCase):
    """Tests for OnairosUserData model."""

    def test_default_values(self):
        """Test that OnairosUserData has correct defaults."""
        data = OnairosUserData()
        self.assertEqual(data.personality_traits, {})
        self.assertEqual(data.memory, "")
        self.assertEqual(data.mbti, {})
        self.assertEqual(data.raw_data, {})

    def test_with_data(self):
        """Test OnairosUserData with actual data."""
        data = OnairosUserData(
            personality_traits={"AI Enthusiasm": 80, "Coffee Lover": 95},
            memory="Reads Daily Stoic every morning",
            mbti={"INFJ": 0.627, "INTJ": 0.585},
        )
        self.assertEqual(data.personality_traits["AI Enthusiasm"], 80)
        self.assertEqual(data.memory, "Reads Daily Stoic every morning")
        self.assertEqual(data.mbti["INFJ"], 0.627)


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
            personality_traits={"Test Trait": 75},
            memory="Test memory",
            mbti={"INFJ": 0.5},
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)
        self.assertTrue(persona.has_data)
        self.assertEqual(persona.user_data.personality_traits["Test Trait"], 75)

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

        user_data = OnairosUserData(personality_traits={"Test": 50})
        persona.set_user_data(user_data)
        self.assertTrue(persona.has_data)

    def test_set_user_data_from_dict(self):
        """Test setting user data from a dictionary."""
        persona = OnairosPersonaInjector(user_id="test_user")

        persona.set_user_data_from_dict(
            {
                "personality_traits": {"Trait1": 80, "Trait2": 60},
                "memory": "User likes coffee",
                "mbti": {"ENFP": 0.7},
            }
        )

        self.assertTrue(persona.has_data)
        self.assertEqual(persona.user_data.personality_traits["Trait1"], 80)
        self.assertEqual(persona.user_data.memory, "User likes coffee")

    def test_format_augmentation_with_all_data(self):
        """Test that augmentation is formatted correctly with all data types."""
        user_data = OnairosUserData(
            personality_traits={"AI Interest": 90, "Music Lover": 75},
            memory="Works in tech, loves hiking",
            mbti={"INFJ": 0.8, "INTJ": 0.7, "ENFJ": 0.6},
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        augmentation = persona._format_augmentation()

        self.assertIn("Personality Traits of User:", augmentation)
        self.assertIn("AI Interest", augmentation)
        self.assertIn("Memory of User:", augmentation)
        self.assertIn("Works in tech", augmentation)
        self.assertIn("MBTI (Personalities User Likes):", augmentation)
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
            personality_traits={"Test": 50},
            memory="",  # Empty memory
            mbti={},  # Empty MBTI
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        augmentation = persona._format_augmentation()

        self.assertIn("Personality Traits of User:", augmentation)
        self.assertNotIn("Memory of User:", augmentation)
        self.assertNotIn("MBTI", augmentation)

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
                include_memory=False,
                include_mbti=True,
                top_mbti_count=3,
                critical_instruction="",
            ),
        )

        augmentation = persona._format_augmentation()

        # Should only have top 3
        self.assertIn("INFJ", augmentation)
        self.assertIn("INTJ", augmentation)
        self.assertIn("ENFJ", augmentation)
        self.assertNotIn("ISFJ", augmentation)  # Not in top 3

    async def test_process_frame_injects_context(self):
        """Test that process_frame injects Onairos context."""
        user_data = OnairosUserData(
            personality_traits={"Test Trait": 80},
            memory="Test memory",
        )
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        # Create a context with a base message
        context = LLMContext([{"role": "system", "content": "Base prompt"}])
        frame = OpenAILLMContextFrame(context=context)

        frames_to_send = [frame]
        expected_down_frames = [OpenAILLMContextFrame]

        (received_down, _) = await run_test(
            persona,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Check that context was augmented
        messages = received_down[0].context.get_messages()
        self.assertEqual(len(messages), 2)  # Base + augmentation
        self.assertIn("Personality Traits of User:", messages[1]["content"])

    async def test_context_only_injected_once(self):
        """Test that context is not injected multiple times."""
        user_data = OnairosUserData(personality_traits={"Test": 50})
        persona = OnairosPersonaInjector(user_id="test_user", user_data=user_data)

        context = LLMContext([{"role": "system", "content": "Base"}])

        # Process first frame
        frame1 = OpenAILLMContextFrame(context=context)
        await persona.process_frame(frame1, None)

        # Process second frame with same context
        frame2 = OpenAILLMContextFrame(context=context)
        await persona.process_frame(frame2, None)

        # Should still only have 2 messages (base + one augmentation)
        messages = context.get_messages()
        self.assertEqual(len(messages), 2)


class TestOnairosMemoryService(unittest.IsolatedAsyncioTestCase):
    """Tests for OnairosMemoryService."""

    def test_initialization_requires_user_id(self):
        """Test that user_id is required."""
        with self.assertRaises(ValueError):
            OnairosMemoryService(api_key="test_key")

    def test_initialization_with_user_id(self):
        """Test initialization with user_id."""
        memory = OnairosMemoryService(
            api_key="test_key", app_id="test_app", user_id="test_user"
        )
        self.assertEqual(memory.user_id, "test_user")


class TestOnairosContextAggregator(unittest.IsolatedAsyncioTestCase):
    """Tests for OnairosContextAggregator."""

    def test_initialization_requires_user_id(self):
        """Test that user_id is required."""
        with self.assertRaises(ValueError):
            OnairosContextAggregator(api_key="test_key")

    def test_initialization_with_user_id(self):
        """Test initialization with user_id."""
        context_agg = OnairosContextAggregator(
            api_key="test_key", user_id="test_user"
        )
        self.assertEqual(context_agg.user_id, "test_user")
        self.assertFalse(context_agg.is_connected)


class TestOnairosAPIIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for Onairos API integration (mocked)."""

    async def test_fetch_user_data_success(self):
        """Test successful API fetch."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "InferenceResult": {
                    "output": {
                        "personality_traits": {"Test": 80},
                        "memory": "Test memory",
                        "mbti": {"INFJ": 0.7},
                    }
                }
            }
        )

        # Create async context manager for response
        async_cm_response = MagicMock()
        async_cm_response.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm_response.__aexit__ = AsyncMock(return_value=None)

        # Create mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=async_cm_response)
        mock_session.closed = False

        persona = OnairosPersonaInjector(
            user_id="test_user",
            api_url="https://api.example.com/inference",
            access_token="test_token",
        )
        # Directly inject mock session
        persona._http_session = mock_session

        result = await persona.fetch_user_data()

        self.assertIsNotNone(result)
        self.assertEqual(result.personality_traits["Test"], 80)
        self.assertEqual(result.memory, "Test memory")

    async def test_fetch_without_credentials_returns_none(self):
        """Test that fetch returns None without credentials."""
        persona = OnairosPersonaInjector(user_id="test_user")
        result = await persona.fetch_user_data()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

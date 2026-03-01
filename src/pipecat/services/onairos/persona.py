#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos persona injection service for Pipecat.

This module provides persona injection capabilities that augment LLM prompts
with rich user context from Onairos, including personality traits, memories,
and MBTI compatibility scores.

Flow:
    1. Frontend: User connects via Onairos SDK
    2. Frontend onComplete: Returns { apiUrl, accessToken }
    3. Backend: Uses apiUrl + accessToken to call Onairos inference API
    4. Backend: Receives personality traits, memory, MBTI
    5. Backend: Augments LLM prompt with this data

Onairos API Base: https://api2.onairos.uk
Endpoints:
    - POST /inferenceNoProof - Get preferences/insights and memories
    - POST /traits - Get personality traits only

Example augmented prompt:
    [Base Prompt]

    Personality Traits of User:
    {"Stoic Wisdom Interest": 80, "AI Enthusiasm": 40}

    Memory of User:
    Reads Daily Stoic every morning. Prefers coffee shop meetups.

    MBTI (Personalities User Likes):
    INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580

    Critical Instruction:
    Always check context before asking. Complete task efficiently.
"""

import json
import os
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import Frame, LLMContextFrame, LLMMessagesFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class OnairosUserData(BaseModel):
    """Structured Onairos user data from inference API.

    Parameters:
        personality_traits: Dict of trait names to scores (0-100).
        memory: Textual memories about the user.
        mbti: Dict of MBTI types to compatibility scores (0-1).
        raw_data: Full raw response from Onairos API.
    """

    personality_traits: Dict[str, float] = Field(default_factory=dict)
    memory: str = Field(default="")
    mbti: Dict[str, float] = Field(default_factory=dict)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class OnairosPersonaInjector(FrameProcessor):
    """Augments LLM prompts with Onairos user context.

    This processor retrieves user data from Onairos using the API URL and
    access token returned by the frontend's onComplete callback, then
    injects it into the conversation context.

    Flow:
        Frontend onComplete → { apiUrl, accessToken }
                                    ↓
        Backend OnairosPersonaInjector.set_api_credentials(apiUrl, token)
                                    ↓
        Backend calls Onairos API → Gets traits, memory, MBTI
                                    ↓
        Augments LLM prompt with user context

    Event handlers available:

    - on_user_data_loaded: Called when Onairos data is successfully loaded
    - on_api_error: Called when API call fails

    Example::

        # Initialize injector
        persona = OnairosPersonaInjector(user_id="user_123")

        # When frontend sends onComplete data via WebSocket/RTVI:
        persona.set_api_credentials(
            api_url=data["apiUrl"],
            access_token=data["accessToken"]
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Onairos persona injector.

        Parameters:
            include_personality_traits: Include personality trait scores.
            include_memory: Include user memories.
            include_mbti: Include MBTI compatibility scores.
            critical_instruction: Instruction appended after context.
            top_mbti_count: Number of top MBTI types to include.
            top_traits_count: Number of top traits to include (0 = all).
        """

        include_personality_traits: bool = Field(default=True)
        include_memory: bool = Field(default=True)
        include_mbti: bool = Field(default=True)
        critical_instruction: str = Field(
            default="Always check context before asking. Use this information to personalize."
        )
        top_mbti_count: int = Field(default=5)
        top_traits_count: int = Field(default=0)  # 0 = include all

    def __init__(
        self,
        *,
        user_id: Optional[str] = None,
        api_url: Optional[str] = None,
        access_token: Optional[str] = None,
        user_data: Optional[OnairosUserData] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the Onairos persona injector.

        Args:
            user_id: The user ID (for logging/tracking).
            api_url: The API URL from frontend onComplete callback.
            access_token: The access token from frontend onComplete callback.
            user_data: Pre-loaded user data (skips API call if provided).
            params: Configuration parameters for persona injection.

        Note:
            api_url and access_token can be set later via set_api_credentials()
            when received from the frontend.
        """
        super().__init__()

        params = params or OnairosPersonaInjector.InputParams()

        self._user_id = user_id
        self._api_url = api_url
        self._access_token = access_token
        self._params = params
        self._user_data: Optional[OnairosUserData] = user_data
        self._context_injected = False
        self._http_session: Optional[aiohttp.ClientSession] = None

        if user_data:
            logger.info(f"OnairosPersonaInjector initialized with pre-loaded data")
        elif api_url and access_token:
            logger.info(f"OnairosPersonaInjector initialized with API credentials")
        else:
            logger.info(
                f"OnairosPersonaInjector initialized - waiting for API credentials "
                f"from frontend onComplete"
            )

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID."""
        return self._user_id

    @property
    def has_data(self) -> bool:
        """Check if user data has been loaded."""
        return self._user_data is not None

    @property
    def has_credentials(self) -> bool:
        """Check if API credentials are set."""
        return bool(self._api_url and self._access_token)

    @property
    def user_data(self) -> Optional[OnairosUserData]:
        """Get the loaded user data."""
        return self._user_data

    def set_api_credentials(self, api_url: str, access_token: str):
        """Set API credentials from frontend onComplete callback.

        Call this when you receive the onComplete data from the frontend.

        Args:
            api_url: The API URL returned by Onairos onComplete.
            access_token: The access token returned by Onairos onComplete.
        """
        self._api_url = api_url
        self._access_token = access_token
        self._context_injected = False  # Allow re-injection
        logger.info(f"Onairos API credentials set for user {self._user_id}")

    def set_user_data(self, data: OnairosUserData):
        """Set user data directly (if you already have it).

        Args:
            data: The OnairosUserData to use for augmentation.
        """
        self._user_data = data
        self._context_injected = False
        logger.info(f"User data set directly for {self._user_id}")

    def set_user_data_from_dict(self, data: Dict[str, Any]):
        """Set user data from a raw dictionary.

        Args:
            data: Raw Onairos API response or onComplete data.
        """
        self._user_data = OnairosUserData(
            personality_traits=data.get("personality_traits", data.get("traits", {})),
            memory=data.get("memory", data.get("memories", "")),
            mbti=data.get("mbti", data.get("mbti_compatibility", {})),
            raw_data=data,
        )
        self._context_injected = False
        logger.info(f"User data set from dict for {self._user_id}")

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def fetch_user_data(self) -> Optional[OnairosUserData]:
        """Fetch user data from Onairos API using the credentials from onComplete.

        The API endpoint is typically:
            POST https://api2.onairos.uk/inferenceNoProof
            or the apiUrl returned by onComplete

        Returns:
            OnairosUserData if successful, None otherwise.
        """
        if self._user_data:
            return self._user_data

        if not self._api_url or not self._access_token:
            logger.warning(
                "Cannot fetch Onairos data: API credentials not set. "
                "Call set_api_credentials() with data from frontend onComplete."
            )
            return None

        try:
            session = await self._get_http_session()

            # The Onairos API expects this format
            payload = {
                "accessToken": self._access_token,
                "inputData": []  # Empty for fetching existing user data
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._access_token}",
            }

            async with session.post(self._api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    raw = await response.json()

                    # Parse the inference result
                    inference_result = raw.get("InferenceResult", raw)
                    output = inference_result.get("output", inference_result)

                    self._user_data = OnairosUserData(
                        personality_traits=output.get(
                            "personality_traits",
                            output.get("traits", {})
                        ),
                        memory=output.get("memory", output.get("memories", "")),
                        mbti=output.get("mbti", output.get("mbti_compatibility", {})),
                        raw_data=raw,
                    )

                    logger.info(f"Fetched Onairos data for user {self._user_id}")
                    await self._call_event_handler("on_user_data_loaded", self._user_data)
                    return self._user_data
                else:
                    error_text = await response.text()
                    logger.warning(f"Onairos API error: {response.status} - {error_text}")
                    await self._call_event_handler(
                        "on_api_error", {"status": response.status, "error": error_text}
                    )
                    return None

        except Exception as e:
            logger.error(f"Error fetching Onairos data: {e}")
            await self._call_event_handler("on_api_error", {"error": str(e)})
            return None

    def _format_augmentation(self) -> str:
        """Format user data into the augmentation prompt.

        Returns:
            Formatted augmentation string to append to base prompt.
        """
        if not self._user_data:
            return ""

        sections = []

        # Personality Traits
        if self._params.include_personality_traits and self._user_data.personality_traits:
            traits = self._user_data.personality_traits
            if self._params.top_traits_count > 0:
                sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
                traits = dict(sorted_traits[: self._params.top_traits_count])
            sections.append(f"Personality Traits of User:\n{json.dumps(traits, indent=2)}")

        # Memory
        if self._params.include_memory and self._user_data.memory:
            sections.append(f"Memory of User:\n{self._user_data.memory}")

        # MBTI
        if self._params.include_mbti and self._user_data.mbti:
            mbti = self._user_data.mbti
            sorted_mbti = sorted(mbti.items(), key=lambda x: x[1], reverse=True)
            top_mbti = sorted_mbti[: self._params.top_mbti_count]
            mbti_str = ", ".join([f"{k}: {v:.3f}" for k, v in top_mbti])
            sections.append(f"MBTI (Personalities User Likes):\n{mbti_str}")

        # Critical Instruction
        if self._params.critical_instruction:
            sections.append(f"Critical Instruction:\n{self._params.critical_instruction}")

        return "\n\n".join(sections)

    def _inject_context(self, context: LLMContext | OpenAILLMContext):
        """Inject Onairos augmentation into the LLM context.

        Args:
            context: The LLM context to augment.
        """
        if self._context_injected:
            return

        augmentation = self._format_augmentation()
        if not augmentation:
            return

        context.add_message({"role": "system", "content": augmentation})
        self._context_injected = True
        logger.debug(f"Injected Onairos context for {self._user_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and inject Onairos context.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        context = None
        messages = None

        if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            context = LLMContext(messages)

        if context:
            try:
                if not self._context_injected and not self._user_data:
                    await self.fetch_user_data()

                if self._user_data:
                    self._inject_context(context)

                if messages is not None:
                    await self.push_frame(LLMMessagesFrame(context.get_messages()))
                else:
                    await self.push_frame(frame)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error injecting Onairos context: {str(e)}", exception=e
                )
                await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await super().cleanup()

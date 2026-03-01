#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos persona injection service for Pipecat.

This module provides persona injection capabilities that augment LLM prompts
with rich user context from Onairos, including personality traits, archetype,
user summary, and MBTI compatibility scores.

Flow:
    1. Frontend: User connects via Onairos SDK
    2. Frontend onComplete: Returns { apiUrl, accessToken }
    3. Backend: Uses apiUrl + accessToken to call Onairos inference API
    4. Backend: Receives personality traits, MBTI scores
    5. Backend: Augments LLM prompt with this data

Onairos API Base: https://api2.onairos.uk
Endpoints:
    - POST /inferenceNoProof - Get MBTI inference scores
    - POST /traits-only - Get personality traits only
    - POST /combined-inference - Get both inference and traits

Example augmented prompt:
    [Base Prompt]

    Positive Traits of User:
    Stoic Wisdom Interest: 80, AI Enthusiasm: 40

    Areas to Improve:
    Social Media Engagement: 35, Public Speaking Confidence: 40

    User Summary:
    You are drawn to deep philosophical thinking...

    Archetype: The Strategic Explorer

    MBTI Alignment (Personalities User Likes):
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

# Standard MBTI type labels used when mapping inference output array to types
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]


class OnairosUserData(BaseModel):
    """Structured Onairos user data from inference and traits APIs.

    Parameters:
        positive_traits: Dict of positive trait names to scores or detail objects.
            Values can be plain numbers (0-100) or dicts with score, emoji, evidence.
        traits_to_improve: Dict of improvement area names to scores or detail objects.
        user_summary: Multi-paragraph description of the user in 2nd person.
        top_traits_explanation: Explanation of reasoning behind top traits.
        archetype: Short archetype label (e.g. "Strategic Explorer").
        nudges: List of suggestion dicts, each with a "text" key.
        mbti: Dict of MBTI type codes to preference scores (0-1).
        raw_data: Full raw response from Onairos API.
    """

    positive_traits: Dict[str, Any] = Field(default_factory=dict)
    traits_to_improve: Dict[str, Any] = Field(default_factory=dict)
    user_summary: str = Field(default="")
    top_traits_explanation: str = Field(default="")
    archetype: str = Field(default="")
    nudges: List[Dict[str, str]] = Field(default_factory=list)
    mbti: Dict[str, float] = Field(default_factory=dict)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


def _extract_score(value: Any) -> Optional[float]:
    """Extract a numeric score from a trait value.

    Trait values may be plain numbers or dicts like {score, emoji, evidence}.

    Args:
        value: The trait value to extract a score from.

    Returns:
        The numeric score, or None if extraction fails.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        score = value.get("score")
        if score is not None:
            return float(score)
    return None


class OnairosPersonaInjector(FrameProcessor):
    """Augments LLM prompts with Onairos user context.

    This processor retrieves user data from Onairos using the API URL and
    access token returned by the frontend's onComplete callback, then
    injects it into the conversation context.

    Flow:
        Frontend onComplete -> { apiUrl, accessToken }
                                    |
        Backend OnairosPersonaInjector.set_api_credentials(apiUrl, token)
                                    |
        Backend calls Onairos API -> Gets traits, MBTI scores
                                    |
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
            include_personality_traits: Include positive personality trait scores.
            include_traits_to_improve: Include areas for improvement.
            include_user_summary: Include the multi-paragraph user summary.
            include_archetype: Include the user archetype label.
            include_nudges: Include actionable nudge suggestions.
            include_mbti: Include MBTI compatibility scores.
            critical_instruction: Instruction appended after context.
            top_mbti_count: Number of top MBTI types to include.
            top_traits_count: Number of top traits to include (0 = all).
        """

        include_personality_traits: bool = Field(default=True)
        include_traits_to_improve: bool = Field(default=True)
        include_user_summary: bool = Field(default=True)
        include_archetype: bool = Field(default=True)
        include_nudges: bool = Field(default=False)
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

        Handles multiple response formats from the Onairos backend:
        - /traits-only: {traits: {positive_traits, traits_to_improve}}
        - /combined-inference: {InferenceResult, traits, ...}
        - Full personality_traits structure with user_summary, archetype, etc.

        Args:
            data: Raw Onairos API response or pre-parsed data.
        """
        traits = data.get("traits", data.get("personality_traits", {}))
        positive_traits = {}
        traits_to_improve = {}

        if isinstance(traits, dict):
            positive_traits = traits.get("positive_traits", {})
            traits_to_improve = traits.get("traits_to_improve", {})

        mbti = data.get("mbti", {})
        if not mbti:
            inference = data.get("InferenceResult", {})
            output = inference.get("output", [])
            if isinstance(output, list) and len(output) == len(MBTI_TYPES):
                mbti = dict(zip(MBTI_TYPES, [float(v) for v in output]))

        self._user_data = OnairosUserData(
            positive_traits=positive_traits,
            traits_to_improve=traits_to_improve,
            user_summary=data.get("user_summary", ""),
            top_traits_explanation=data.get("top_traits_explanation", ""),
            archetype=data.get("archetype", ""),
            nudges=data.get("nudges", []),
            mbti=mbti,
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

        Handles the following real backend response formats:

        /combined-inference:
            {InferenceResult: {output: [...]}, traits: {positive_traits, traits_to_improve}}

        /inferenceNoProof:
            {InferenceResult: {output: [0.584, 0.500, ...]}}

        /traits-only:
            {success: true, traits: {positive_traits: {...}, traits_to_improve: {...}}}

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

            payload = {
                "accessToken": self._access_token,
                "inputData": []
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._access_token}",
            }

            async with session.post(self._api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    raw = await response.json()

                    # Parse traits from /traits-only or /combined-inference format
                    traits_data = raw.get("traits", {})
                    positive_traits = {}
                    traits_to_improve = {}
                    if isinstance(traits_data, dict):
                        positive_traits = traits_data.get("positive_traits", {})
                        traits_to_improve = traits_data.get("traits_to_improve", {})

                    # Parse MBTI from InferenceResult.output array
                    mbti = {}
                    inference_result = raw.get("InferenceResult", {})
                    if isinstance(inference_result, dict):
                        output = inference_result.get("output", [])
                        if isinstance(output, list) and len(output) == len(MBTI_TYPES):
                            mbti = dict(
                                zip(MBTI_TYPES, [float(v) for v in output])
                            )

                    # Parse full personality_traits structure if present
                    personality = raw.get("personality_traits", {})
                    user_summary = ""
                    top_traits_explanation = ""
                    archetype = ""
                    nudges = []

                    if isinstance(personality, dict):
                        if not positive_traits:
                            positive_traits = personality.get("positive_traits", {})
                        if not traits_to_improve:
                            traits_to_improve = personality.get("traits_to_improve", {})
                        user_summary = personality.get("user_summary", "")
                        top_traits_explanation = personality.get(
                            "top_traits_explanation", ""
                        )
                        archetype = personality.get("archetype", "")
                        nudges = personality.get("nudges", [])

                    self._user_data = OnairosUserData(
                        positive_traits=positive_traits,
                        traits_to_improve=traits_to_improve,
                        user_summary=user_summary,
                        top_traits_explanation=top_traits_explanation,
                        archetype=archetype,
                        nudges=nudges,
                        mbti=mbti,
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

        # Positive Traits
        if self._params.include_personality_traits and self._user_data.positive_traits:
            traits = self._user_data.positive_traits
            scores = {}
            for name, value in traits.items():
                score = _extract_score(value)
                if score is not None:
                    scores[name] = score

            if self._params.top_traits_count > 0:
                sorted_traits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                scores = dict(sorted_traits[: self._params.top_traits_count])

            if scores:
                traits_str = ", ".join([f"{k}: {int(v)}" for k, v in scores.items()])
                sections.append(f"Positive Traits of User:\n{traits_str}")

        # Areas to Improve
        if self._params.include_traits_to_improve and self._user_data.traits_to_improve:
            traits = self._user_data.traits_to_improve
            scores = {}
            for name, value in traits.items():
                score = _extract_score(value)
                if score is not None:
                    scores[name] = score

            if scores:
                traits_str = ", ".join([f"{k}: {int(v)}" for k, v in scores.items()])
                sections.append(f"Areas to Improve:\n{traits_str}")

        # User Summary
        if self._params.include_user_summary and self._user_data.user_summary:
            sections.append(f"User Summary:\n{self._user_data.user_summary}")

        # Archetype
        if self._params.include_archetype and self._user_data.archetype:
            sections.append(f"Archetype: {self._user_data.archetype}")

        # Nudges
        if self._params.include_nudges and self._user_data.nudges:
            nudge_texts = [n.get("text", "") for n in self._user_data.nudges if n.get("text")]
            if nudge_texts:
                nudges_str = "\n".join([f"- {t}" for t in nudge_texts])
                sections.append(f"Nudges:\n{nudges_str}")

        # MBTI
        if self._params.include_mbti and self._user_data.mbti:
            mbti = self._user_data.mbti
            sorted_mbti = sorted(mbti.items(), key=lambda x: x[1], reverse=True)
            top_mbti = sorted_mbti[: self._params.top_mbti_count]
            mbti_str = ", ".join([f"{k}: {v:.3f}" for k, v in top_mbti])
            sections.append(f"MBTI Alignment (Personalities User Likes):\n{mbti_str}")

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

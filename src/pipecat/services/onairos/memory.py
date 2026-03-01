#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos memory service integration for Pipecat.

This module provides a memory service that integrates with the Onairos platform
to store and retrieve conversational memories, enabling persistent context across
sessions and personalized user experiences.

Onairos API Documentation: https://onairos.uk/docs/api-endpoints/
"""

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


class OnairosMemoryService(FrameProcessor):
    """A memory service that integrates with the Onairos personalization platform.

    This service intercepts message frames in the pipeline, retrieves user persona
    data from Onairos, and enhances context with relevant information before passing
    it downstream. Provides persistent personalization across sessions.

    Onairos API Base URL: https://api.onairos.uk/v1

    Event handlers available:

    - on_persona_fetched: Called when persona data is successfully retrieved
    - on_context_enhanced: Called when context is enhanced with persona data

    Example::

        memory = OnairosMemoryService(
            api_key=os.getenv("ONAIROS_API_KEY"),
            app_id=os.getenv("ONAIROS_APP_ID"),
            user_id="user_123"
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Onairos memory service.

        Parameters:
            system_prompt: Prefix text for persona context messages.
            add_as_system_message: Whether to add persona as system messages.
            cache_persona: Whether to cache persona data to reduce API calls.
            include_preferences: Include user preferences in context.
            include_traits: Include user traits in context.
            include_interests: Include user interests in context.
        """

        system_prompt: str = Field(
            default="User Profile (use this to personalize your responses):\n"
        )
        add_as_system_message: bool = Field(default=True)
        cache_persona: bool = Field(default=True)
        include_preferences: bool = Field(default=True)
        include_traits: bool = Field(default=True)
        include_interests: bool = Field(default=True)

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        params: Optional[InputParams] = None,
        base_url: str = "https://api.onairos.uk/v1",
    ):
        """Initialize the Onairos memory service.

        Args:
            api_key: The Onairos API key (sk_live_* or sk_test_*).
                     Can also be set via ONAIROS_API_KEY env var.
            app_id: Your Onairos application ID.
                    Can also be set via ONAIROS_APP_ID env var.
            user_id: The user ID to retrieve persona for.
            params: Configuration parameters for memory retrieval.
            base_url: Onairos API base URL (default: https://api.onairos.uk/v1).

        Raises:
            ValueError: If user_id is not provided.
        """
        super().__init__()

        params = params or OnairosMemoryService.InputParams()

        if not user_id:
            raise ValueError("user_id must be provided for Onairos memory service")

        self._api_key = api_key or os.getenv("ONAIROS_API_KEY")
        self._app_id = app_id or os.getenv("ONAIROS_APP_ID")
        self._user_id = user_id
        self._base_url = base_url
        self._params = params
        self._cached_persona: Optional[Dict[str, Any]] = None
        self._persona_injected = False
        self._http_session: Optional[aiohttp.ClientSession] = None

        if not self._api_key:
            logger.warning(
                "ONAIROS_API_KEY not set. Onairos personalization will be disabled. "
                "Get your API key from https://dashboard.onairos.uk/"
            )

        logger.info(f"Initialized OnairosMemoryService for user_id={user_id}")

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID associated with this memory service."""
        return self._user_id

    @property
    def persona(self) -> Optional[Dict[str, Any]]:
        """Get the cached persona data."""
        return self._cached_persona

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._http_session

    async def _fetch_persona(self) -> Optional[Dict[str, Any]]:
        """Fetch user persona from Onairos API.

        API Endpoint: GET /personas/:userId

        Returns:
            User persona dictionary containing preferences, traits, and interests.
            Returns None if API key not set or request fails.
        """
        if not self._api_key:
            return None

        if self._params.cache_persona and self._cached_persona:
            return self._cached_persona

        try:
            session = await self._get_http_session()
            url = f"{self._base_url}/personas/{self._user_id}"

            async with session.get(url) as response:
                if response.status == 200:
                    persona = await response.json()
                    if self._params.cache_persona:
                        self._cached_persona = persona
                    logger.debug(f"Fetched persona for user {self._user_id}")
                    return persona
                elif response.status == 404:
                    logger.debug(f"No persona found for user {self._user_id}")
                    return None
                else:
                    logger.warning(
                        f"Onairos API error: {response.status} - {await response.text()}"
                    )
                    return None
        except Exception as e:
            logger.error(f"Error fetching persona from Onairos: {e}")
            return None

    def _format_persona_prompt(self, persona: Dict[str, Any]) -> str:
        """Format persona data into a prompt string for LLM context.

        Args:
            persona: The persona data from Onairos API.

        Returns:
            Formatted persona prompt string.
        """
        prompt = self._params.system_prompt

        if self._params.include_preferences:
            if preferences := persona.get("preferences", {}):
                if topics := preferences.get("contentTopics"):
                    prompt += f"- Interests: {', '.join(topics)}\n"
                if style := preferences.get("communicationStyle"):
                    prompt += f"- Communication style: {style}\n"
                if timezone := preferences.get("timezone"):
                    prompt += f"- Timezone: {timezone}\n"

        if self._params.include_traits:
            if traits := persona.get("traits", {}):
                if openness := traits.get("openness"):
                    prompt += f"- Openness level: {openness}\n"
                if interests := traits.get("interests"):
                    prompt += f"- Key interests: {', '.join(interests)}\n"

        return prompt

    def _enhance_context_with_persona(self, context: LLMContext | OpenAILLMContext):
        """Enhance the LLM context with persona information.

        Args:
            context: The LLM context to enhance with persona data.
        """
        if self._persona_injected:
            return

        if not self._cached_persona:
            return

        persona_prompt = self._format_persona_prompt(self._cached_persona)

        if self._params.add_as_system_message:
            context.add_message({"role": "system", "content": persona_prompt})
        else:
            context.add_message({"role": "user", "content": persona_prompt})

        self._persona_injected = True
        logger.debug(f"Enhanced context with persona for user {self._user_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, intercept context frames for persona integration.

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
                if not self._persona_injected:
                    await self._fetch_persona()
                    self._enhance_context_with_persona(context)

                if messages is not None:
                    await self.push_frame(LLMMessagesFrame(context.get_messages()))
                else:
                    await self.push_frame(frame)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error processing with Onairos: {str(e)}", exception=e
                )
                await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await super().cleanup()

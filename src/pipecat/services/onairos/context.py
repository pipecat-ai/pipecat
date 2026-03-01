#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos context aggregator for Pipecat.

This module provides enhanced context aggregation with persistent state,
enabling seamless onboarding flows and adaptive conversation management.

Onairos API Documentation: https://onairos.uk/docs/api-endpoints/
"""

import os
from typing import Any, Dict, Optional

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


class OnairosContextAggregator(FrameProcessor):
    """Enhances frame processing with Onairos connection management.

    This aggregator handles the Onairos connection flow, checking if users
    have connected their Onairos account and managing the onboarding experience
    for new users who haven't connected yet.

    Event handlers available:

    - on_user_connected: Called when user has an active Onairos connection
    - on_connection_needed: Called when user needs to connect Onairos
    - on_connection_created: Called when a new connection URL is generated

    Example::

        context = OnairosContextAggregator(
            api_key=os.getenv("ONAIROS_API_KEY"),
            app_id=os.getenv("ONAIROS_APP_ID"),
            user_id="user_123",
            redirect_url="https://yourapp.com/callback"
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Onairos context aggregator.

        Parameters:
            enable_connection_prompt: Whether to prompt users to connect Onairos.
            permissions: Onairos permissions to request from users.
            connection_prompt: Message shown when user needs to connect.
            welcome_connected_prompt: Message for users with Onairos connection.
        """

        enable_connection_prompt: bool = Field(default=True)
        permissions: list = Field(default=["preferences", "interests", "traits"])
        connection_prompt: str = Field(
            default=(
                "I can provide a much more personalized experience if you connect "
                "your Onairos profile. Would you like to do that?"
            )
        )
        welcome_connected_prompt: str = Field(
            default="Great to see you! I've loaded your preferences."
        )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        params: Optional[InputParams] = None,
        base_url: str = "https://api.onairos.uk/v1",
    ):
        """Initialize the Onairos context aggregator.

        Args:
            api_key: The Onairos API key (sk_live_* or sk_test_*).
            app_id: Your Onairos application ID.
            user_id: The user ID to track connections for.
            redirect_url: URL to redirect after Onairos connection.
            params: Configuration parameters for context aggregation.
            base_url: Onairos API base URL.

        Raises:
            ValueError: If user_id is not provided.
        """
        super().__init__()

        params = params or OnairosContextAggregator.InputParams()

        if not user_id:
            raise ValueError("user_id must be provided for Onairos context aggregator")

        self._api_key = api_key or os.getenv("ONAIROS_API_KEY")
        self._app_id = app_id or os.getenv("ONAIROS_APP_ID")
        self._user_id = user_id
        self._redirect_url = redirect_url
        self._base_url = base_url
        self._params = params
        self._connection_checked = False
        self._is_connected = False
        self._connection_url: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        if not self._api_key:
            logger.warning("ONAIROS_API_KEY not set. Connection management disabled.")

        logger.info(f"Initialized OnairosContextAggregator for user_id={user_id}")

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID associated with this context aggregator."""
        return self._user_id

    @property
    def is_connected(self) -> bool:
        """Check if user has connected their Onairos account."""
        return self._is_connected

    @property
    def connection_url(self) -> Optional[str]:
        """Get the connection URL for users to connect Onairos."""
        return self._connection_url

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

    async def check_connection(self) -> bool:
        """Check if user has an active Onairos connection.

        Returns:
            True if user is connected, False otherwise.
        """
        if not self._api_key:
            return False

        try:
            session = await self._get_http_session()
            url = f"{self._base_url}/personas/{self._user_id}"

            async with session.get(url) as response:
                self._is_connected = response.status == 200
                self._connection_checked = True

                if self._is_connected:
                    await self._call_event_handler("on_user_connected", self._user_id)
                else:
                    await self._call_event_handler("on_connection_needed", self._user_id)

                return self._is_connected
        except Exception as e:
            logger.error(f"Error checking Onairos connection: {e}")
            return False

    async def create_connection_url(self) -> Optional[str]:
        """Create a connection URL for the user to connect their Onairos account.

        API Endpoint: POST /connections

        Returns:
            Connection URL string or None if creation failed.
        """
        if not self._api_key or not self._redirect_url:
            return None

        try:
            session = await self._get_http_session()
            url = f"{self._base_url}/connections"

            payload = {
                "userId": self._user_id,
                "redirectUrl": self._redirect_url,
                "permissions": self._params.permissions,
                "metadata": {"source": "pipecat-voice-agent"},
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    self._connection_url = data.get("connectionUrl")
                    await self._call_event_handler(
                        "on_connection_created", self._connection_url
                    )
                    return self._connection_url
                else:
                    logger.warning(f"Failed to create connection: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error creating Onairos connection: {e}")
            return None

    def _apply_connection_context(self, context: LLMContext | OpenAILLMContext):
        """Apply connection-aware context for onboarding.

        Args:
            context: The LLM context to enhance.
        """
        if not self._params.enable_connection_prompt:
            return

        if self._is_connected:
            context.add_message(
                {"role": "system", "content": self._params.welcome_connected_prompt}
            )
        elif not self._is_connected and self._connection_checked:
            context.add_message(
                {"role": "system", "content": self._params.connection_prompt}
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage connection state.

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
                if not self._connection_checked:
                    await self.check_connection()
                    self._apply_connection_context(context)

                if messages is not None:
                    await self.push_frame(LLMMessagesFrame(context.get_messages()))
                else:
                    await self.push_frame(frame)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error processing context: {str(e)}", exception=e
                )
                await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await super().cleanup()

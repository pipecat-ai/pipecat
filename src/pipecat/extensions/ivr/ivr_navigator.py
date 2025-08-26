#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""IVR navigator for Pipecat."""

import json
from typing import Literal, Optional

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    KeypadEntry,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    OutputDTMFUrgentFrame,
    StartFrame,
    VADParamsUpdateFrame,
)
from pipecat.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService


class IVRProcessor(FrameProcessor):
    """IVR processor for Pipecat."""

    def __init__(
        self,
        *,
        ivr_prompt: str,
        conversation_prompt: str,
        ivr_response_delay: float,
        conversation_response_delay: float,
        initial_mode: Literal["ivr", "conversation"],
    ):
        """Initialize the IVR processor.

        Args:
            ivr_prompt: The prompt to use for IVR navigation.
            conversation_prompt: The prompt to use for conversation navigation.
            ivr_response_delay: The delay to wait before responding to the IVR.
            conversation_response_delay: The delay to wait before responding to the conversation.
            initial_mode: The initial mode to start in.
        """
        super().__init__()

        self._ivr_prompt = ivr_prompt
        self._conversation_prompt = conversation_prompt
        self._ivr_response_delay = ivr_response_delay
        self._conversation_response_delay = conversation_response_delay
        self._initial_mode = initial_mode

        # Response aggregation state
        self._processing_response = False
        self._response_buffer = ""
        self._is_first_token = False

        # Register the IVR stuck event
        self._register_event_handler("on_ivr_stuck")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and aggregate XML tag content.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Push the StartFrame right away
            await self.push_frame(frame, direction)

            # Update the context with the appropriate prompt based on the initial mode
            if self._initial_mode == "conversation":
                messages = [{"role": "system", "content": self._conversation_prompt}]
                llm_update_frame = LLMMessagesUpdateFrame(messages=messages)
                await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)
            else:
                messages = [{"role": "system", "content": self._ivr_prompt}]
                llm_update_frame = LLMMessagesUpdateFrame(messages=messages)
                await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

        elif isinstance(frame, LLMFullResponseStartFrame):
            # Begin aggregating a new LLM response
            self._processing_response = True
            self._response_buffer = ""
            self._is_first_token = True

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Complete response received - process any JSON actions
            if self._processing_response:
                await self._process_json_actions(self._response_buffer, direction)
            self._processing_response = False
            self._response_buffer = ""

        elif isinstance(frame, LLMTextFrame) and self._processing_response:
            # If first token doesn't suggest JSON, stop aggregating and stream normally
            if self._is_first_token and not self._contains_json_actions(frame.text):
                self._processing_response = False
                self._response_buffer = ""
                await self.push_frame(frame, direction)
            else:
                # Aggregating JSON actions
                self._response_buffer += frame.text

            self._is_first_token = False

        else:
            # Pass all non-LLM frames through
            await self.push_frame(frame, direction)

    def _contains_json_actions(self, text: str) -> bool:
        """Check if text contains or might contain JSON action objects.

        Looks for JSON structures like:
        {"action": "dtmf", "value": "4"} or {"action": "ivr", "status": "completed"}

        Only checks for JSON patterns if this is the first token in the response,
        and specifically looks for our expected JSON structure to avoid false positives.

        Args:
            text: The text to check for JSON patterns.

        Returns:
            True if this appears to be JSON with action structure.
        """
        # If we've already detected JSON in this response, keep buffering
        if "{" in self._response_buffer:
            return True

        # If first token contains '{', buffer the entire response
        # Trade-off: May buffer some non-action JSON, but processing validates it properly
        if self._is_first_token and "{" in text:
            return True

        return False

    async def _process_json_actions(self, response_text: str, direction: FrameDirection):
        """Process JSON action objects from the complete LLM response.

        Processes JSON like:
        {"action": "dtmf", "value": "4"}
        {"action": "ivr", "status": "completed"}

        Args:
            response_text: The complete aggregated response text from the LLM.
            direction: The direction of frame flow in the pipeline.
        """
        try:
            # Try to parse the entire response as JSON
            data = json.loads(response_text.strip())

            if isinstance(data, dict) and "action" in data:
                action = data["action"]

                if action == "dtmf" and "value" in data:
                    await self._handle_dtmf_action(data["value"], direction)

                elif action == "ivr" and "status" in data:
                    await self._handle_ivr_action(data["status"], direction)

        except (json.JSONDecodeError, KeyError) as e:
            # If not valid JSON action, treat as regular text
            logger.debug(f"Not a JSON action, treating as text: {e}")
            await self.push_frame(LLMTextFrame(response_text.strip()), direction)

    async def _handle_dtmf_action(self, value: str, direction: FrameDirection):
        """Handle DTMF action by creating and pushing DTMF frame.

        Args:
            value: The DTMF value to send (0-9, *, #).
            direction: The direction of frame flow in the pipeline.
        """
        logger.debug(f"DTMF detected: {value}")

        try:
            # Convert the value to a KeypadEntry
            keypad_entry = KeypadEntry(value)
            dtmf_frame = OutputDTMFUrgentFrame(button=keypad_entry)
            await self.push_frame(dtmf_frame, direction)
        except ValueError:
            logger.warning(f"Invalid DTMF value: {value}. Must be 0-9, *, or #")

    async def _handle_ivr_action(self, status: str, direction: FrameDirection):
        """Handle IVR status action.

        Args:
            status: The IVR status (detected, completed, stuck).
            direction: The direction of frame flow in the pipeline.
        """
        logger.debug(f"IVR status detected: {status}")

        if status == "detected":
            await self._handle_ivr_detected()
        elif status == "completed":
            await self._handle_ivr_completed()
        elif status == "stuck":
            await self._handle_ivr_stuck()
        else:
            logger.warning(f"Unknown IVR status: {status}")

    async def _handle_ivr_detected(self):
        """Handle IVR detection by switching to IVR mode.

        Only switches if initial_mode was "conversation".
        """
        # Only switch to IVR mode if we started in conversation mode
        if self._initial_mode == "conversation":
            logger.info("IVR detected - switching to IVR navigation mode")

            # Switch to IVR prompt
            messages = [{"role": "system", "content": self._ivr_prompt}]
            llm_update_frame = LLMMessagesUpdateFrame(messages=messages, run_llm=True)
            await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

            # Update VAD parameters for IVR response timing
            vad_params = VADParams(stop_secs=self._ivr_response_delay)
            vad_update_frame = VADParamsUpdateFrame(params=vad_params)
            await self.push_frame(vad_update_frame, FrameDirection.UPSTREAM)

        else:
            logger.debug("IVR detected but already in IVR mode - no action needed")

    async def _handle_ivr_completed(self):
        """Handle IVR completion by switching back to conversation mode.

        Updates the context to the conversation prompt and VAD parameters for conversation response timing.
        This action should be called when the IVR navigation is completed and before the conversation starts.
        The bot's first response will be in response to the user's first input.
        """
        logger.info("IVR navigation completed - switching back to conversation mode")

        # Switch back to conversation prompt
        messages = [{"role": "system", "content": self._conversation_prompt}]
        llm_update_frame = LLMMessagesUpdateFrame(messages=messages)  # run_llm=None (default)
        await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

        # Update VAD parameters for conversation response timing
        vad_params = VADParams(stop_secs=self._conversation_response_delay)
        vad_update_frame = VADParamsUpdateFrame(params=vad_params)
        await self.push_frame(vad_update_frame, FrameDirection.UPSTREAM)

    async def _handle_ivr_stuck(self):
        """Handle IVR stuck state by triggering event handler.

        Emits the on_ivr_stuck event for external handling of stuck scenarios.
        """
        logger.info("IVR navigation stuck - triggering event handler")
        await self._call_event_handler("on_ivr_stuck", self)


class IVRNavigator(Pipeline):
    """IVR navigator for Pipecat."""

    IVR_DETECTED_PROMPT = """IMPORTANT: When you detect an IVR system, respond ONLY with `{"action": "ivr", "status": "detected"}`."""

    IVR_NAVIGATION_PROMPT = """IMPORTANT: When you have completed the IVR navigation, respond ONLY with `{"action": "ivr", "status": "completed"}`. If you are stuck and cannot find a viable solution to navigating the IVR system, respond ONLY with `{"action": "ivr", "status": "stuck"}`.
    
    You are navigating an IVR system and will be given a list of options to choose from. When those options are keypresses, respond with the keypress option as JSON. For example, to respond with the number 1, you would respond ONLY with `{"action": "dtmf", "value": "1"}`.
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        ivr_prompt: str,
        conversation_prompt: str,
        ivr_response_delay: float = 2.0,
        conversation_response_delay: float = 0.8,
        initial_mode: Optional[Literal["ivr", "conversation"]] = "conversation",
    ):
        """Initialize the IVR navigator.

        Args:
            llm: The LLM service to use for navigation.
            ivr_prompt: The prompt to use for IVR navigation.
            conversation_prompt: The prompt to use for conversation navigation.
            ivr_response_delay: The delay to wait before responding to the IVR.
            conversation_response_delay: The delay to wait before responding to the conversation.
            initial_mode: The initial mode to start in. Default is "conversation".
        """
        self._llm = llm
        self._ivr_prompt = self.IVR_NAVIGATION_PROMPT + "\n\n" + ivr_prompt
        self._conversation_prompt = self.IVR_DETECTED_PROMPT + "\n\n" + conversation_prompt
        self._ivr_response_delay = ivr_response_delay
        self._conversation_response_delay = conversation_response_delay
        self._initial_mode = initial_mode

        self._ivr_processor = IVRProcessor(
            ivr_prompt=self._ivr_prompt,
            conversation_prompt=self._conversation_prompt,
            ivr_response_delay=self._ivr_response_delay,
            conversation_response_delay=self._conversation_response_delay,
            initial_mode=self._initial_mode,
        )

        super().__init__([self._llm, self._ivr_processor])

        # Register the IVR stuck event after super().__init__()
        self._register_event_handler("on_ivr_stuck")

    def add_event_handler(self, event_name: str, handler):
        """Add an event handler for IVR navigation events.

        Args:
            event_name: The name of the event to handle.
            handler: The function to call when the event occurs.
        """
        if event_name == "on_ivr_stuck":
            self._ivr_processor.add_event_handler(event_name, handler)
        else:
            super().add_event_handler(event_name, handler)

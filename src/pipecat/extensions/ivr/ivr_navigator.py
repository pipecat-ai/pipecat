#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""IVR navigator for Pipecat."""

from typing import Literal, Optional

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    KeypadEntry,
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    OutputDTMFUrgentFrame,
    StartFrame,
    VADParamsUpdateFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch, PatternPairAggregator


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

        # XML pattern aggregation
        self._aggregator = PatternPairAggregator()
        self._setup_xml_patterns()

        # Register the IVR stuck event
        self._register_event_handler("on_ivr_stuck")

    def _setup_xml_patterns(self):
        """Set up XML pattern detection and handlers."""
        # Register DTMF pattern
        self._aggregator.add_pattern_pair("dtmf", "<dtmf>", "</dtmf>", remove_match=True)
        self._aggregator.on_pattern_match("dtmf", self._handle_dtmf_pattern)

        # Register IVR pattern
        self._aggregator.add_pattern_pair("ivr", "<ivr>", "</ivr>", remove_match=True)
        self._aggregator.on_pattern_match("ivr", self._handle_ivr_pattern)

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
                # Set the conversation prompt and push it upstream
                messages = [{"role": "system", "content": self._conversation_prompt}]
                llm_update_frame = LLMMessagesUpdateFrame(messages=messages)
                await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

                # Set the VAD parameters delay and push it upstream
                params = VADParams(stop_secs=self._conversation_response_delay)
                vad_update_frame = VADParamsUpdateFrame(params=params)
                await self.push_frame(vad_update_frame, FrameDirection.UPSTREAM)
            else:
                # Set the IVR prompt and push it upstream
                messages = [{"role": "system", "content": self._ivr_prompt}]
                llm_update_frame = LLMMessagesUpdateFrame(messages=messages)
                await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

                # Set the VAD parameters delay and push it upstream
                params = VADParams(stop_secs=self._ivr_response_delay)
                vad_update_frame = VADParamsUpdateFrame(params=params)
                await self.push_frame(vad_update_frame, FrameDirection.UPSTREAM)

        elif isinstance(frame, LLMTextFrame):
            # Process text through the pattern aggregator
            result = await self._aggregator.aggregate(frame.text)
            if result:
                # Push aggregated text that doesn't contain XML patterns
                await self.push_frame(LLMTextFrame(result), direction)

        else:
            # Pass all non-LLM frames through
            await self.push_frame(frame, direction)

    async def _handle_dtmf_pattern(self, match: PatternMatch):
        """Handle DTMF XML pattern matches.

        Args:
            match: The pattern match containing DTMF content.
        """
        await self._handle_dtmf_action(match.content)

    async def _handle_ivr_pattern(self, match: PatternMatch):
        """Handle IVR XML pattern matches.

        Args:
            match: The pattern match containing IVR status content.
        """
        await self._handle_ivr_action(match.content)

    async def _handle_dtmf_action(self, value: str):
        """Handle DTMF action by creating and pushing DTMF frame.

        Args:
            value: The DTMF value to send (0-9, *, #).
        """
        logger.debug(f"DTMF detected: {value}")

        try:
            # Convert the value to a KeypadEntry
            keypad_entry = KeypadEntry(value)
            dtmf_frame = OutputDTMFUrgentFrame(button=keypad_entry)
            await self.push_frame(dtmf_frame, FrameDirection.DOWNSTREAM)
        except ValueError:
            logger.warning(f"Invalid DTMF value: {value}. Must be 0-9, *, or #")

    async def _handle_ivr_action(self, status: str):
        """Handle IVR status action.

        Args:
            status: The IVR status (detected, completed, stuck, wait).
        """
        logger.debug(f"IVR status detected: {status}")

        if status == "detected":
            await self._handle_ivr_detected()
        elif status == "completed":
            await self._handle_ivr_completed()
        elif status == "stuck":
            await self._handle_ivr_stuck()
        elif status == "wait":
            await self._handle_ivr_wait()
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

    async def _handle_ivr_wait(self):
        """Handle IVR wait state when transcription is incomplete.

        The LLM is indicating it needs more information to make a decision.
        This is a no-op since the system will continue to provide more transcription.
        """
        logger.debug("IVR waiting for more complete transcription")


class IVRNavigator(Pipeline):
    """IVR navigator for Pipecat."""

    IVR_DETECTED_PROMPT = (
        """IMPORTANT: When you detect an IVR system, respond ONLY with `<ivr>detected</ivr>`."""
    )

    IVR_NAVIGATION_BASE = """You are navigating an Interactive Voice Response (IVR) system to accomplish a specific goal. You receive text transcriptions of the IVR system's audio prompts and menu options.

YOUR NAVIGATION GOAL:
{goal}

NAVIGATION RULES:
1. When you see menu options with keypress instructions (e.g., "Press 1 for...", "Press 2 for..."), ONLY respond with a keypress if one of the options aligns with your navigation goal
2. If an option closely matches your goal, respond with: `<dtmf>NUMBER</dtmf>` (e.g., `<dtmf>1</dtmf>`)
3. For sequences of numbers (dates, account numbers, phone numbers), enter each digit separately: `<dtmf>1</dtmf><dtmf>2</dtmf><dtmf>3</dtmf>` for "123"
4. When the system asks for verbal responses (e.g., "Say Yes or No", "Please state your name", "What department?"), respond with natural language text ending with punctuation
5. If multiple options seem relevant, choose the most specific or direct path
6. If NO options are relevant to your goal, respond with `<ivr>wait</ivr>` - the system may present more options
7. If the transcription is incomplete or unclear, respond with `<ivr>wait</ivr>` to indicate you need more information

COMPLETION CRITERIA - Respond with `<ivr>completed</ivr>` when:
- You see "Please hold while I transfer you" or similar transfer language
- You see "You're being connected to..." or "Connecting you to..."
- The system says "One moment please" after selecting your final option
- The system indicates you've reached the target department/service
- You've successfully navigated to your goal and are being transferred to a human

WAIT CRITERIA - Respond with `<ivr>wait</ivr>` when:
- NONE of the presented options are relevant to your navigation goal
- The transcription appears to be cut off mid-sentence
- You can see partial menu options but the list seems incomplete
- The transcription is unclear or garbled
- You suspect there are more options that weren't captured in the transcription
- The system presents options for specific user types that don't apply to your goal

IMPORTANT: Do NOT feel pressured to select an option if none match your goal. Waiting is often the correct response when the IVR system is presenting partial menus or options intended for different user types.

STUCK CRITERIA - Respond with `<ivr>stuck</ivr>` when:
- You've been through the same menu options 3+ times without progress
- No available options relate to your goal after careful consideration
- You encounter an error message or "invalid selection" repeatedly
- The system asks for information you don't have (account numbers, PINs, etc.)
- You reach a dead end with no relevant options and no way back

STRATEGY TIPS:
- Look for keywords in menu options that match your goal
- Try general options like "Customer Service" or "Other Services" if specific options aren't available
- Pay attention to sub-menus. Sometimes the path requires multiple steps through different menu layers
- If you see "For all other inquiries, press..." that's often a good fallback option
- Remember that reaching your goal may require navigating through several menu levels
- Be patient - IVR systems often present options in waves, and waiting for the right option is better than selecting the wrong one

SEQUENCE INPUT EXAMPLES:
- For date of birth "01/15/1990": `<dtmf>0</dtmf><dtmf>1</dtmf><dtmf>1</dtmf><dtmf>5</dtmf><dtmf>1</dtmf><dtmf>9</dtmf><dtmf>9</dtmf><dtmf>0</dtmf>`
- For account number "12345": `<dtmf>1</dtmf><dtmf>2</dtmf><dtmf>3</dtmf><dtmf>4</dtmf><dtmf>5</dtmf>`
- For phone number last 4 digits "6789": `<dtmf>6</dtmf><dtmf>7</dtmf><dtmf>8</dtmf><dtmf>9</dtmf>`

VERBAL RESPONSE EXAMPLES:
- "Is your date of birth 01/15/1990? Say Yes or No" → "Yes."
- "Please state your first and last name" → "John Smith."
- "What department are you trying to reach?" → "Billing."
- "Are you calling about an existing order? Please say Yes or No" → "No."
- "Did I hear that correctly? Please say Yes or No" → "Yes."

Remember: Respond with `<dtmf>NUMBER</dtmf>` (single or multiple for sequences), `<ivr>completed</ivr>`, `<ivr>stuck</ivr>`, `<ivr>wait</ivr>`, OR natural language text when verbal responses are requested. No other response types.
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        ivr_prompt: str,
        conversation_prompt: str,
        ivr_response_delay: float = 2.5,
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
        self._ivr_prompt = self.IVR_NAVIGATION_BASE.format(goal=ivr_prompt)
        self._conversation_prompt = self.IVR_DETECTED_PROMPT + "\n\n" + conversation_prompt
        self._ivr_response_delay = ivr_response_delay
        self._conversation_response_delay = conversation_response_delay
        self._initial_mode = initial_mode

        print(self._ivr_prompt)

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

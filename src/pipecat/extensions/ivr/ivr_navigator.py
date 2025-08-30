#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Interactive Voice Response (IVR) navigation components.

This module provides classes for automated navigation of IVR phone systems
using LLM-based decision making and DTMF tone generation.
"""

from enum import Enum
from typing import List, Optional

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    OutputDTMFUrgentFrame,
    StartFrame,
    TextFrame,
    VADParamsUpdateFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch, PatternPairAggregator


class IVRStatus(Enum):
    """Enumeration of IVR navigation status values.

    These statuses are used to communicate the current state of IVR navigation
    between the LLM and the IVR processing system.
    """

    DETECTED = "detected"
    COMPLETED = "completed"
    STUCK = "stuck"
    WAIT = "wait"


class IVRProcessor(FrameProcessor):
    """Processes LLM responses for IVR navigation commands.

    Aggregates XML-tagged commands from LLM text streams and executes
    corresponding actions like DTMF tone generation and mode switching.

    Supported features:

    - DTMF command processing (`<dtmf>1</dtmf>`)
    - IVR state management (see IVRStatus enum: `<ivr>detected</ivr>`, `<ivr>completed</ivr>`, `<ivr>stuck</ivr>`, `<ivr>wait</ivr>`)
    - Automatic prompt and VAD parameter switching
    - Event emission via on_ivr_status_changed for detected, completed, and stuck states
    """

    def __init__(
        self,
        *,
        classifier_prompt: str,
        ivr_prompt: str,
        ivr_vad_params: Optional[VADParams] = None,
    ):
        """Initialize the IVR processor.

        Args:
            classifier_prompt: System prompt for classifying IVR or conversation.
            ivr_prompt: System prompt for IVR navigation mode.
            ivr_vad_params: VAD parameters for IVR navigation mode. If None, defaults to VADParams(stop_secs=2.0).
        """
        super().__init__()

        self._ivr_prompt = ivr_prompt
        self._ivr_vad_params = ivr_vad_params or VADParams(stop_secs=2.0)
        self._classifier_prompt = classifier_prompt

        # Store saved context messages
        self._saved_messages: List[dict] = []

        # XML pattern aggregation
        self._aggregator = PatternPairAggregator()
        self._setup_xml_patterns()

        # Register IVR events
        self._register_event_handler("on_conversation_detected")
        self._register_event_handler("on_ivr_status_changed")

    def update_saved_messages(self, messages: List[dict]) -> None:
        """Update the saved context messages.

        Sets the messages that are saved when switching between
        conversation and IVR navigation modes.

        Args:
            messages: List of message dictionaries to save.
        """
        self._saved_messages = messages

    def _get_conversation_history(self) -> List[dict]:
        """Get saved context messages without the system message.

        Returns:
            List of message dictionaries excluding the first system message.
        """
        return self._saved_messages[1:] if self._saved_messages else []

    def _setup_xml_patterns(self):
        """Set up XML pattern detection and handlers."""
        # Register DTMF pattern
        self._aggregator.add_pattern_pair("dtmf", "<dtmf>", "</dtmf>", remove_match=True)
        self._aggregator.on_pattern_match("dtmf", self._handle_dtmf_action)

        # Register mode pattern
        self._aggregator.add_pattern_pair("mode", "<mode>", "</mode>", remove_match=True)
        self._aggregator.on_pattern_match("mode", self._handle_mode_action)

        # Register IVR pattern
        self._aggregator.add_pattern_pair("ivr", "<ivr>", "</ivr>", remove_match=True)
        self._aggregator.on_pattern_match("ivr", self._handle_ivr_action)

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

            # Set the classifier prompt and push it upstream
            messages = [{"role": "system", "content": self._classifier_prompt}]
            llm_update_frame = LLMMessagesUpdateFrame(messages=messages)
            await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

        elif isinstance(frame, LLMTextFrame):
            # Process text through the pattern aggregator
            result = await self._aggregator.aggregate(frame.text)
            if result:
                # Push aggregated text that doesn't contain XML patterns
                await self.push_frame(LLMTextFrame(result), direction)

        else:
            await self.push_frame(frame, direction)

    async def _handle_dtmf_action(self, match: PatternMatch):
        """Handle DTMF action by creating and pushing DTMF frame.

        Args:
            match: The pattern match containing DTMF content.
        """
        value = match.content
        logger.debug(f"DTMF detected: {value}")

        try:
            # Convert the value to a KeypadEntry
            keypad_entry = KeypadEntry(value)
            dtmf_frame = OutputDTMFUrgentFrame(button=keypad_entry)
            await self.push_frame(dtmf_frame)
            # Push a TextFrame to add DTMF message to the context
            text_frame = TextFrame(text=f"<dtmf>{value}</dtmf>")
            text_frame.skip_tts = True
            await self.push_frame(text_frame)
        except ValueError:
            logger.warning(f"Invalid DTMF value: {value}. Must be 0-9, *, or #")

    async def _handle_ivr_action(self, match: PatternMatch):
        """Handle IVR status action.

        Args:
            match: The pattern match containing IVR status content.
        """
        status = match.content
        logger.trace(f"IVR status detected: {status}")

        # Convert string to enum, with validation
        try:
            ivr_status = IVRStatus(status)
        except ValueError:
            logger.warning(f"Unknown IVR status: {status}")
            return

        match ivr_status:
            case IVRStatus.DETECTED:
                await self._handle_ivr_detected()
            case IVRStatus.COMPLETED:
                await self._handle_ivr_completed()
            case IVRStatus.STUCK:
                await self._handle_ivr_stuck()
            case IVRStatus.WAIT:
                await self._handle_ivr_wait()

        # Push a TextFrame to add the IVR detected signal to the context
        ivr_text_frame = TextFrame(text=f"<ivr>{status}</ivr>")
        ivr_text_frame.skip_tts = True
        await self.push_frame(ivr_text_frame)

    async def _handle_mode_action(self, match: PatternMatch):
        """Handle mode action by switching to the appropriate mode.

        Args:
            match: The pattern match containing mode content.
        """
        mode = match.content
        logger.debug(f"Mode detected: {mode}")
        if mode == "conversation":
            await self._handle_conversation()
        elif mode == "ivr":
            await self._handle_ivr_detected()

        # No TextFrame is pushed for the mode selection, as the mode
        # selection conversation is ephemeral and the system message
        # is removed after the mode is detected.

    async def _handle_conversation(self):
        """Handle conversation mode by switching to conversation mode.

        Emit an on_conversation_detected event with saved conversation history.
        """
        logger.debug("Conversation detected - emitting on_conversation_detected event")

        # Extract conversation history for the event handler
        conversation_history = self._get_conversation_history()

        await self._call_event_handler("on_conversation_detected", conversation_history)

    async def _handle_ivr_detected(self):
        """Handle IVR detection by switching to IVR mode.

        Allows bidirectional switching for error recovery and complex IVR flows.
        Saves previous messages from the conversation context when available.
        """
        logger.debug("IVR detected - switching to IVR navigation mode")

        # Create new context with IVR system prompt and saved messages
        messages = [{"role": "system", "content": self._ivr_prompt}]

        # Add saved conversation history if available
        conversation_history = self._get_conversation_history()
        if conversation_history:
            messages.extend(conversation_history)

        # Push the messages upstream and run the LLM with the new context
        llm_update_frame = LLMMessagesUpdateFrame(messages=messages, run_llm=True)
        await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)

        # Update VAD parameters for IVR response timing
        vad_update_frame = VADParamsUpdateFrame(params=self._ivr_vad_params)
        await self.push_frame(vad_update_frame, FrameDirection.UPSTREAM)

        # Emit status changed event
        await self._call_event_handler("on_ivr_status_changed", IVRStatus.DETECTED)

    async def _handle_ivr_completed(self):
        """Handle IVR completion by triggering the status changed event.

        Emits on_ivr_status_changed with IVRStatus.COMPLETED.
        """
        logger.debug("IVR navigation completed - triggering status change event")

        await self._call_event_handler("on_ivr_status_changed", IVRStatus.COMPLETED)

    async def _handle_ivr_stuck(self):
        """Handle IVR stuck state by triggering the status changed event.

        Emits on_ivr_status_changed with IVRStatus.STUCK for external handling of stuck scenarios.
        """
        logger.debug("IVR navigation stuck - triggering status change event")

        await self._call_event_handler("on_ivr_status_changed", IVRStatus.STUCK)

    async def _handle_ivr_wait(self):
        """Handle IVR wait state when transcription is incomplete.

        The LLM is indicating it needs more information to make a decision.
        This is a no-op since the system will continue to provide more transcription.
        """
        logger.debug("IVR waiting for more complete transcription")


class IVRNavigator(Pipeline):
    """Pipeline for automated IVR system navigation.

    Orchestrates LLM-based IVR navigation by combining an LLM service with
    IVR processing capabilities. Starts with mode classification to classify input
    as conversation or IVR system.

    Navigation behavior:

    - Detects conversation vs IVR systems automatically
    - Navigates IVR menus using DTMF tones and verbal responses
    - Provides event hooks for mode classification and status changes (on_conversation_detected, on_ivr_status_changed)
    - Developers control conversation handling via on_conversation_detected event
    """

    CLASSIFIER_PROMPT = """You are an IVR detection classifier. Analyze the transcribed text to determine if it's an automated IVR system or a live human conversation.

IVR SYSTEM (respond `<mode>ivr</mode>`):
- Menu options: "Press 1 for billing", "Press 2 for technical support", "Press 0 to speak to an agent"
- Automated instructions: "Please enter your account number", "Say or press your selection", "Enter your phone number followed by the pound key"
- System prompts: "Thank you for calling [company]", "Your call is important to us", "Please hold while we connect you"
- Scripted introductions: "Welcome to [company] customer service", "For faster service, have your account number ready"
- Navigation phrases: "To return to the main menu", "Press star to repeat", "Say 'agent' or press 0"
- Hold messages: "Please continue to hold", "Your estimated wait time is", "Thank you for your patience"
- Carrier messages: "All circuits are busy", "Due to high call volume"

HUMAN CONVERSATION (respond `<mode>conversation</mode>`):
- Personal greetings: "Hello, this is Sarah", "Good morning, how can I help you?", "Customer service, this is Mike"
- Interactive responses: "Who am I speaking with?", "What can I do for you today?", "How are you calling about?"
- Natural speech patterns: hesitations, informal language, conversational flow
- Direct engagement: "I see you're calling about...", "Let me look that up for you", "Can you spell that for me?"
- Spontaneous responses: "Oh, I can help with that", "Sure, no problem", "Hmm, let me check"

RESPOND ONLY with either:
- `<mode>ivr</mode>` for IVR system
- `<mode>conversation</mode>` for human conversation"""

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

Remember: Respond with `<dtmf>NUMBER</dtmf>` (single or multiple for sequences), `<ivr>completed</ivr>`, `<ivr>stuck</ivr>`, `<ivr>wait</ivr>`, OR natural language text when verbal responses are requested. No other response types."""

    def __init__(
        self,
        *,
        llm: LLMService,
        ivr_prompt: str,
        ivr_vad_params: Optional[VADParams] = None,
    ):
        """Initialize the IVR navigator.

        Args:
            llm: LLM service for text generation and decision making.
            ivr_prompt: Navigation goal prompt integrated with IVR navigation instructions.
            ivr_vad_params: VAD parameters for IVR navigation mode. If None, defaults to VADParams(stop_secs=2.0).
        """
        self._llm = llm
        self._ivr_prompt = self.IVR_NAVIGATION_BASE.format(goal=ivr_prompt)
        self._ivr_vad_params = ivr_vad_params or VADParams(stop_secs=2.0)
        self._classifier_prompt = self.CLASSIFIER_PROMPT

        self._ivr_processor = IVRProcessor(
            classifier_prompt=self._classifier_prompt,
            ivr_prompt=self._ivr_prompt,
            ivr_vad_params=self._ivr_vad_params,
        )

        # Add the IVR processor to the pipeline
        super().__init__([self._llm, self._ivr_processor])

        # Register IVR events
        self._register_event_handler("on_conversation_detected")
        self._register_event_handler("on_ivr_status_changed")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames at the pipeline level to intercept context frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        if isinstance(frame, (OpenAILLMContextFrame, LLMContextFrame)):
            # Extract messages and pass to IVR processor
            all_messages = frame.context.get_messages()

            # Store messages in the IVR processor for mode switching
            self._ivr_processor.update_saved_messages(all_messages)

        # Let the pipeline handle normal frame processing
        await super().process_frame(frame, direction)

    def add_event_handler(self, event_name: str, handler):
        """Add event handler for IVR navigation events.

        Args:
            event_name: Event name ("on_conversation_detected", "on_ivr_status_changed").
            handler: Async function called when event occurs.
                    - on_conversation_detected: Receives IVRProcessor instance and conversation_history list
                    - on_ivr_status_changed: Receives IVRProcessor instance and IVRStatus enum value
        """
        if event_name in (
            "on_conversation_detected",
            "on_ivr_status_changed",
        ):
            self._ivr_processor.add_event_handler(event_name, handler)
        else:
            super().add_event_handler(event_name, handler)

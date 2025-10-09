#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail detection module for Pipecat.

This module provides voicemail detection capabilities using parallel pipeline
processing to classify incoming calls as either voicemail messages or live
conversations. It's specifically designed for outbound calling scenarios where
a bot needs to determine if a human answered or if the call went to voicemail.

Note:
    The voicemail module is optimized for text LLMs only.
"""

import asyncio
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StopFrame,
    SystemFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier


class NotifierGate(FrameProcessor):
    """Base gate processor that controls frame flow based on notifier signals.

    This base class provides common gate functionality for processors that need to
    start open and close permanently when a notifier signals. Subclasses define
    which frames are allowed through when the gate is closed.

    The gate starts open to allow initial processing and closes permanently once
    the notifier signals. This ensures controlled frame flow based on external
    decisions or events.
    """

    def __init__(self, notifier: BaseNotifier, task_name: str = "gate"):
        """Initialize the notifier gate.

        Args:
            notifier: Notifier that signals when the gate should close.
            task_name: Name for the notification waiting task (for debugging).
        """
        super().__init__()
        self._notifier = notifier
        self._task_name = task_name
        self._gate_opened = True
        self._gate_task: Optional[asyncio.Task] = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._gate_task = self.create_task(self._wait_for_notification())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on notifier signals.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Gate logic: open gate allows all frames, closed gate filters frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif isinstance(
            frame,
            (SystemFrame, EndFrame, StopFrame),
        ):
            await self.push_frame(frame, direction)

    async def _wait_for_notification(self):
        """Wait for notifier signal and close the gate.

        This method blocks until the notifier signals, then closes the gate
        permanently to change frame filtering behavior.
        """
        await self._notifier.wait()

        if self._gate_opened:
            self._gate_opened = False


class ClassifierGate(NotifierGate):
    """Gate processor that controls frame flow based on classification decisions.

    Inherits from NotifierGate and starts open to allow initial classification
    processing. Closes permanently once a classification decision is made
    (CONVERSATION or VOICEMAIL). This ensures the classifier only runs until a
    definitive decision is reached, preventing unnecessary LLM calls and maintaining
    system efficiency.

    When closed, only allows system frames and user speaking frames to continue.
    Speaking frames are needed for voicemail timing control, but not for conversation.
    """

    def __init__(self, gate_notifier: BaseNotifier, conversation_notifier: BaseNotifier):
        """Initialize the classifier gate.

        Args:
            gate_notifier: Notifier that signals when a classification decision has
                been made and the gate should close.
            conversation_notifier: Notifier that signals when conversation is detected.
        """
        super().__init__(gate_notifier, task_name="classifier_gate")
        self._conversation_notifier = conversation_notifier
        self._conversation_detected = False
        self._conversation_task: Optional[asyncio.Task] = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._conversation_task = self.create_task(self._wait_for_conversation())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on notifier signals.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await FrameProcessor.process_frame(self, frame, direction)

        # Gate logic: open gate allows all frames, closed gate filters frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            # Only allow speaking frames if conversation was NOT detected (i.e., voicemail case)
            # This prevents the UserContextAggregator from issuing a warning about no aggregation
            # to push.
            if not self._conversation_detected:
                await self.push_frame(frame, direction)
        elif isinstance(frame, (SystemFrame, EndFrame, StopFrame)):
            # Always allow system frames through
            # This includes the UserStartedSpeakingFrame and UserStoppedSpeakingFrame
            # which are used to detect voicemail timing.
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection notification and mark conversation detected."""
        await self._conversation_notifier.wait()
        self._conversation_detected = True


class ConversationGate(NotifierGate):
    """Gate processor that blocks conversation flow when voicemail is detected.

    Inherits from NotifierGate and starts open to allow normal conversation
    processing. Closes permanently when voicemail is detected to prevent the
    main conversation LLM from processing additional input after voicemail
    classification.

    When closed, only allows system frames and user speaking frames to continue.
    """

    def __init__(self, voicemail_notifier: BaseNotifier):
        """Initialize the conversation gate.

        Args:
            voicemail_notifier: Notifier that signals when voicemail has been
                detected and the conversation should be blocked.
        """
        super().__init__(voicemail_notifier, task_name="conversation_gate")


class ClassificationProcessor(FrameProcessor):
    """Processor that handles LLM classification responses and triggers events.

    This processor aggregates LLM text tokens into complete responses and analyzes
    them to determine if the call reached a voicemail system or a live person.
    It uses the LLM response frame delimiters (LLMFullResponseStartFrame and
    LLMFullResponseEndFrame) to ensure complete token aggregation regardless
    of how the LLM tokenizes the response words.

    The processor expects responses containing either "CONVERSATION" (indicating
    a human answered) or "VOICEMAIL" (indicating an automated system). Once a
    decision is made, it triggers the appropriate notifications and event handlers.

    For voicemail detection, the event handler timer starts immediately and is cancelled
    and restarted based on user speech patterns to ensure proper timing.
    """

    def __init__(
        self,
        *,
        gate_notifier: BaseNotifier,
        conversation_notifier: BaseNotifier,
        voicemail_notifier: BaseNotifier,
        voicemail_response_delay: float,
    ):
        """Initialize the voicemail processor.

        Args:
            gate_notifier: Notifier to signal the ClassifierGate about classification
                decisions so it can close and stop processing.
            conversation_notifier: Notifier to signal the TTSGate to release
                all gated TTS frames for normal conversation flow.
            voicemail_notifier: Notifier to signal the TTSGate to clear
                gated TTS frames since voicemail was detected.
            voicemail_response_delay: Delay in seconds after user stops speaking
                before triggering the voicemail event handler. This ensures the voicemail
                greeting or user message is complete before responding.
        """
        super().__init__()
        self._gate_notifier = gate_notifier
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._voicemail_response_delay = voicemail_response_delay

        # Register the voicemail detected event
        self._register_event_handler("on_voicemail_detected")

        # Aggregation state for collecting complete LLM responses
        self._processing_response = False
        self._response_buffer = ""
        self._decision_made = False

        # Voicemail timing state
        self._voicemail_detected = False
        self._voicemail_task: Optional[asyncio.Task] = None
        self._voicemail_event = asyncio.Event()
        self._voicemail_event.set()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._voicemail_task = self.create_task(self._delayed_voicemail_handler())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle LLM classification responses.

        This method implements a state machine for aggregating LLM responses:
        1. LLMFullResponseStartFrame: Begin collecting tokens
        2. LLMTextFrame: Accumulate text tokens into buffer
        3. LLMFullResponseEndFrame: Process complete response and make decision
        4. UserStartedSpeakingFrame/UserStoppedSpeakingFrame: Manage voicemail timing

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Begin aggregating a new LLM response
            self._processing_response = True
            self._response_buffer = ""

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Complete response received - make classification decision
            if self._processing_response and not self._decision_made:
                await self._process_classification(self._response_buffer.strip())
            self._processing_response = False
            self._response_buffer = ""

        elif isinstance(frame, LLMTextFrame) and self._processing_response:
            # Accumulate text tokens from the streaming LLM response
            self._response_buffer += frame.text

        elif isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking - set the voicemail event
            if self._voicemail_detected:
                self._voicemail_event.set()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # User stopped speaking - clear the voicemail event
            if self._voicemail_detected:
                self._voicemail_event.clear()

        else:
            # Pass all non-LLM frames through
            # Blocking LLM frames prevents interference with the downstream LLM
            await self.push_frame(frame, direction)

    async def _process_classification(self, full_response: str):
        """Process the complete LLM classification response and trigger actions.

        Analyzes the aggregated response text to determine if it contains
        "CONVERSATION" or "VOICEMAIL" and triggers the appropriate notifications
        and callbacks based on the classification result.

        Args:
            full_response: The complete aggregated response text from the LLM.
        """
        if self._decision_made:
            return

        response = full_response.upper()
        logger.debug(f"{self}: Classifying response: '{full_response}'")

        if "CONVERSATION" in response:
            # Human answered - continue normal conversation flow
            self._decision_made = True
            logger.info(f"{self}: CONVERSATION detected")
            await self._gate_notifier.notify()  # Close the classifier gate
            await self._conversation_notifier.notify()  # Release buffered TTS frames

        elif "VOICEMAIL" in response:
            # Voicemail detected - trigger voicemail handling
            self._decision_made = True
            self._voicemail_detected = True
            logger.info(f"{self}: VOICEMAIL detected")
            await self._gate_notifier.notify()  # Close the classifier gate
            await self._voicemail_notifier.notify()  # Clear buffered TTS frames

            # Interrupt the current pipeline to stop any ongoing processing
            await self.push_interruption_task_frame_and_wait()

            # Set the voicemail event to trigger the voicemail handler
            self._voicemail_event.clear()

        else:
            # This can happen if the LLM is interrupted before completing the response
            logger.debug(f"{self}: No classification found: '{full_response}'")

    async def _delayed_voicemail_handler(self):
        """Execute the voicemail event handler after the configured delay.

        This method waits for the specified delay period, then triggers the
        developer's voicemail event handler. The timer can be cancelled and restarted
        based on user speech patterns to ensure proper timing.
        """
        while True:
            try:
                await asyncio.wait_for(
                    self._voicemail_event.wait(), timeout=self._voicemail_response_delay
                )
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                await self._call_event_handler("on_voicemail_detected")
                break


class TTSGate(FrameProcessor):
    """Gates TTS frames until voicemail classification decision is made.

    This processor holds TTS output frames in a gate while the voicemail
    classification is in progress. This prevents audio from being played
    to the caller before determining if they're human or a voicemail system.

    The gate operates in two modes based on the classification result:

    - CONVERSATION: Opens the gate to release all held frames for normal dialogue
    - VOICEMAIL: Clears held frames since they're not needed for voicemail

    The gating only applies to TTS-related frames (TTSTextFrame, TTSAudioRawFrame).
    All other frames pass through immediately to maintain proper pipeline flow.
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the TTS gate.

        Args:
            conversation_notifier: Notifier that signals when a conversation is
                detected and gated frames should be released for playback.
            voicemail_notifier: Notifier that signals when voicemail is detected
                and gated frames should be cleared (not played).
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._frame_buffer: List[tuple[Frame, FrameDirection]] = []
        self._gating_active = True
        self._conversation_task: Optional[asyncio.Task] = None
        self._voicemail_task: Optional[asyncio.Task] = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        self._conversation_task = self.create_task(self._wait_for_conversation())
        self._voicemail_task = self.create_task(self._wait_for_voicemail())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle gating logic based on frame type.

        TTS frames are gated while classification is active. All other frames
        pass through immediately. The gating state is controlled by the
        classification notifications.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Core gating logic: hold TTS frames, pass everything else through
        if self._gating_active and isinstance(
            frame, (TTSStartedFrame, TTSStoppedFrame, TTSTextFrame, TTSAudioRawFrame)
        ):
            # Gate TTS frames while waiting for classification decision
            self._frame_buffer.append((frame, direction))
        else:
            # Pass through all non-TTS frames immediately
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection notification and release gated frames.

        When a conversation is detected, all gated TTS frames are released
        in order to continue normal dialogue flow. This allows the bot to
        respond naturally to the human caller.
        """
        await self._conversation_notifier.wait()

        # Release all gated frames in original order
        self._gating_active = False
        for frame, direction in self._frame_buffer:
            await self.push_frame(frame, direction)
        self._frame_buffer.clear()

    async def _wait_for_voicemail(self):
        """Wait for voicemail detection notification and clear gated frames.

        When voicemail is detected, all gated TTS frames are discarded
        since they were intended for human conversation and are not appropriate
        for voicemail systems. The developer event handlers will handle voicemail-
        specific audio output.
        """
        await self._voicemail_notifier.wait()

        # Clear gated frames without playing them
        self._gating_active = False
        self._frame_buffer.clear()


class VoicemailDetector(ParallelPipeline):
    """Parallel pipeline for detecting voicemail vs. live conversation in outbound calls.

    This detector uses a parallel pipeline architecture to perform real-time
    classification of outbound phone calls without interrupting the conversation
    flow. It determines whether a human answered the phone or if the call went
    to a voicemail system.

    Architecture:

    - Conversation branch: Empty pipeline that allows normal frame flow
    - Classification branch: Contains the LLM classifier and decision logic

    The system uses a gate mechanism to control when classification runs and
    a gating system to prevent TTS output until classification is complete.
    Once a decision is made, the appropriate action is taken:

    - CONVERSATION: Continue normal bot dialogue
    - VOICEMAIL: Trigger developer event handler for custom voicemail handling

    Example::

        classification_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        detector = VoicemailDetector(llm=classification_llm)

        @detector.event_handler("on_voicemail_detected")
        async def handle_voicemail(processor):
            await processor.push_frame(TTSSpeakFrame("Please leave a message."))

        pipeline = Pipeline([
            transport.input(),
            stt,
            detector.detector(),          # Classification
            context_aggregator.user(),
            llm,
            tts,
            detector.gate(),              # TTS gating
            transport.output(),
            context_aggregator.assistant(),
        ])

        # For custom prompts, append the required response instruction:
        custom_prompt = "Your custom classification logic here. " + VoicemailDetector.CLASSIFIER_RESPONSE_INSTRUCTION

    Events:
        on_voicemail_detected: Triggered when voicemail is detected after the configured
            delay. The event handler receives one argument: the ClassificationProcessor
            instance which can be used to push frames.

    Constants:
        CLASSIFIER_RESPONSE_INSTRUCTION: The exact text that must be included in custom
            system prompts to ensure proper classification functionality.
    """

    CLASSIFIER_RESPONSE_INSTRUCTION = 'Respond with ONLY "CONVERSATION" if a person answered, or "VOICEMAIL" if it\'s voicemail/recording.'

    DEFAULT_SYSTEM_PROMPT = (
        """You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text.

HUMAN ANSWERED - LIVE CONVERSATION (respond "CONVERSATION"):
- Personal greetings: "Hello?", "Hi", "Yeah?", "John speaking"
- Interactive responses: "Who is this?", "What do you want?", "Can I help you?"
- Conversational tone expecting back-and-forth dialogue
- Questions directed at the caller: "Hello? Anyone there?"
- Informal responses: "Yep", "What's up?", "Speaking"
- Natural, spontaneous speech patterns
- Immediate acknowledgment of the call

VOICEMAIL SYSTEM (respond "VOICEMAIL"):
- Automated voicemail greetings: "Hi, you've reached [name], please leave a message"
- Phone carrier messages: "The number you have dialed is not in service", "Please leave a message", "All circuits are busy"
- Professional voicemail: "This is [name], I'm not available right now"
- Instructions about leaving messages: "leave a message", "leave your name and number"
- References to callback or messaging: "call me back", "I'll get back to you"
- Carrier system messages: "mailbox is full", "has not been set up"
- Business hours messages: "our office is currently closed"

"""
        + CLASSIFIER_RESPONSE_INSTRUCTION
    )

    def __init__(
        self,
        *,
        llm: LLMService,
        voicemail_response_delay: float = 2.0,
        custom_system_prompt: Optional[str] = None,
    ):
        """Initialize the voicemail detector with classification and buffering components.

        Args:
            llm: LLM service used for voicemail vs conversation classification.
                Should be fast and reliable for real-time classification.
            voicemail_response_delay: Delay in seconds after user stops speaking
                before triggering the voicemail event handler. This allows voicemail
                responses to be played back after a short delay to ensure the response
                occurs during the voicemail recording. Default is 2.0 seconds.
            custom_system_prompt: Optional custom system prompt for classification. If None,
                uses the default prompt optimized for outbound calling scenarios.
                Custom prompts should instruct the LLM to respond with exactly
                "CONVERSATION" or "VOICEMAIL" for proper detection functionality.
        """
        self._classifier_llm = llm
        self._prompt = (
            custom_system_prompt if custom_system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        )
        self._voicemail_response_delay = voicemail_response_delay

        # Validate custom prompts to ensure they work with the detection logic
        if custom_system_prompt is not None:
            self._validate_prompt(custom_system_prompt)

        # Set up the LLM context with the classification prompt
        self._messages = [
            {
                "role": "system",
                "content": self._prompt,
            },
        ]

        # Create the LLM context and aggregators for conversation management
        self._context = LLMContext(self._messages)
        self._context_aggregator = LLMContextAggregatorPair(self._context)

        # Create notification system for coordinating between components
        self._gate_notifier = EventNotifier()  # Signals classification completion
        self._conversation_notifier = EventNotifier()  # Signals conversation detected
        self._voicemail_notifier = EventNotifier()  # Signals voicemail detected

        # Create the processor components
        self._classifier_gate = ClassifierGate(self._gate_notifier, self._conversation_notifier)
        self._conversation_gate = ConversationGate(self._voicemail_notifier)
        self._classification_processor = ClassificationProcessor(
            gate_notifier=self._gate_notifier,
            conversation_notifier=self._conversation_notifier,
            voicemail_notifier=self._voicemail_notifier,
            voicemail_response_delay=voicemail_response_delay,
        )
        self._voicemail_gate = TTSGate(self._conversation_notifier, self._voicemail_notifier)

        # Initialize the parallel pipeline with conversation and classifier branches
        super().__init__(
            # Conversation branch: gate to blocks after voicemail detection
            [self._conversation_gate],
            # Classification branch: gate -> context -> LLM -> processor -> context
            [
                self._classifier_gate,
                self._context_aggregator.user(),
                self._classifier_llm,
                self._classification_processor,
                self._context_aggregator.assistant(),
            ],
        )

        # Register the voicemail detected event after super().__init__()
        self._register_event_handler("on_voicemail_detected")

    def _validate_prompt(self, prompt: str) -> None:
        """Validate custom prompt contains required response format instructions.

        Custom prompts must instruct the LLM to respond with exactly "CONVERSATION"
        or "VOICEMAIL" for the detection logic to work properly. This method
        checks for the presence of these keywords and warns if they're missing.

        Args:
            prompt: The custom system prompt to validate.
        """
        has_conversation = "CONVERSATION" in prompt
        has_voicemail = "VOICEMAIL" in prompt

        if not has_conversation or not has_voicemail:
            logger.warning(
                "Custom system prompt should instruct the LLM to respond with exactly "
                '"CONVERSATION" or "VOICEMAIL" for proper detection functionality. '
                f"Consider appending VoicemailDetector.CLASSIFIER_RESPONSE_INSTRUCTION to your prompt: "
                f'"{self.CLASSIFIER_RESPONSE_INSTRUCTION}"'
            )

    def detector(self) -> "VoicemailDetector":
        """Get the detector pipeline for placement after STT in the main pipeline.

        This should be placed after the STT service and before the context
        aggregator in your main pipeline to enable voicemail classification.

        Returns:
            The VoicemailDetector instance itself (which is a ParallelPipeline).
        """
        return self

    def gate(self) -> TTSGate:
        """Get the gate processor for placement after TTS in the main pipeline.

        This should be placed after the TTS service and before the transport
        output to enable TTS frame gating during classification.

        Returns:
            The TTSGate processor instance.
        """
        return self._voicemail_gate

    def add_event_handler(self, event_name: str, handler):
        """Add an event handler for voicemail detection events.

        Args:
            event_name: The name of the event to handle.
            handler: The function to call when the event occurs.
        """
        if event_name == "on_voicemail_detected":
            self._classification_processor.add_event_handler(event_name, handler)
        else:
            super().add_event_handler(event_name, handler)

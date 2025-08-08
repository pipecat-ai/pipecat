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
"""

import asyncio
from typing import Awaitable, Callable, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    CancelTaskFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier


class ClassifierGate(FrameProcessor):
    """Gate processor that controls frame flow based on classification decisions.

    The gate starts open to allow initial classification processing and closes
    permanently once a classification decision is made (CONVERSATION or VOICEMAIL).
    This ensures the classifier only runs until a definitive decision is reached,
    preventing unnecessary LLM calls and maintaining system efficiency.

    The gate allows all frames to pass through while open, but once closed, only
    allows system frames (interruptions, end frames, cancel frames) to continue.
    """

    def __init__(self, notifier: BaseNotifier):
        """Initialize the classifier gate.

        Args:
            notifier: Notifier that signals when a classification decision has
                been made and the gate should close.
        """
        super().__init__()
        self._notifier = notifier
        self._gate_opened = True
        self._gate_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on classification decisions.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Start the notification waiting task immediately
            self._gate_task = self.create_task(self._wait_for_notification())

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Clean up the gate task when pipeline ends or is cancelled
            if self._gate_task:
                await self.cancel_task(self._gate_task)
                self._gate_task = None

        # Gate logic: open gate allows all frames, closed gate only allows specific system frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif not self._gate_opened and isinstance(
            frame, (BotInterruptionFrame, EndTaskFrame, EndFrame, CancelTaskFrame, CancelFrame)
        ):
            await self.push_frame(frame, direction)

    async def _wait_for_notification(self):
        """Wait for classification decision notification and close the gate.

        This method blocks until the VoicemailProcessor makes a classification
        decision and signals through the notifier. Once notified, the gate
        closes permanently to stop further classification processing.
        """
        try:
            await self._notifier.wait()

            if self._gate_opened:
                self._gate_opened = False
                logger.debug(f"{self}: Gate closed - classification complete")
        except asyncio.CancelledError:
            logger.debug(f"{self}: Gate task was cancelled")
            raise
        except Exception as e:
            logger.exception(f"{self}: Error in gate task: {e}")
            raise


class VoicemailProcessor(FrameProcessor):
    """Processor that handles LLM classification responses and triggers callbacks.

    This processor aggregates LLM text tokens into complete responses and analyzes
    them to determine if the call reached a voicemail system or a live person.
    It uses the LLM response frame delimiters (LLMFullResponseStartFrame and
    LLMFullResponseEndFrame) to ensure complete token aggregation regardless
    of how the LLM tokenizes the response words.

    The processor expects responses containing either "CONVERSATION" (indicating
    a human answered) or "VOICEMAIL" (indicating an automated system). Once a
    decision is made, it triggers the appropriate notifications and callbacks.
    """

    def __init__(
        self,
        *,
        gate_notifier: BaseNotifier,
        conversation_notifier: BaseNotifier,
        voicemail_notifier: BaseNotifier,
        on_voicemail_detected: Optional[Callable[["VoicemailProcessor"], Awaitable[None]]] = None,
    ):
        """Initialize the voicemail processor.

        Args:
            gate_notifier: Notifier to signal the ClassifierGate about classification
                decisions so it can close and stop processing.
            conversation_notifier: Notifier to signal the VoicemailBuffer to release
                all buffered TTS frames for normal conversation flow.
            voicemail_notifier: Notifier to signal the VoicemailBuffer to clear
                buffered TTS frames since voicemail was detected.
            on_voicemail_detected: Optional callback function called when voicemail
                is detected. The callback receives this processor instance and can
                use it to push custom frames (like voicemail greetings).
        """
        super().__init__()
        self._gate_notifier = gate_notifier
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._on_voicemail_detected = on_voicemail_detected

        # Aggregation state for collecting complete LLM responses
        self._processing_response = False
        self._response_buffer = ""
        self._decision_made = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle LLM classification responses.

        This method implements a state machine for aggregating LLM responses:
        1. LLMFullResponseStartFrame: Begin collecting tokens
        2. LLMTextFrame: Accumulate text tokens into buffer
        3. LLMFullResponseEndFrame: Process complete response and make decision

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Begin aggregating a new LLM response
            self._processing_response = True
            self._response_buffer = ""
            logger.debug(f"{self}: Starting LLM response aggregation")

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Complete response received - make classification decision
            if self._processing_response and not self._decision_made:
                await self._process_classification(self._response_buffer.strip())
            self._processing_response = False
            self._response_buffer = ""

        elif isinstance(frame, LLMTextFrame) and self._processing_response:
            # Accumulate text tokens from the streaming LLM response
            self._response_buffer += frame.text
            logger.trace(f"{self}: Buffer: '{self._response_buffer}'")

        else:
            # Pass all non-LLM frames through
            # Blocking LLM frames prevents interferences with the downsteram LLM
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
            logger.debug(f"{self}: Decision already made, ignoring response")
            return

        response = full_response.upper()
        logger.info(f"{self}: Classifying response: '{full_response}'")

        if "CONVERSATION" in response:
            # Human answered - continue normal conversation flow
            self._decision_made = True
            logger.info(f"{self}: CONVERSATION detected - releasing TTS buffer")
            await self._gate_notifier.notify()  # Close the classifier gate
            await self._conversation_notifier.notify()  # Release buffered TTS frames

        elif "VOICEMAIL" in response:
            # Voicemail detected - trigger voicemail handling
            self._decision_made = True
            logger.info(f"{self}: VOICEMAIL detected - clearing TTS buffer and triggering callback")
            await self._gate_notifier.notify()  # Close the classifier gate
            await self._voicemail_notifier.notify()  # Clear buffered TTS frames

            # Interrupt the current pipeline to stop any ongoing processing
            await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

            # Execute developer callback for custom voicemail handling
            if self._on_voicemail_detected:
                try:
                    await self._on_voicemail_detected(self)
                except Exception as e:
                    logger.exception(f"{self}: Error in voicemail callback: {e}")
        else:
            # Unexpected response - log warning but don't crash
            logger.warning(f"{self}: Unexpected classification response: '{full_response}'")


class VoicemailBuffer(FrameProcessor):
    """Buffers TTS frames until voicemail classification decision is made.

    This processor holds TTS output frames in a buffer while the voicemail
    classification is in progress. This prevents audio from being played
    to the caller before determining if they're human or a voicemail system.

    The buffer operates in two modes based on the classification result:

    - CONVERSATION: Releases all buffered frames to continue normal dialogue
    - VOICEMAIL: Clears buffered frames since they're not needed for voicemail

    The buffering only applies to TTS-related frames (TTSTextFrame, TTSAudioRawFrame).
    All other frames pass through immediately to maintain proper pipeline flow.
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the voicemail buffer.

        Args:
            conversation_notifier: Notifier that signals when a conversation is
                detected and buffered frames should be released for playback.
            voicemail_notifier: Notifier that signals when voicemail is detected
                and buffered frames should be cleared (not played).
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._frame_buffer: List[tuple[Frame, FrameDirection]] = []
        self._buffering_active = True
        self._conversation_task: Optional[asyncio.Task] = None
        self._voicemail_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle buffering logic based on frame type.

        TTS frames are buffered while classification is active. All other frames
        pass through immediately. The buffering state is controlled by the
        classification notifications.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Start notification waiting tasks for both conversation and voicemail
            self._conversation_task = self.create_task(self._wait_for_conversation())
            self._voicemail_task = self.create_task(self._wait_for_voicemail())
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Clean up notification tasks when pipeline ends
            if self._conversation_task:
                await self.cancel_task(self._conversation_task)
                self._conversation_task = None
            if self._voicemail_task:
                await self.cancel_task(self._voicemail_task)
                self._voicemail_task = None
            await self.push_frame(frame, direction)

        # Core buffering logic: hold TTS frames, pass everything else through
        elif self._buffering_active and isinstance(frame, (TTSTextFrame, TTSAudioRawFrame)):
            # Buffer TTS frames while waiting for classification decision
            self._frame_buffer.append((frame, direction))
        else:
            # Pass through all non-TTS frames immediately
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection notification and release buffered frames.

        When a conversation is detected, all buffered TTS frames are released
        in order to continue normal dialogue flow. This allows the bot to
        respond naturally to the human caller.
        """
        try:
            await self._conversation_notifier.wait()

            # Release all buffered frames in original order
            self._buffering_active = False
            for frame, direction in self._frame_buffer:
                await self.push_frame(frame, direction)
            self._frame_buffer.clear()

            # Cancel the voicemail task since decision is final
            if self._voicemail_task:
                await self.cancel_task(self._voicemail_task)
                self._voicemail_task = None

        except asyncio.CancelledError:
            logger.debug(f"{self}: Conversation task was cancelled")
            raise

    async def _wait_for_voicemail(self):
        """Wait for voicemail detection notification and clear buffered frames.

        When voicemail is detected, all buffered TTS frames are discarded
        since they were intended for human conversation and are not appropriate
        for voicemail systems. The developer callback will handle voicemail-
        specific audio output.
        """
        try:
            await self._voicemail_notifier.wait()

            # Clear buffered frames without playing them
            self._buffering_active = False
            self._frame_buffer.clear()

            # Cancel the conversation task since decision is final
            if self._conversation_task:
                await self.cancel_task(self._conversation_task)
                self._conversation_task = None

        except asyncio.CancelledError:
            logger.debug(f"{self}: Voicemail task was cancelled")
            raise


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
    a buffering system to prevent TTS output until classification is complete.
    Once a decision is made, the appropriate action is taken:

    - CONVERSATION: Continue normal bot dialogue
    - VOICEMAIL: Trigger developer callback for custom voicemail handling

    Example::

        classification_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        async def handle_voicemail(processor):
            await processor.push_frame(TTSSpeakFrame("Please leave a message."))

        detector = VoicemailDetector(
            llm=classification_llm,
            on_voicemail_detected=handle_voicemail
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            detector.detector(),          # Classification
            context_aggregator.user(),
            llm,
            tts,
            detector.buffer(),            # TTS buffering
            transport.output(),
            context_aggregator.assistant(),
        ])
    """

    DEFAULT_SYSTEM_PROMPT = """You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text.

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

Respond with ONLY "CONVERSATION" if a person answered, or "VOICEMAIL" if it's voicemail/recording."""

    def __init__(
        self,
        *,
        llm: LLMService,
        on_voicemail_detected: Optional[Callable[["VoicemailProcessor"], Awaitable[None]]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the voicemail detector with classification and buffering components.

        Args:
            llm: LLM service used for voicemail vs conversation classification.
                Should be fast and reliable for real-time classification.
            on_voicemail_detected: Optional callback function invoked when voicemail
                is detected. Receives the VoicemailProcessor instance which can be
                used to push frames (like custom voicemail greetings).
            system_prompt: Optional custom system prompt for classification. If None,
                uses the default prompt optimized for outbound calling scenarios.
                Custom prompts should instruct the LLM to respond with exactly
                "CONVERSATION" or "VOICEMAIL" for proper detection functionality.
        """
        self._classifier_llm = llm
        self._prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

        # Validate custom prompts to ensure they work with the detection logic
        if system_prompt is not None:
            self._validate_prompt(system_prompt)

        # Set up the LLM context with the classification prompt
        self._messages = [
            {
                "role": "system",
                "content": self._prompt,
            },
        ]

        # Create the LLM context and aggregators for conversation management
        self._context = OpenAILLMContext(self._messages)
        self._context_aggregator = llm.create_context_aggregator(self._context)

        # Create notification system for coordinating between components
        self._gate_notifier = EventNotifier()  # Signals classification completion
        self._conversation_notifier = EventNotifier()  # Signals conversation detected
        self._voicemail_notifier = EventNotifier()  # Signals voicemail detected

        # Create the processor components
        self._classifier_gate = ClassifierGate(self._gate_notifier)
        self._voicemail_processor = VoicemailProcessor(
            gate_notifier=self._gate_notifier,
            conversation_notifier=self._conversation_notifier,
            voicemail_notifier=self._voicemail_notifier,
            on_voicemail_detected=on_voicemail_detected,
        )
        self._voicemail_buffer = VoicemailBuffer(
            self._conversation_notifier, self._voicemail_notifier
        )

        # Initialize the parallel pipeline with conversation and classifier branches
        super().__init__(
            # Conversation branch: empty pipeline for normal frame flow
            [],
            # Classification branch: gate -> context -> LLM -> processor -> context
            [
                self._classifier_gate,
                self._context_aggregator.user(),
                self._classifier_llm,
                self._voicemail_processor,
                self._context_aggregator.assistant(),
            ],
        )

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
                'Example: "Respond with ONLY \\"CONVERSATION\\" if a person answered, or \\"VOICEMAIL\\" if it\'s voicemail/recording."'
            )

    def detector(self) -> "VoicemailDetector":
        """Get the detector pipeline for placement after STT in the main pipeline.

        This should be placed after the STT service and before the context
        aggregator in your main pipeline to enable voicemail classification.

        Returns:
            The VoicemailDetector instance itself (which is a ParallelPipeline).
        """
        return self

    def buffer(self) -> VoicemailBuffer:
        """Get the buffer processor for placement after TTS in the main pipeline.

        This should be placed after the TTS service and before the transport
        output to enable TTS frame buffering during classification.

        Returns:
            The VoicemailBuffer processor instance.
        """
        return self._voicemail_buffer

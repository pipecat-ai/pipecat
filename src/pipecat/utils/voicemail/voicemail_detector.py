#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail detection module for Pipecat.

This module provides voicemail detection capabilities using parallel pipeline
processing to classify incoming calls as either voicemail messages or live
conversations.
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

    The gate starts open and closes permanently once a classification decision
    is made (LIVE or MAIL). This ensures the classifier only runs until a definitive
    decision is reached.
    """

    def __init__(self, notifier: BaseNotifier):
        """Initialize the classifier gate.

        Args:
            notifier: Notifier that signals when to close the gate.
        """
        super().__init__()
        self._notifier = notifier
        self._gate_opened = True
        self._gate_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Start the task immediately, don't wait for other conditions
            self._gate_task = self.create_task(self._wait_for_notification())
            logger.info(f"{self}: Gate task started, waiting for notification")

        elif isinstance(frame, (EndFrame, CancelFrame)):
            if self._gate_task:
                await self.cancel_task(self._gate_task)
                self._gate_task = None

        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif not self._gate_opened and isinstance(
            frame, (BotInterruptionFrame, EndTaskFrame, EndFrame, CancelTaskFrame, CancelFrame)
        ):
            await self.push_frame(frame, direction)

    async def _wait_for_notification(self):
        """Wait for classification decision notification."""
        try:
            logger.info(f"{self}: Waiting for notification...")
            await self._notifier.wait()
            logger.info(f"{self}: Received notification!")

            if self._gate_opened:
                self._gate_opened = False
                logger.info(f"{self}: Gate closed")
        except asyncio.CancelledError:
            logger.debug(f"{self}: Gate task was cancelled")
            raise
        except Exception as e:
            logger.exception(f"{self}: Error in gate task: {e}")
            raise


class VoicemailProcessor(FrameProcessor):
    """Processor that handles LLM classification responses and triggers callbacks.

    Processes LLM text responses to determine if the call is a voicemail (MAIL)
    or conversation (LIVE), then triggers appropriate actions including
    developer callbacks for voicemail detection.
    """

    def __init__(
        self,
        *,
        gate_notifier: BaseNotifier,
        conversation_notifier: BaseNotifier,  # Buffer should release frames
        voicemail_notifier: BaseNotifier,  # Buffer should clear frames
        on_voicemail_detected: Optional[Callable[["VoicemailProcessor"], Awaitable[None]]] = None,
    ):
        """Initialize the voicemail processor.

        Args:
            gate_notifier: Notifier to signal gate about classification decisions.
            conversation_notifier: Notifier to signal buffer to release frames.
            voicemail_notifier: Notifier to signal buffer to clear frames.
            on_voicemail_detected: Callback function called when voicemail is detected.
        """
        super().__init__()
        self._gate_notifier = gate_notifier
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._on_voicemail_detected = on_voicemail_detected

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle LLM classification responses.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            response = frame.text.strip().upper()
            if "LIVE" in response:
                logger.info(f"{self}: LIVE conversation detected - releasing buffer")
                await self._gate_notifier.notify()
                await self._conversation_notifier.notify()
            elif "MAIL" in response:
                logger.info(f"{self}: VOICEMAIL detected - triggering callback")
                # Notify gate to close (decision is final)
                await self._gate_notifier.notify()
                await self._voicemail_notifier.notify()
                # Push BotInterruptionFrame to clear the pipeline
                await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
                # Call developer callback if provided
                if self._on_voicemail_detected:
                    try:
                        await self._on_voicemail_detected(self)
                    except Exception as e:
                        logger.exception(f"{self}: Error in voicemail callback: {e}")

        else:
            # Push the frame
            await self.push_frame(frame, direction)


class VoicemailBuffer(FrameProcessor):
    """Buffers TTS frames until voicemail classification decision is made.

    Holds TTS frames in a buffer while voicemail classification is in progress.
    Releases all buffered frames when conversation is detected, or keeps them
    buffered when voicemail is detected.
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the voicemail buffer.

        Args:
            conversation_notifier: Notifier that signals when to release buffered frames.
            voicemail_notifier: Notifier that signals when to keep buffered frames.
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._frame_buffer: List[tuple[Frame, FrameDirection]] = []
        self._buffering_active = True
        self._conversation_task: Optional[asyncio.Task] = None
        self._voicemail_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle buffering logic.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._conversation_task = self.create_task(self._wait_for_conversation())
            self._voicemail_task = self.create_task(self._wait_for_voicemail())
            logger.info(f"{self}: Buffer tasks started")
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            if self._conversation_task:
                await self.cancel_task(self._conversation_task)
                self._conversation_task = None
            if self._voicemail_task:
                await self.cancel_task(self._voicemail_task)
                self._voicemail_task = None
            await self.push_frame(frame, direction)

        # Buffer TTS frames while buffering is active
        if self._buffering_active and isinstance(frame, (TTSTextFrame, TTSAudioRawFrame)):
            self._frame_buffer.append((frame, direction))
        else:
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection - release buffered frames."""
        try:
            await self._conversation_notifier.wait()
            logger.info(f"{self}: CONVERSATION - releasing frames")

            self._buffering_active = False
            for frame, direction in self._frame_buffer:
                await self.push_frame(frame, direction)
            self._frame_buffer.clear()

            # Cancel the other task
            if self._voicemail_task:
                await self.cancel_task(self._voicemail_task)
                self._voicemail_task = None

        except asyncio.CancelledError:
            raise

    async def _wait_for_voicemail(self):
        """Wait for voicemail detection - clear buffered frames."""
        try:
            await self._voicemail_notifier.wait()
            logger.info(f"{self}: VOICEMAIL - clearing frames")

            self._buffering_active = False
            self._frame_buffer.clear()

            # Cancel the other task
            if self._conversation_task:
                await self.cancel_task(self._conversation_task)
                self._conversation_task = None

        except asyncio.CancelledError:
            raise


class VoicemailDetector(ParallelPipeline):
    """Parallel pipeline for detecting voicemail vs. live conversation.

    Uses a parallel pipeline architecture with two branches:
    1. Conversation branch: Normal frame flow for live conversations
    2. Classification branch: LLM-based classification that can interrupt for voicemail

    The classifier runs in parallel and makes a one-time decision to either:
    - Continue normal conversation flow (LIVE response)
    - Interrupt and trigger voicemail handling (MAIL response)
    """

    # Default prompt
    DEFAULT_SYSTEM_PROMPT = """You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text.

HUMAN ANSWERED - LIVE CONVERSATION (respond "LIVE"):
- Personal greetings: "Hello?", "Hi", "Yeah?", "John speaking"
- Interactive responses: "Who is this?", "What do you want?", "Can I help you?"
- Conversational tone expecting back-and-forth dialogue
- Questions directed at the caller: "Hello? Anyone there?"
- Informal responses: "Yep", "What's up?", "Speaking"
- Natural, spontaneous speech patterns
- Immediate acknowledgment of the call

VOICEMAIL SYSTEM (respond "MAIL"):
- Automated voicemail greetings: "Hi, you've reached [name], please leave a message"
- Phone carrier messages: "The number you have dialed is not in service", "Please leave a message", "All circuits are busy"
- Professional voicemail: "This is [name], I'm not available right now"
- Instructions about leaving messages: "leave a message", "leave your name and number"
- References to callback or messaging: "call me back", "I'll get back to you"
- Carrier system messages: "mailbox is full", "has not been set up"
- Business hours messages: "our office is currently closed"

Respond with ONLY "LIVE" if a person answered, or "MAIL" if it's voicemail/recording."""

    def __init__(
        self,
        *,
        llm: LLMService,
        on_voicemail_detected: Optional[Callable[[], Awaitable[None]]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the voicemail detector.

        Args:
            llm: LLM service for classification.
            on_voicemail_detected: Callback function called when voicemail is detected.
            system_prompt: Optional custom system prompt for classification. If None, uses
                default prompt optimized for outbound calling scenarios. If providing a
                custom prompt, ensure it results in a clear "LIVE" or "MAIL" response, where
                "LIVE" indicates a human answered and "MAIL" indicates voicemail.
        """
        self._classifier_llm = llm
        self._prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

        if system_prompt is not None:
            self._validate_prompt(system_prompt)

        self._messages = [
            {
                "role": "system",
                "content": self._prompt,
            },
        ]

        self._context = OpenAILLMContext(self._messages)
        self._context_aggregator = llm.create_context_aggregator(self._context)
        self._gate_notifier = EventNotifier()
        self._conversation_notifier = EventNotifier()
        self._voicemail_notifier = EventNotifier()
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

        super().__init__(
            # Conversation branch
            [],
            # Classifer branch
            [
                self._classifier_gate,
                self._context_aggregator.user(),
                self._classifier_llm,
                self._voicemail_processor,
                self._context_aggregator.assistant(),
            ],
        )

    def detector(self) -> "VoicemailDetector":
        """Get the detector pipeline (for placement after STT).

        Returns:
            The VoicemailDetector instance itself.
        """
        return self

    def buffer(self) -> VoicemailBuffer:
        """Get the buffer processor (for placement after TTS).

        Returns:
            The VoicemailBuffer processor instance.
        """
        return self._voicemail_buffer

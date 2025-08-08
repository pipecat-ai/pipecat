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
    is made (YES or NO). This ensures the classifier only runs until a definitive
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

    Processes LLM text responses to determine if the call is a voicemail (YES)
    or conversation (NO), then triggers appropriate actions including developer
    callbacks for voicemail detection.
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
            if "NO" in response:
                logger.info(f"{self}: CONVERSATION detected - notifying to close gate")
                await self._gate_notifier.notify()
                await self._conversation_notifier.notify()
            elif "YES" in response:
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
    - Continue normal conversation flow (NO response)
    - Interrupt and trigger voicemail handling (YES response)
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        on_voicemail_detected: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """Initialize the voicemail detector.

        Args:
            llm: LLM service for classification.
            on_voicemail_detected: Callback function called when voicemail is detected.
        """
        self._classifier_llm = llm
        self._messages = [
            {
                "role": "system",
                "content": """You are a voicemail detection classifier. Your job is to determine if the caller is leaving a voicemail message or trying to have a live conversation.

VOICEMAIL INDICATORS (respond "YES"):
- One-way communication (caller talks without expecting immediate responses)
- Messages like "Hi, this is [name], please call me back"
- "I'm calling about..." followed by details without pausing for response
- "Leave me a message" or "call me when you get this"
- Monologue-style speech patterns
- Mentions of time/date when they're calling
- Business-like messages with contact information

CONVERSATION INDICATORS (respond "NO"):
- Interactive speech ("Hello?", "Are you there?", "Can you hear me?")
- Questions directed at the recipient expecting immediate answers
- Responses to prompts or questions
- Back-and-forth dialogue patterns
- Greetings expecting responses ("Hi, how are you?")
- Real-time problem solving or discussion

Respond with ONLY "YES" if it's a voicemail, or "NO" if it's a conversation attempt. Do not explain your reasoning.""",
            },
        ]
        self._context = OpenAILLMContext(self._messages)
        self._context_aggregator = llm.create_context_aggregator(self._context)
        self._gate_notifier = EventNotifier()
        self._conversation_notifier = EventNotifier()  # For releasing buffer
        self._voicemail_notifier = EventNotifier()  # For clearing buffer
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

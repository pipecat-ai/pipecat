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
from typing import Awaitable, Callable, Optional

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
        notifier: BaseNotifier,
        on_voicemail_detected: Optional[Callable[["VoicemailProcessor"], Awaitable[None]]] = None,
    ):
        """Initialize the voicemail processor.

        Args:
            notifier: Notifier to signal classification decisions.
            on_voicemail_detected: Callback function called when voicemail is detected.
        """
        super().__init__()
        self._notifier = notifier
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
                await self._notifier.notify()
            elif "YES" in response:
                logger.info(f"{self}: VOICEMAIL detected - triggering callback")
                # Notify gate to close (decision is final)
                await self._notifier.notify()
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
        self._conversation_notifier = EventNotifier()
        self._classifier_gate = ClassifierGate(self._conversation_notifier)
        self._voicemail_processor = VoicemailProcessor(
            self._conversation_notifier, on_voicemail_detected
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

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection


class IdleHandler:
    """Helper class to manage user idle retry logic with contextually-aware LLM responses.

    This handler uses the LLM to generate contextually appropriate follow-up messages
    when a user becomes idle during a conversation. It implements an escalating retry
    strategy with three attempts before ending the conversation.
    """

    def __init__(self, llm):
        """Initialize the idle handler.

        Args:
            llm: The LLM service to use for generating contextual follow-ups.
        """
        self._llm = llm
        self._retry_count = 0

    def reset(self):
        """Reset the retry count when user becomes active."""
        self._retry_count = 0

    async def handle_idle(self, aggregator):
        """Handle user idle event with escalating prompts using contextual LLM inference.

        Args:
            aggregator: The user aggregator that triggered the idle event.
        """
        self._retry_count += 1

        if self._retry_count <= 2:
            # Clone the messages from the aggregator's context
            inference_messages = aggregator.messages.copy()

            # Add a system message to guide the LLM's response
            if self._retry_count == 1:
                inference_messages.append(
                    {
                        "role": "system",
                        "content": """The user has been quiet for a moment. Based on the conversation context, generate a brief, natural follow-up to re-engage them. Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational (not robotic)
- Be concise (1-2 sentences max)
- Gently prompt them to continue without being pushy

Examples:
- If you asked a question: "Take your time! I'm curious to hear your thoughts."
- If discussing a topic: "What do you think about that?"
- If they seemed engaged: "Are you still there? I'd love to hear more."

Generate ONLY the follow-up message, nothing else.""",
                    }
                )
            else:  # retry_count == 2
                inference_messages.append(
                    {
                        "role": "system",
                        "content": """The user has been quiet for a while now. Generate a brief, friendly check-in message. Your response should:
- Acknowledge they might be busy or thinking
- Offer to continue or pause the conversation
- Be warm and understanding
- Be very brief (1 sentence)

Examples:
- "No rush! Let me know if you'd like to continue."
- "Take your time - I'm here when you're ready."
- "Should we pick this up later?"

Generate ONLY the check-in message, nothing else.""",
                    }
                )

            # Create a temporary context for inference
            temp_context = LLMContext(messages=inference_messages)

            # Run inference to get a contextually-aware response
            response = None
            try:
                logger.info(f"Running inference for idle follow-up (attempt {self._retry_count})")
                response = await self._llm.run_inference(temp_context)
            except Exception as e:
                logger.error(f"Error during idle inference: {e}")

            # Use LLM response if available, otherwise fall back to generic message
            if response:
                logger.info(f"Generated contextual follow-up: {response}")
                await aggregator.push_frame(TTSSpeakFrame(response))
            else:
                logger.warning("No response from LLM inference, using fallback")
                fallback = (
                    "Are you still there?"
                    if self._retry_count == 1
                    else "Let me know if you'd like to continue."
                )
                await aggregator.push_frame(TTSSpeakFrame(fallback))
        else:
            # Third attempt: End the conversation gracefully
            logger.info("User idle timeout reached, ending conversation")
            await aggregator.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await aggregator.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

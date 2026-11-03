#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM text processor module for aggregating raw LLM output text.

This processor converts LLMTextFrames into AggregatedTextFrames using the configured
text aggregator. Text transforms can optionally be registered on it; by default the
aggregations are pushed on as the aggregator produced them. Because the aggregated frames
reach every consumer downstream, a transform registered here changes what the TTS speaks,
what the context records and what clients display, whereas one registered on a TTS
service changes the text for the synthesizer only.
"""

from pipecat.frames.frames import (
    AggregatedTextFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.text.base_text_aggregator import Aggregation, BaseTextAggregator, TextType
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.text.transforms.text_transform import TextTransform


class LLMTextProcessor(FrameProcessor):
    """Aggregates LLM text frames before they are processed further.

    This processor converts LLMTextFrames into AggregatedTextFrames based on the configured
    text aggregator. By default it leaves the text alone; the frames it pushes carry the
    aggregations as the aggregator produced them.

    Text transforms can be registered to rewrite the aggregations on their way through.
    The frames feed the TTS service, the context aggregator and client output alike, so a
    transform registered here changes the text for all of them, which suits a habit the
    LLM keeps repeating (emoji, stage directions), a name it should not use, or a pattern
    to redact. To change only what is spoken, register the transform on the TTS service
    instead.

    A transform is an async callable ``(text, text_type) -> text``, registered for one
    text type or ``"*"`` for all. A transform whose output is only whitespace drops the
    aggregation, so nothing is spoken or added to the context for it.

    Example::

        # Aggregates by sentence, nothing rewritten.
        llm_text_processor = LLMTextProcessor()

        # Rewrites what everyone sees: the voice, the context and the client.
        llm_text_processor = LLMTextProcessor(text_transforms=[("*", strip_emoji)])

        # Rewrites what the voice says only.
        tts = CartesiaTTSService(text_transforms=[("*", strip_markdown)])
    """

    def __init__(
        self,
        *,
        text_aggregator: BaseTextAggregator | None = None,
        text_transforms: list[tuple[TextType | str, TextTransform]] | None = None,
        **kwargs,
    ):
        """Initialize the LLM text processor.

        Args:
            text_aggregator: An optional text aggregator to use for processing LLM text frames. By
                default, a SimpleTextAggregator aggregating by sentence will be used.
            text_transforms: Optional callables to rewrite aggregations with, as
                ``(text_type | "*", transform)`` pairs. None by default. A rewrite
                reaches everything downstream: the TTS, the context and the client.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._text_aggregator: BaseTextAggregator = text_aggregator or SimpleTextAggregator()
        self._text_transforms: list[tuple[TextType | str, TextTransform]] = text_transforms or []
        self._skip_tts: bool | None = None

    def add_text_transformer(
        self, transform_function: TextTransform, text_type: TextType | str = "*"
    ):
        """Transform aggregated text of a specific type.

        Args:
            transform_function: The function to apply. It takes the text and its type and
                returns the transformed text, e.g.
                ``async def my_transform(text: str, text_type: str) -> str``.
            text_type: The type of text to transform. Defaults to "*", every type.
        """
        self._text_transforms.append((text_type, transform_function))

    def remove_text_transformer(
        self, transform_function: TextTransform, text_type: TextType | str = "*"
    ):
        """Remove a text transformer for a specific text type.

        Args:
            transform_function: The function to remove.
            text_type: The type of text the transformer was registered for.
        """
        self._text_transforms = [
            (transform_type, func)
            for transform_type, func in self._text_transforms
            if not (transform_type == text_type and func == transform_function)
        ]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an LLMTextFrames using the aggregator to generate AggregatedTextFrames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame.skip_tts)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStoppedFrame):
            # A speech-to-speech service stops its speech before it ends the
            # response. The text still held was spoken, so it goes out ahead of
            # the stop: text behind it would reach the output transport after
            # the bot stopped speaking.
            await self._handle_llm_end(self._skip_tts)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._handle_llm_end()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruption(self, _):
        """Handle interruptions by resetting the text aggregator."""
        await self._text_aggregator.handle_interruption()

    async def reset(self):
        """Reset the internal state of the text processor and its aggregator."""
        await self._text_aggregator.reset()

    async def _handle_llm_text(self, in_frame: LLMTextFrame):
        self._skip_tts = in_frame.skip_tts
        async for aggregation in self._text_aggregator.aggregate(in_frame.text):
            out_frame = await self._aggregated_text_frame(aggregation)
            if out_frame:
                out_frame.append_to_context = True
                out_frame.skip_tts = in_frame.skip_tts
                await self.push_frame(out_frame)

    async def _handle_llm_end(self, skip_tts: bool | None = None):
        # Flush any remaining text
        remaining = await self._text_aggregator.flush()
        if remaining:
            out_frame = await self._aggregated_text_frame(remaining)
            if out_frame:
                out_frame.skip_tts = skip_tts
                await self.push_frame(out_frame)

    async def _aggregated_text_frame(self, aggregation: Aggregation) -> AggregatedTextFrame | None:
        """Build the frame for an aggregation, or None when a registered transform dropped it."""
        text = await self._transform_text(aggregation.text, aggregation.type)
        if text is None:
            return None
        return AggregatedTextFrame(
            text=text, text_type=aggregation.type, raw_text=self._raw_text(aggregation, text)
        )

    async def _transform_text(self, text: str, text_type: TextType | str) -> str | None:
        """Run the transforms registered for ``text_type`` on the text.

        Returns:
            The transformed text, or None when it should be dropped: a transform left
            nothing but whitespace, or one failed.
        """
        transformed_text = text
        try:
            for transform_type, transform in self._text_transforms:
                if transform_type in (text_type, "*"):
                    transformed_text = await transform(transformed_text, text_type)
        except Exception as e:
            # The failure came from the application's transform. Passing the text on
            # untransformed isn't safe, as a transform may exist to remove something,
            # so the aggregation is dropped.
            await self.push_error(
                error_msg=f"Error transforming LLM text [{transformed_text}]: {e}",
                exception=e,
                category=ErrorCategory.APPLICATION,
            )
            return None
        if transformed_text != text and not transformed_text.strip():
            return None
        return transformed_text

    @staticmethod
    def _raw_text(aggregation: Aggregation, text: str) -> str:
        """The text the context records: a pattern match keeps its delimiters."""
        if not isinstance(aggregation, PatternMatch):
            return text
        if text == aggregation.text:
            return aggregation.full_match
        # Splice the rewritten content back between the delimiters. The content is
        # the first occurrence of itself in the full match.
        start = aggregation.full_match.find(aggregation.text)
        if start < 0:
            return text
        end = start + len(aggregation.text)
        return aggregation.full_match[:start] + text + aggregation.full_match[end:]

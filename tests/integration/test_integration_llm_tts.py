#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM-to-TTS pipeline integration tests for XMLFunctionTagFilter with audio validation."""

import asyncio
import pytest

from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from mock_llm_service import MockLLMService
from mock_tts_service import PredictableMockTTSService


async def aggregate_llm_output_text(llm_text: str) -> list[str]:
    """Aggregate LLM output text into sentences using SimpleTextAggregator.
    
    Args:
        llm_text: Raw LLM output text to aggregate into sentences
        
    Returns:
        List of aggregated sentences
    """
    text_aggregator = SimpleTextAggregator()
    sentences = []
    
    for char in llm_text:
        result = await text_aggregator.aggregate(char)
        if result:
            sentences.append(result)
    
    # Get any remaining text
    await text_aggregator.reset()
    final_result = text_aggregator.text
    if final_result.strip():
        sentences.append(final_result)
    
    return sentences


async def filter_aggregated_sentences(sentences: list[str]) -> list[str]:
    """Filter aggregated sentences to remove XML function tags.
    
    Args:
        sentences: List of aggregated sentences to filter
        
    Returns:
        List of filtered sentences with function tags removed
    """
    xml_filter = XMLFunctionTagFilter()
    filtered_sentences = []
    
    for sentence in sentences:
        filtered = await xml_filter.filter(sentence)
        if filtered.strip():
            filtered_sentences.append(filtered.strip())
    
    return filtered_sentences


async def generate_expected_audio_from_sentences(filtered_sentences: list[str]) -> bytes:
    """Generate expected audio data from filtered sentences.
    
    Args:
        filtered_sentences: List of filtered sentences to generate audio for
        
    Returns:
        Combined audio bytes for all sentences
    """
    # Calculate expected combined audio
    all_expected_audio = b""
    for sentence in filtered_sentences:
        sentence_audio = PredictableMockTTSService.create_deterministic_audio_from_text(sentence)
        all_expected_audio += sentence_audio
    
    return all_expected_audio


async def run_test_collect_frames(pipeline, frames_to_send):
    """Run test and collect all downstream frames without validation."""
    received_down = asyncio.Queue()
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=True,
    )
    
    test_pipeline = Pipeline([pipeline, sink])
    task = PipelineTask(test_pipeline, params=PipelineParams())
    
    async def push_frames():
        await asyncio.sleep(0.01)
        for frame in frames_to_send:
            await task.queue_frame(frame)
        await task.queue_frame(EndFrame())
    
    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), push_frames())
    
    # Collect all frames
    received_frames = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame):
            received_frames.append(frame)
    
    return received_frames


@pytest.mark.asyncio
async def test_end_to_end_llm_to_tts_with_xml_filtering():
    """End-to-end test: LLM with function calls → TTS with XMLFunctionTagFilter."""

    text_with_tags = (
        "Hello! I can help you schedule that meeting. "
        '<function=schedule_interview>{"date": "tomorrow"}</function> '
        "The interview has been scheduled successfully!"
    )

    # Process text through the same pipeline as TTS service: aggregation then filtering
    aggregated_sentences = await aggregate_llm_output_text(text_with_tags)
    expected_filtered_sentences = await filter_aggregated_sentences(aggregated_sentences)
    
    # Generate expected audio data
    all_expected_audio = await generate_expected_audio_from_sentences(expected_filtered_sentences)

    llm_chunks = MockLLMService.create_text_chunks(text_with_tags)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    mock_tts = PredictableMockTTSService(
        text_filters=[XMLFunctionTagFilter()],
    )
    # Run pipeline
    pipeline = Pipeline([mock_llm, mock_tts])

    messages = [{"role": "user", "content": "Schedule a meeting"}]
    context = LLMContext(messages)
    frames_to_send = [LLMContextFrame(context)]

    # Collect frames
    received_frames = await run_test_collect_frames(pipeline, frames_to_send)
    actual_audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
            
    # Verify TTS received properly filtered text (function tags removed)
    assert len(mock_tts.received_texts) == len(expected_filtered_sentences), \
        f"Expected {len(expected_filtered_sentences)} sentences, got {len(mock_tts.received_texts)}"
    
    assert len(actual_audio_frames) > 0, "Should generate audio frames"
        
    # Verify combined audio data matches expected deterministic output
    actual_combined_audio = b"".join(f.audio for f in actual_audio_frames)
    expected_combined_audio = all_expected_audio
    assert actual_combined_audio == expected_combined_audio, \
        "Combined audio data should match expected deterministic output"


    print(f"LLM Output: {text_with_tags}")
    print(f"TTS Received (filtered sentences): {mock_tts.received_texts}")
    print(f"Generated {len(actual_audio_frames)} audio frames")
    print("Audio frames match expected output")


@pytest.mark.asyncio
async def test_different_inputs_produce_different_audio():
    """Test that different filtered text produces different audio."""

    xml_filter = XMLFunctionTagFilter()
    
    llm_outputs = [
        "Hello <function=test></function> world",
        "Goodbye <function=end_call></function> friend",
    ]
    
    scenarios = [
        {
            "llm_output": output,
            "expected_filtered": await xml_filter.filter(output),
        }
        for output in llm_outputs
    ]

    audio_results = []

    for scenario in scenarios:
        llm_chunks = MockLLMService.create_text_chunks(scenario["llm_output"])
        mock_llm = MockLLMService(mock_chunks=llm_chunks)
        mock_tts = PredictableMockTTSService(text_filters=[XMLFunctionTagFilter()])

        # Run pipeline
        pipeline = Pipeline([mock_llm, mock_tts])

        messages = [{"role": "user", "content": "Test"}]
        context = LLMContext(messages)

        # Collect frames
        received_frames = await run_test_collect_frames(pipeline, [LLMContextFrame(context)])

        audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
        audio_data = b"".join(f.audio for f in audio_frames)

        received_text = " ".join(mock_tts.received_texts)
        audio_results.append(
            {"filtered_text": received_text, "audio_data": audio_data}
        )

    # Verify different filtered text produce different audio content
    assert (
        audio_results[0]["audio_data"] != audio_results[1]["audio_data"]
    ), "Different text should produce different audio"

    print("Different filtered text produces different audio")


@pytest.mark.asyncio
async def test_multiple_function_calls_filtering():
    """Test filtering multiple function calls in LLM output."""

    text_with_multiple_tags = (
        "Starting the call <function=move_to_main_agenda></function> "
        "with agenda items. <function=end_call></function> Call ended."
    )

    xml_filter = XMLFunctionTagFilter()
    await xml_filter.filter(text_with_multiple_tags)
    
    expected_sentences = ["Starting the call with agenda items.", "Call ended."]

    llm_chunks = MockLLMService.create_text_chunks(text_with_multiple_tags)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    mock_tts = PredictableMockTTSService(text_filters=[XMLFunctionTagFilter()])

    # Run pipeline
    pipeline = Pipeline([mock_llm, mock_tts])
    messages = [{"role": "user", "content": "Handle the call"}]
    context = LLMContext(messages)

    # Collect frames
    received_frames = await run_test_collect_frames(pipeline, [LLMContextFrame(context)])

    assert len(mock_tts.received_texts) == 2
    assert mock_tts.received_texts == expected_sentences

    # Verify audio frames are generated for filtered content
    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) > 0
    
    # Verify audio data matches expected deterministic output for filtered sentences
    actual_combined_audio = b"".join(f.audio for f in audio_frames)
    expected_audio = b""
    for sentence in expected_sentences:
        expected_audio += PredictableMockTTSService.create_deterministic_audio_from_text(sentence)
    
    assert actual_combined_audio == expected_audio, \
        "Combined audio should match expected output for filtered sentences"

    print(f"Original: {text_with_multiple_tags}")
    print(f"Filtered sentences: {mock_tts.received_texts}")
    print("Multiple function calls filtered successfully")


@pytest.mark.asyncio
async def test_empty_function_call_handling():
    """Test handling of text with only function calls (results in empty text)."""

    text_with_only_tags = "<function=end_call></function>"
    
    xml_filter = XMLFunctionTagFilter()
    filtered_text = await xml_filter.filter(text_with_only_tags)
    
    # Empty text should result in no TTS calls
    assert filtered_text.strip() == "", "Filtered text should be empty"

    llm_chunks = MockLLMService.create_text_chunks(text_with_only_tags)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)
    mock_tts = PredictableMockTTSService(text_filters=[XMLFunctionTagFilter()])

    # Run pipeline
    pipeline = Pipeline([mock_llm, mock_tts])
    messages = [{"role": "user", "content": "End call"}]
    context = LLMContext(messages)

    # Collect frames
    received_frames = await run_test_collect_frames(pipeline, [LLMContextFrame(context)])

    # Verify empty filtered text doesn't trigger TTS processing
    assert len(mock_tts.received_texts) == 0, "Empty text should not trigger TTS calls"

    # Verify no audio frames generated for empty content
    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 0, "Empty text should not generate audio frames"
    
    llm_frames = [f for f in received_frames if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))]
    assert len(llm_frames) > 0, "LLM should still generate response frames"

    print("Empty text handling works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
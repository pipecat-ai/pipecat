#!/usr/bin/env python3
"""Test Hecttor filter with pre-recorded audio files through a pipecat pipeline.

This script processes an audio file through:
  1. Hecttor denoising filter (ASR or human mode)
  2. OpenAI Whisper STT (transcription)
  3. OpenAI LLM (generates a response)
  4. OpenAI TTS (speaks the response)
  5. Saves the TTS output to a WAV file

Usage:
    python test_hecttor_filter_audiofile.py input.wav output.wav
    python test_hecttor_filter_audiofile.py input.wav output.wav --mode human --voice-boost

Requirements:
    pip install hecttor_sdk (from https://admin.hecttor.ai)
    Set HECTTOR_API_KEY and OPENAI_API_KEY environment variables
"""

import argparse
import asyncio
import os
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src directory to Python path for development environment
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from loguru import logger

from pipecat.audio.filters.hecttor_filter import HecttorFilter
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor


def read_audio_file(path: str):
    """Read an audio file and return int16 samples and sample rate."""
    data, sample_rate = sf.read(path, dtype="float32")
    if len(data.shape) > 1:
        data = data[:, 0]  # Take first channel if stereo
    samples_int16 = (data * 32767).astype(np.int16)
    return samples_int16, sample_rate


class AudioCollector(FrameProcessor):
    """Collects TTS audio frames, transcriptions, and LLM responses from the pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_chunks = []
        self.sample_rate = None
        self.transcription = None
        self.llm_response = ""
        self.tts_done = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self.transcription = frame.text
            logger.info(f"Transcription: {frame.text}")
        elif isinstance(frame, TextFrame):
            self.llm_response += frame.text
        elif isinstance(frame, TTSAudioRawFrame):
            self.audio_chunks.append(frame.audio)
            self.sample_rate = frame.sample_rate
        elif isinstance(frame, TTSStoppedFrame):
            logger.info(f"LLM response: {self.llm_response}")
            self.tts_done.set()

        await self.push_frame(frame, direction)


async def process_audio_file(
    input_path: str,
    output_path: str,
    api_key: str,
    openai_api_key: str,
    mode: str = "asr",
    enable_voice_boost: bool = False,
    enhancer_weight: float = 1.0,
):
    """Process an audio file through Hecttor filter + OpenAI pipeline."""
    # Read audio file
    audio_data, sample_rate = read_audio_file(input_path)
    duration = len(audio_data) / sample_rate
    print(f"Input: {input_path}")
    print(f"  {sample_rate}Hz, {len(audio_data)} samples, {duration:.1f}s")

    # Create Hecttor filter
    hecttor_filter = HecttorFilter(
        api_key=api_key,
        mode=mode,
        enable_voice_boost=enable_voice_boost,
        enhancer_weight=enhancer_weight,
    )

    # Initialize filter
    await hecttor_filter.start(sample_rate)

    # Process audio through filter
    print(f"\nDenoising with Hecttor ({mode} mode)...")
    chunk_size = hecttor_filter._samples_per_chunk
    filtered_chunks = []

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        if len(chunk) < chunk_size:
            break
        filtered_bytes = await hecttor_filter.filter(chunk.tobytes())
        if filtered_bytes:
            filtered_chunks.append(np.frombuffer(filtered_bytes, dtype=np.int16))

    await hecttor_filter.stop()

    if not filtered_chunks:
        print("Error: No filtered audio produced")
        return

    filtered_audio = np.concatenate(filtered_chunks)
    print(f"  Filtered: {len(filtered_audio)} samples")

    # Save filtered audio for comparison
    filtered_path = output_path.replace(".wav", "_filtered.wav")
    sf.write(filtered_path, filtered_audio.astype(np.float32) / 32768.0, sample_rate)
    print(f"  Saved filtered audio: {filtered_path}")

    # Build pipecat pipeline: STT → LLM → TTS → collector
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.openai.tts import OpenAITTSService

    stt = OpenAISTTService(
        api_key=openai_api_key,
        settings=OpenAISTTService.Settings(model="whisper-1"),
    )

    llm = OpenAILLMService(
        api_key=openai_api_key,
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant. The user's message was transcribed from audio. "
                "Respond briefly and conversationally."
            ),
        ),
    )

    tts = OpenAITTSService(
        api_key=openai_api_key,
        settings=OpenAITTSService.Settings(voice="alloy"),
    )

    collector = AudioCollector()

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            stt,
            user_aggregator,
            llm,
            tts,
            collector,
            assistant_aggregator,
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams())
    runner = PipelineRunner()

    # Feed filtered audio into the pipeline as InputAudioRawFrames
    print("\nRunning pipeline: STT → LLM → TTS...")

    async def feed_audio():
        # Signal start of speech (STT needs VAD frames to trigger transcription)
        await task.queue_frame(VADUserStartedSpeakingFrame())

        # Feed audio in 20ms chunks (typical for STT)
        feed_chunk_size = int(sample_rate * 0.02)  # 20ms
        for j in range(0, len(filtered_audio), feed_chunk_size):
            audio_chunk = filtered_audio[j : j + feed_chunk_size]
            if len(audio_chunk) < feed_chunk_size:
                break
            frame = InputAudioRawFrame(
                audio=audio_chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
            )
            await task.queue_frame(frame)
            await asyncio.sleep(0.01)  # Simulate real-time pacing

        # Signal end of speech — triggers STT to transcribe the buffered audio
        await task.queue_frame(VADUserStoppedSpeakingFrame())

        # Wait for the pipeline to finish processing STT → LLM → TTS
        await collector.tts_done.wait()
        await task.queue_frame(EndFrame())

    await asyncio.gather(feed_audio(), runner.run(task))

    # Save TTS output
    if collector.audio_chunks:
        all_audio = b"".join(collector.audio_chunks)
        tts_sample_rate = collector.sample_rate or 24000

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts_sample_rate)
            wf.writeframes(all_audio)

        print(f"\nOutput saved: {output_path}")
        print(f"  {tts_sample_rate}Hz, {len(all_audio) // 2} samples")
    else:
        print("\nNo TTS audio output was produced.")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Test Hecttor filter with pipecat pipeline",
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument(
        "--mode",
        choices=["asr", "human"],
        default="asr",
        help="Enhancer mode (default: asr)",
    )
    parser.add_argument(
        "--voice-boost",
        action="store_true",
        help="Enable voice boost (human mode only)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Enhancer weight 0.0-1.0 (default: 1.0)",
    )

    args = parser.parse_args()

    api_key = os.getenv("HECTTOR_API_KEY")
    if not api_key:
        print("Error: Set HECTTOR_API_KEY environment variable")
        sys.exit(1)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    asyncio.run(
        process_audio_file(
            args.input,
            args.output,
            api_key=api_key,
            openai_api_key=openai_api_key,
            mode=args.mode,
            enable_voice_boost=args.voice_boost,
            enhancer_weight=args.weight,
        )
    )


if __name__ == "__main__":
    main()

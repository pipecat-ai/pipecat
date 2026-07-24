#!/usr/bin/env python3
"""Standalone script to test the Hecttor filter with real audio files.

This script processes an audio file through the Hecttor ASR speech enhancer and
saves the output, allowing you to compare the original and enhanced audio.

Usage:
    python test_hecttor_filter_audiofile.py input.wav output.wav
    python test_hecttor_filter_audiofile.py input.wav output.wav --model crest-2.0
    python test_hecttor_filter_audiofile.py input.wav output.wav --enhancer-weight 0.8

Requirements:
    uv add soundfile numpy
    Install the hecttor_sdk wheel manually (contact Hecttor for SDK access)
    Set the HECTTOR_API_KEY environment variable
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: uv add soundfile numpy")
    sys.exit(1)

# Add src directory to Python path for development environment
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from pipecat.audio.filters.hecttor_filter import SUPPORTED_MODELS, HecttorFilter
except ImportError as e:
    print(f"Error: Could not import the Hecttor filter: {e}")
    print("Contact Hecttor (https://hecttor.ai) for SDK access and an API key.")
    sys.exit(1)


def read_audio_file(input_path: str) -> tuple[np.ndarray, int]:
    """Read an audio file and convert it to int16 mono.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is int16 mono.
    """
    info = sf.info(input_path)
    print(f"Input format: {info.subtype}, {info.channels} channel(s), {info.samplerate} Hz")

    if info.subtype in ("PCM_16", "PCM_S16"):
        audio_data, sample_rate = sf.read(input_path, dtype="int16")
    elif info.subtype in ("FLOAT", "DOUBLE"):
        audio_data, sample_rate = sf.read(input_path, dtype="float32")
        audio_data = (audio_data * 32767).astype(np.int16)
    else:
        print(f"Error: Unsupported audio format: {info.subtype}")
        print("Supported formats: PCM_16, PCM_S16, FLOAT, DOUBLE")
        sys.exit(1)

    # Convert stereo to mono if needed, widening to int32 to avoid overflow.
    if audio_data.ndim > 1:
        print(f"Converting from {audio_data.shape[1]} channels to mono")
        audio_data = audio_data.astype(np.int32).mean(axis=1).astype(np.int16)

    duration = len(audio_data) / sample_rate
    print(f"Loaded {len(audio_data)} samples, {sample_rate} Hz, {duration:.2f} seconds")

    return audio_data, sample_rate


def audio_stats(audio_data: np.ndarray) -> dict:
    """Calculate statistics for audio data.

    Args:
        audio_data: Audio data as a numpy array.

    Returns:
        Dictionary with audio statistics.
    """
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    return {
        "min": int(audio_data.min()),
        "max": int(audio_data.max()),
        "rms": float(rms),
        "samples": len(audio_data),
    }


async def process_audio_file(
    input_path: str,
    output_path: str,
    model_name: str,
    chunk_size_ms: int,
    enhancer_weight: float | None,
) -> None:
    """Process an audio file through the Hecttor filter.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the enhanced audio.
        model_name: Hecttor ASR enhancement model to use.
        chunk_size_ms: Chunk size in milliseconds (16 or 20).
        enhancer_weight: Blend factor between original and enhanced audio.
    """
    audio_data, sample_rate = read_audio_file(input_path)

    print(f"\nInitializing Hecttor filter:")
    print(f"  - Model: {model_name}")
    print(f"  - Chunk size: {chunk_size_ms}ms")
    print(f"  - Enhancer weight: {enhancer_weight if enhancer_weight is not None else 'default'}")
    print(f"  - Sample rate: {sample_rate}Hz")

    filter_obj = HecttorFilter(
        model_name=model_name,
        chunk_size_ms=chunk_size_ms,
        enhancer_weight=enhancer_weight,
    )

    start_time = time.time()
    await filter_obj.start(sample_rate)
    print(f"\nFilter started in {(time.time() - start_time) * 1000:.2f}ms")

    try:
        # Feed audio in chunks matching the filter's configured chunk size.
        chunk_size = int(sample_rate * chunk_size_ms / 1000)
        print(f"Processing in chunks of {chunk_size} samples ({chunk_size_ms}ms)...")

        enhanced_chunks = []
        process_start = time.time()

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            # Skip the incomplete final chunk.
            if len(chunk) < chunk_size:
                print(f"Skipping incomplete final chunk: {len(chunk)} samples")
                break

            enhanced_bytes = await filter_obj.filter(chunk.tobytes())
            if enhanced_bytes:
                enhanced_chunks.append(np.frombuffer(enhanced_bytes, dtype=np.int16))

        process_duration = time.time() - process_start

        if not enhanced_chunks:
            print("Error: The filter produced no output audio.")
            sys.exit(1)

        enhanced_audio = np.concatenate(enhanced_chunks)
    finally:
        await filter_obj.stop()

    sf.write(output_path, enhanced_audio, sample_rate)

    audio_duration = len(audio_data) / sample_rate
    realtime_factor = audio_duration / process_duration if process_duration > 0 else float("inf")

    print(f"\nProcessed {audio_duration:.2f}s of audio in {process_duration:.2f}s")
    print(f"Realtime factor: {realtime_factor:.1f}x")

    original_stats = audio_stats(audio_data)
    enhanced_stats = audio_stats(enhanced_audio)

    print("\nAudio statistics:")
    print(f"  {'':<10} {'original':>12} {'enhanced':>12}")
    for key in ("samples", "min", "max", "rms"):
        original_value = original_stats[key]
        enhanced_value = enhanced_stats[key]
        if key == "rms":
            print(f"  {key:<10} {original_value:>12.1f} {enhanced_value:>12.1f}")
        else:
            print(f"  {key:<10} {original_value:>12} {enhanced_value:>12}")

    # The filter buffers audio internally, so the output is expected to be
    # slightly shorter than the input.
    dropped = len(audio_data) - len(enhanced_audio)
    if dropped > 0:
        print(f"\nNote: output is {dropped} samples shorter (filter warm-up and buffering).")

    print(f"\nSaved enhanced audio to: {output_path}")


def main():
    """Parse arguments and run the Hecttor filter over an audio file."""
    parser = argparse.ArgumentParser(
        description="Test the Hecttor filter with real audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input audio file (WAV, FLAC, OGG)")
    parser.add_argument("output", help="Output audio file (WAV, FLAC, OGG)")
    parser.add_argument(
        "--model",
        default="coda-vi-1.0",
        choices=SUPPORTED_MODELS,
        help="ASR enhancement model to use (default: coda-vi-1.0)",
    )
    parser.add_argument(
        "--chunk-size-ms",
        type=int,
        default=20,
        choices=(16, 20),
        help="Chunk size in milliseconds (default: 20)",
    )
    parser.add_argument(
        "--enhancer-weight",
        type=float,
        default=None,
        help="Blend factor 0.0-1.0 between original and enhanced audio (default: model default)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.getenv("HECTTOR_API_KEY"):
        print("Error: HECTTOR_API_KEY environment variable not set")
        print("Contact Hecttor (https://hecttor.ai) for SDK access and an API key.")
        sys.exit(1)

    asyncio.run(
        process_audio_file(
            args.input,
            args.output,
            args.model,
            args.chunk_size_ms,
            args.enhancer_weight,
        )
    )


if __name__ == "__main__":
    main()

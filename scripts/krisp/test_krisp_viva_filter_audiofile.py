#!/usr/bin/env python3
"""Standalone script to test Krisp VIVA filter with real audio files.

This script processes audio files through Krisp VIVA filter (noise reduction) and saves the output,
allowing you to compare the original and filtered audio.

Usage:
    python test_krisp_viva_filter_audiofile.py input.wav output.wav
    python test_krisp_viva_filter_audiofile.py input.wav output.wav --level 80

Requirements:
    pip install soundfile numpy pipecat-ai[krisp]
    Set KRISP_VIVA_FILTER_MODEL_PATH environment variable to point to your .kef model file
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
    from audio_file_utils import calculate_audio_stats, read_audio_file, write_audio_file
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: pip install soundfile numpy")
    sys.exit(1)

# Add src directory to Python path for development environment
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import Krisp VIVA filter
try:
    from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
    from pipecat.audio.krisp_instance import KRISP_SAMPLE_RATES
except ImportError as e:
    print(f"Error: Could not import Krisp VIVA filter: {e}")
    print("Make sure pipecat-ai is installed: pip install pipecat-ai[krisp]")
    sys.exit(1)


def validate_model_path():
    """Validate that the Krisp VIVA model path is set and exists."""
    env_var = "KRISP_VIVA_FILTER_MODEL_PATH"

    model_path = os.getenv(env_var)
    if not model_path:
        print(f"Error: {env_var} environment variable not set")
        print(f"Set it with: export {env_var}=/path/to/model.kef")
        print(f"Or in PowerShell: $env:{env_var}='C:\\path\\to\\model.kef'")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    return model_path


async def process_audio_file(
    input_path: str,
    output_path: str,
    noise_suppression_level: int = 100,
    frame_duration_ms: int = 10,
    verbose: bool = False,
) -> None:
    """Process an audio file through Krisp VIVA filter.

    Args:
        input_path: Path to input audio file
        output_path: Path to save filtered audio
        noise_suppression_level: Noise suppression level (0-100)
        frame_duration_ms: Frame duration in milliseconds (for chunking input)
        verbose: Show detailed processing information
    """
    # Read and convert audio file
    audio_data, sample_rate = read_audio_file(input_path, verbose=True)

    # Validate model path
    model_path = validate_model_path()

    # Check if sample rate is supported
    supported_rates = list(KRISP_SAMPLE_RATES.keys())
    if sample_rate not in supported_rates:
        print(f"Warning: Sample rate {sample_rate} not in supported rates {supported_rates}")
        print("Resampling may be required. Continuing anyway...")

    print(f"\nInitializing VIVA filter:")
    print(f"  - Model path: {model_path}")
    print(f"  - Noise suppression level: {noise_suppression_level}")
    print(f"  - Frame duration: {frame_duration_ms}ms (processing chunk size)")
    print(f"  - Sample rate: {sample_rate}Hz")

    # Create filter instance and measure preload time
    print("\nInitializing filter (preloading model)...")
    preload_start_time = time.time()
    filter_obj = KrispVivaFilter(
        model_path=model_path,
        noise_suppression_level=noise_suppression_level,
    )
    preload_duration = time.time() - preload_start_time
    print(f"Model preloaded in {preload_duration * 1000:.2f}ms")

    try:
        # Measure filter start time
        print("\nStarting filter...")
        start_time = time.time()
        await filter_obj.start(sample_rate)
        start_duration = time.time() - start_time
        print(f"Filter started in {start_duration * 1000:.2f}ms")

        print("\nProcessing audio...")
        filtered_samples = []
        total_frames = 0

        # Use chunk size matching filter frame duration
        chunk_size = int(sample_rate * frame_duration_ms / 1000)
        print(f"  - Chunk size: {chunk_size} samples ({frame_duration_ms}ms)")

        if verbose:
            print(f"  - Processing {len(audio_data)} samples in chunks of {chunk_size}")

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            if len(chunk) == 0:
                break

            # Skip incomplete chunks
            if len(chunk) < chunk_size:
                if verbose:
                    print(f"\n  Skipping incomplete final chunk: {len(chunk)} samples")
                break

            # Filter the chunk
            filtered_chunk_bytes = await filter_obj.filter(chunk.tobytes())

            # Collect filtered samples
            if filtered_chunk_bytes:
                filtered_chunk = np.frombuffer(filtered_chunk_bytes, dtype=np.int16)
                filtered_samples.append(filtered_chunk)
                total_frames += 1

                if verbose and total_frames <= 3:
                    print(
                        f"    Frame {total_frames}: {len(chunk)} -> {len(filtered_chunk)} samples"
                    )

            # Progress indicator
            if i % (chunk_size * 50) == 0:
                progress = (i / len(audio_data)) * 100
                print(f"  Progress: {progress:.1f}%", end="\r")

        print(f"  Progress: 100.0% - Processed {total_frames} frames")

        # Concatenate all filtered samples
        if filtered_samples:
            filtered_audio = np.concatenate(filtered_samples)
            print(f"\nFiltered audio: {len(filtered_audio)} samples")

            # Save the filtered audio
            write_audio_file(output_path, filtered_audio, sample_rate, verbose=True)

            # Calculate statistics
            original_stats = calculate_audio_stats(audio_data)
            filtered_stats = calculate_audio_stats(filtered_audio)

            print("\nAudio Statistics:")
            print(f"  Original RMS: {original_stats['rms']:.2f}")
            print(f"  Filtered RMS: {filtered_stats['rms']:.2f}")
            print(f"  RMS Ratio: {filtered_stats['rms'] / original_stats['rms']:.2f}")

            if filtered_stats["rms"] < 0.01:
                print("\n  ⚠️  WARNING: Filtered audio is very quiet or silent!")
                print("     This may indicate a processing issue.")

            print("\n✅ Processing complete!")
            print(f"   Original: {input_path}")
            print(f"   Filtered: {output_path}")
            print("\nListen to both files to compare the results.")

        else:
            print("Error: No filtered audio produced")
            sys.exit(1)

    finally:
        # Cleanup
        await filter_obj.stop()
        print("Filter stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Krisp VIVA filter with real audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_krisp_viva_audiofile.py noisy_input.wav clean_output.wav
  python test_krisp_viva_audiofile.py input.wav output.wav --level 80

Supported audio formats: WAV, FLAC, OGG, etc. (via soundfile)
Supported sample rates: 8000, 16000, 24000, 32000, 44100, 48000 Hz

Note: Set KRISP_VIVA_FILTER_MODEL_PATH environment variable to point to your .kef model file
        """,
    )

    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("output", help="Output audio file path")
    parser.add_argument(
        "--level",
        type=int,
        default=100,
        help="Noise suppression level (0-100, default: 100)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=10,
        choices=[10, 15, 20, 30, 32],
        help="Frame duration in milliseconds (default: 10)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the audio
    asyncio.run(
        process_audio_file(
            args.input,
            args.output,
            noise_suppression_level=args.level,
            frame_duration_ms=args.frame_duration,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()

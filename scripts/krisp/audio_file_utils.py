"""Utility functions for reading and writing audio files in integration tests.

This module provides consistent audio file I/O operations for test scripts,
handling format detection and conversion to int16 PCM format.
"""

import sys
from typing import Tuple

import numpy as np
import soundfile as sf


def read_audio_file(input_path: str, verbose: bool = False) -> Tuple[np.ndarray, int]:
    """Read an audio file and convert to int16 mono format.

    This function:
    - Detects the audio format from the file header
    - Reads PCM_16 files directly without conversion
    - Converts float formats by scaling to int16 range
    - Converts stereo to mono
    - Validates the audio data range

    Args:
        input_path: Path to the input audio file
        verbose: If True, print detailed format information

    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is int16 mono

    Raises:
        SystemExit: If the audio format is not supported
    """
    if verbose:
        print(f"Loading audio from: {input_path}")

    # Get audio file info to determine the format
    info = sf.info(input_path)
    if verbose:
        print(
            f"Audio file format: {info.subtype}, {info.channels} channel(s), {info.samplerate} Hz"
        )

    # Read audio data based on the source format
    if info.subtype in ["PCM_16", "PCM_S16"]:
        # File is already int16, read directly to avoid unnecessary conversion
        audio_data, sample_rate = sf.read(input_path, dtype="int16")
        if verbose:
            print("Read as int16 (native format)")
    elif info.subtype in ["FLOAT", "DOUBLE"]:
        # File is float format, read as float32 and scale to int16
        audio_data, sample_rate = sf.read(input_path, dtype="float32")
        # Convert float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
        audio_data = (audio_data * 32767).astype(np.int16)
        if verbose:
            print("Read as float32 and scaled to int16")
    else:
        print(f"Error: Unsupported audio format: {info.subtype}")
        print(f"Supported formats: PCM_16, PCM_S16, FLOAT, DOUBLE")
        sys.exit(1)

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        if verbose:
            print(f"Converting from {audio_data.shape[1]} channels to mono")
        if audio_data.dtype == np.int16:
            # For int16, convert to int32 for averaging to avoid overflow
            audio_data = audio_data.astype(np.int32).mean(axis=1).astype(np.int16)
        else:
            audio_data = audio_data.mean(axis=1).astype(np.int16)

    # Verify the audio has proper range
    audio_max = abs(audio_data.max())
    audio_min = abs(audio_data.min())
    audio_range = max(audio_max, audio_min)

    if audio_range < 100:
        print(
            f"⚠️  WARNING: Audio values are very small (max: {audio_data.max()}, min: {audio_data.min()})"
        )
        print(f"   Expected int16 range: -32768 to 32767")
        print(f"   This may indicate a format conversion issue.")
    elif verbose:
        print(f"Audio range: {audio_data.min()} to {audio_data.max()} ✓")

    if verbose:
        print(
            f"Audio info: {len(audio_data)} samples, {sample_rate} Hz, {len(audio_data) / sample_rate:.2f} seconds"
        )

    return audio_data, sample_rate


def write_audio_file(
    output_path: str, audio_data: np.ndarray, sample_rate: int, verbose: bool = False
) -> None:
    """Write audio data to a file.

    Args:
        output_path: Path to the output audio file
        audio_data: Audio data as numpy array (int16)
        sample_rate: Sample rate in Hz
        verbose: If True, print status information

    Raises:
        ValueError: If output file extension is not supported
    """
    # Validate output file extension
    valid_extensions = [".wav", ".flac", ".ogg"]
    output_ext = output_path[output_path.rfind(".") :].lower() if "." in output_path else ""

    if output_ext not in valid_extensions:
        raise ValueError(
            f"Invalid output file extension: '{output_ext}'. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )

    if verbose:
        print(f"Saving audio to: {output_path}")
        print(f"  - Format: {output_ext[1:].upper()}")
        print(f"  - Samples: {len(audio_data)}")
        print(f"  - Sample rate: {sample_rate} Hz")

    # Write the audio file
    sf.write(output_path, audio_data, sample_rate)

    if verbose:
        print(f"✓ Audio saved successfully")


def calculate_audio_stats(audio_data: np.ndarray) -> dict:
    """Calculate statistics for audio data.

    Args:
        audio_data: Audio data as numpy array

    Returns:
        Dictionary with audio statistics
    """
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    return {
        "min": int(audio_data.min()),
        "max": int(audio_data.max()),
        "mean": float(audio_data.mean()),
        "std": float(audio_data.std()),
        "rms": float(rms),
        "samples": len(audio_data),
    }

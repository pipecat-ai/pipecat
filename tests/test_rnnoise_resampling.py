import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock pyrnnoise BEFORE importing RNNoiseFilter
mock_pyrnnoise = MagicMock()
mock_rnnoise_class = MagicMock()
mock_pyrnnoise.RNNoise = mock_rnnoise_class
sys.modules["pyrnnoise"] = mock_pyrnnoise

# Now import the filter
try:
    from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
    from pipecat.frames.frames import FilterEnableFrame
except ImportError as e:
    print(f"Failed to import RNNoiseFilter: {e}")
    sys.exit(1)


async def test_rnnoise_resampling_16k_to_48k_and_back():
    print("\nStarting Resampling Test: 16kHz -> 48kHz -> 16kHz")

    # Configure Mock with buffering behavior
    processed_chunks_count = 0
    buffer = np.array([], dtype=np.int16)

    def side_effect_process_chunk(audio_samples, partial=False):
        nonlocal buffer, processed_chunks_count

        # Append new samples to buffer
        if len(audio_samples) > 0:
            buffer = np.concatenate((buffer, audio_samples))

        # Yield 480-sample chunks
        while len(buffer) >= 480:
            chunk = buffer[:480]
            buffer = buffer[480:]
            processed_chunks_count += 1

            # Simulate processing (pass through)
            # Convert int16 -> float32 [-1, 1]
            normalized = chunk.astype(np.float32) / 32768.0
            yield 0.99, normalized

    mock_rnnoise_instance = MagicMock()
    mock_rnnoise_instance.denoise_chunk.side_effect = side_effect_process_chunk
    mock_rnnoise_class.return_value = mock_rnnoise_instance

    # 1. Generate 1 second of 16kHz audio (sine wave 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    audio_bytes = audio_data.tobytes()

    print(f"Input audio: {len(audio_bytes)} bytes, {len(audio_data)} samples at {sample_rate}Hz")

    # 2. Initialize RNNoiseFilter
    rnnoise_filter = RNNoiseFilter()
    await rnnoise_filter.start(sample_rate)

    # Enable filtering
    await rnnoise_filter.process_frame(FilterEnableFrame(enable=True))

    # 3. Process audio in chunks
    chunk_size = 320  # 160 samples (10ms at 16k) * 2 bytes
    processed_audio = b""

    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i : i + chunk_size]
        result = await rnnoise_filter.filter(chunk)
        processed_audio += result

    await rnnoise_filter.stop()

    print(f"Output audio: {len(processed_audio)} bytes")
    print(f"Processed chunks (internal 480 samples): {processed_chunks_count}")

    # 4. Verify output length
    # Expect roughly same length
    # Input: 16000 samples.
    # Upsampled to 48000.
    # 48000 / 480 = 100 chunks.
    # So we expect roughly 100 calls to process_chunk.
    expected_chunks = (len(audio_data) * 48000 // sample_rate) // 480
    print(f"Expected chunks: ~{expected_chunks}")

    # Check that we actually processed something
    assert processed_chunks_count >= expected_chunks - 5, "Too few chunks processed"

    # Check output length
    assert len(processed_audio) > 0, "Output should not be empty"

    # Check length matches input (with some tolerance for buffering latency)
    # Since we don't flush the filter explicitly (no flush method in RNNoiseFilter yet),
    # some data might remain in buffers.
    # Max loss:
    # - Resampler input buffer
    # - RNNoise buffer (max 480 samples = 10ms)
    # - Resampler output buffer

    # 100ms tolerance?
    byte_tolerance = int(0.2 * sample_rate * 2)
    assert len(processed_audio) >= len(audio_bytes) - byte_tolerance, (
        f"Output too short: {len(processed_audio)} vs {len(audio_bytes)}"
    )
    assert len(processed_audio) <= len(audio_bytes) + byte_tolerance, (
        f"Output too long: {len(processed_audio)} vs {len(audio_bytes)}"
    )

    # 5. Check sample rate / pitch preservation
    # If we upsampled and downsampled correctly, the pitch should be 440Hz.
    output_data = np.frombuffer(processed_audio, dtype=np.int16)

    if len(output_data) > 2000:
        # Use a window in the middle
        start_idx = len(output_data) // 4
        end_idx = 3 * len(output_data) // 4
        segment = output_data[start_idx:end_idx]

        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), d=1 / sample_rate)
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]

        print(f"Peak frequency: {peak_freq:.2f} Hz")
        assert abs(peak_freq - 440) < 50, f"Frequency shifted significantly: {peak_freq} vs 440"

    print("Test Passed: Resampling logic verified (with mocked RNNoise).")


if __name__ == "__main__":
    asyncio.run(test_rnnoise_resampling_16k_to_48k_and_back())

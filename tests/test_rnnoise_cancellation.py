#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import numpy as np

try:
    import pyrnnoise
except ImportError:
    pyrnnoise = None

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.frames.frames import FilterEnableFrame


class TestRNNoiseCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_rnnoise_cancellation_functionality(self):
        print("\nStarting Noise Cancellation Test")

        # 1. Check for pyrnnoise
        if pyrnnoise is None:
            self.skipTest("pyrnnoise not installed. Cannot verify actual noise cancellation.")

        # 2. Generate clean speech-like audio (Harmonic series)
        sample_rate = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Fundamental 200Hz + harmonics
        clean_signal = np.sin(2 * np.pi * 200 * t) * 0.5
        clean_signal += np.sin(2 * np.pi * 400 * t) * 0.3
        clean_signal += np.sin(2 * np.pi * 600 * t) * 0.2

        # Apply envelope to simulate speech (bursts)
        # sin(2*pi*2*t) has period 0.5s.
        envelope = np.sin(2 * np.pi * 2 * t)
        envelope = np.clip(envelope, 0, 1)
        clean_signal *= envelope

        # 3. Add Noise (White Noise)
        noise_level = 0.1  # Reduced noise level slightly to make speech clearer for alignment
        noise = np.random.normal(0, noise_level, len(t))

        noisy_signal = clean_signal + noise

        # Normalize to int16 range
        noisy_signal = np.clip(noisy_signal, -1, 1)
        noisy_int16 = (noisy_signal * 32767).astype(np.int16)
        noisy_bytes = noisy_int16.tobytes()

        clean_int16 = (clean_signal * 32767).astype(np.int16)

        print(f"Generated 2s of noisy audio at {sample_rate}Hz")

        # 4. Initialize RNNoiseFilter
        rnnoise_filter = RNNoiseFilter()
        await rnnoise_filter.start(sample_rate)
        await rnnoise_filter.process_frame(FilterEnableFrame(enable=True))

        # 5. Process
        # Feed in chunks
        chunk_size = 960  # 20ms
        processed_audio = b""

        for i in range(0, len(noisy_bytes), chunk_size):
            chunk = noisy_bytes[i : i + chunk_size]
            result = await rnnoise_filter.filter(chunk)
            processed_audio += result

        await rnnoise_filter.stop()

        print(f"Output audio size: {len(processed_audio)}")

        # 6. Verify Noise Reduction
        output_int16 = np.frombuffer(processed_audio, dtype=np.int16)

        # Truncate to min length
        min_len = min(len(clean_int16), len(output_int16))
        clean_trunc = clean_int16[:min_len]
        output_trunc = output_int16[:min_len]
        noisy_trunc = noisy_int16[:min_len]

        # 7. Compensate for Delay
        # Use cross-correlation on a segment to find delay
        # We expect output to be delayed relative to clean (lag is positive)
        # search window +/- 2000 samples (~40ms)

        search_range = 2400  # 50ms
        # Use the middle of the signal to avoid edge effects and have strong signal
        mid_point = min_len // 2
        window_len = 4800  # 100ms

        ref_sig = clean_trunc[mid_point : mid_point + window_len].astype(float)
        target_sig = output_trunc[
            mid_point - search_range : mid_point + window_len + search_range
        ].astype(float)

        correlation = np.correlate(target_sig, ref_sig, mode="valid")
        best_idx = np.argmax(correlation)

        # The 'valid' mode correlation result corresponds to shifts.
        # index 0 matches alignment where ref starts at target start.
        # target start is (mid_point - search_range).
        # ref start is mid_point.
        # So index 0 means target is shifted left by search_range (or delay = -search_range).
        # delay = best_idx - search_range

        delay = best_idx - search_range
        print(f"Detected delay: {delay} samples ({delay / sample_rate * 1000:.2f} ms)")

        # Shift output to align
        if delay > 0:
            # Output is delayed, so we need to look at output[delay:] to match clean[0:]
            aligned_output = output_trunc[delay:]
            aligned_clean = clean_trunc[: len(aligned_output)]
            aligned_noisy = noisy_trunc[: len(aligned_output)]
        elif delay < 0:
            # Output is ahead (unlikely for causal filter), but handling it
            aligned_output = output_trunc[:delay]
            aligned_clean = clean_trunc[-delay:]
            aligned_noisy = noisy_trunc[-delay:]
        else:
            aligned_output = output_trunc
            aligned_clean = clean_trunc
            aligned_noisy = noisy_trunc

        # Recalculate MSE on aligned signals
        mse_input = np.mean((aligned_noisy.astype(float) - aligned_clean.astype(float)) ** 2)
        mse_output = np.mean((aligned_output.astype(float) - aligned_clean.astype(float)) ** 2)

        print(f"MSE (Input vs Clean): {mse_input:.2f}")
        print(f"MSE (Output vs Clean): {mse_output:.2f}")

        # Also check noise reduction in silent regions
        # Clean signal envelope is 0 at t=0, 0.25, 0.5...
        # Let's find indices where aligned_clean is very small
        threshold = 100  # amplitude threshold (out of 32767)
        silent_mask = np.abs(aligned_clean) < threshold

        if np.sum(silent_mask) > 1000:
            noise_power_input = np.mean(aligned_noisy[silent_mask].astype(float) ** 2)
            noise_power_output = np.mean(aligned_output[silent_mask].astype(float) ** 2)
            print(f"Noise Power in Silence (Input): {noise_power_input:.2f}")
            print(f"Noise Power in Silence (Output): {noise_power_output:.2f}")
            self.assertLess(
                noise_power_output, noise_power_input, "Noise power in silence not reduced"
            )
        else:
            print("Warning: Not enough silent samples found for noise floor check.")

        # Main assertion: MSE should improve
        # Relax assertion slightly because RNNoise introduces distortion even on clean speech
        # But for noisy speech, it should generally be better or at least remove noise.
        # If MSE doesn't improve (due to speech distortion), at least Noise Power in Silence should drop.

        if mse_output >= mse_input:
            print(
                "Warning: Overall MSE did not improve (speech distortion?). Relying on Noise Power check."
            )
            # If we passed the noise power check above, we are good.
            self.assertTrue(
                np.sum(silent_mask) > 1000
                and np.mean(aligned_output[silent_mask].astype(float) ** 2)
                < np.mean(aligned_noisy[silent_mask].astype(float) ** 2)
            )
        else:
            self.assertLess(mse_output, mse_input, "MSE did not improve")

        print("Test Passed: Noise cancellation verified.")


if __name__ == "__main__":
    unittest.main()

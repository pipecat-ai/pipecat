#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Krisp SDK Manager (singleton with reference counting)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock package version check before importing pipecat
# This allows tests to run in development mode without installed package
_version_patcher = patch("importlib.metadata.version", return_value="0.0.0-dev")
_version_patcher.start()

# Mock krisp_audio module BEFORE any pipecat imports
# This allows tests to run without krisp_audio installed
mock_krisp_audio = MagicMock()
mock_krisp_audio.SamplingRate.Sr8000Hz = 8000
mock_krisp_audio.SamplingRate.Sr16000Hz = 16000
mock_krisp_audio.SamplingRate.Sr24000Hz = 24000
mock_krisp_audio.SamplingRate.Sr32000Hz = 32000
mock_krisp_audio.SamplingRate.Sr44100Hz = 44100
mock_krisp_audio.SamplingRate.Sr48000Hz = 48000
mock_krisp_audio.FrameDuration.Fd10ms = "10ms"
mock_krisp_audio.FrameDuration.Fd15ms = "15ms"
mock_krisp_audio.FrameDuration.Fd20ms = "20ms"
mock_krisp_audio.FrameDuration.Fd30ms = "30ms"
mock_krisp_audio.FrameDuration.Fd32ms = "32ms"
mock_krisp_audio.LogLevel.Off = 0

# Mock getVersion to return a version object
mock_version = MagicMock()
mock_version.major = 1
mock_version.minor = 0
mock_version.patch = 0
mock_krisp_audio.getVersion.return_value = mock_version

# Install the mock in sys.modules before importing
sys.modules["krisp_audio"] = mock_krisp_audio

# Mock pipecat_ai_krisp package
mock_pipecat_krisp = MagicMock()
sys.modules["pipecat_ai_krisp"] = mock_pipecat_krisp
sys.modules["pipecat_ai_krisp.audio"] = MagicMock()
sys.modules["pipecat_ai_krisp.audio.krisp_processor"] = MagicMock()

# Now we can safely import
from pipecat.audio.krisp_instance import (
    KRISP_SAMPLE_RATES,
    KrispVivaSDKManager,
    int_to_krisp_sample_rate,
)


class TestKrispVivaSDKManager:
    """Tests for KrispVivaSDKManager singleton."""

    def setup_method(self):
        """Reset mocks and SDK state before each test."""
        mock_krisp_audio.reset_mock()
        mock_krisp_audio.getVersion.return_value = mock_version

        # Reset the SDK manager state for clean tests
        # We access internal state to ensure tests are isolated
        with KrispVivaSDKManager._lock:
            # Release any leftover references from previous tests
            while KrispVivaSDKManager._reference_count > 0:
                KrispVivaSDKManager._reference_count -= 1
            KrispVivaSDKManager._initialized = False

    def test_reference_counting(self):
        """Test that SDK manager properly tracks references."""
        # Initial state
        initial_count = KrispVivaSDKManager.get_reference_count()
        assert initial_count == 0

        # Acquire first reference
        KrispVivaSDKManager.acquire()
        assert KrispVivaSDKManager.get_reference_count() == initial_count + 1
        assert KrispVivaSDKManager.is_initialized()

        # Verify globalInit was called
        mock_krisp_audio.globalInit.assert_called_once()

        # Acquire second reference
        KrispVivaSDKManager.acquire()
        assert KrispVivaSDKManager.get_reference_count() == initial_count + 2
        assert KrispVivaSDKManager.is_initialized()

        # globalInit should NOT be called again
        assert mock_krisp_audio.globalInit.call_count == 1

        # Release first reference
        KrispVivaSDKManager.release()
        assert KrispVivaSDKManager.get_reference_count() == initial_count + 1
        assert KrispVivaSDKManager.is_initialized()

        # globalDestroy should NOT be called yet
        mock_krisp_audio.globalDestroy.assert_not_called()

        # Release second reference
        KrispVivaSDKManager.release()
        assert KrispVivaSDKManager.get_reference_count() == initial_count

        # globalDestroy should be called now
        mock_krisp_audio.globalDestroy.assert_called_once()

    def test_multiple_acquire_release_cycles(self):
        """Test multiple acquire/release cycles."""
        initial_count = KrispVivaSDKManager.get_reference_count()

        for i in range(3):
            KrispVivaSDKManager.acquire()
            assert KrispVivaSDKManager.get_reference_count() > initial_count
            assert KrispVivaSDKManager.is_initialized()
            KrispVivaSDKManager.release()
            assert KrispVivaSDKManager.get_reference_count() == initial_count

        # Verify globalInit/globalDestroy were called for each cycle
        assert mock_krisp_audio.globalInit.call_count == 3
        assert mock_krisp_audio.globalDestroy.call_count == 3

    def test_sdk_initialization_failure(self):
        """Test that SDK initialization failures are handled properly."""
        mock_krisp_audio.globalInit.side_effect = Exception("SDK init failed")

        with pytest.raises(Exception, match="SDK init failed"):
            KrispVivaSDKManager.acquire()

        # Verify SDK is not initialized after failure
        assert not KrispVivaSDKManager.is_initialized()
        assert KrispVivaSDKManager.get_reference_count() == 0

        # Reset the side effect for other tests
        mock_krisp_audio.globalInit.side_effect = None

    def test_release_without_acquire(self):
        """Test that release without acquire is safe."""
        initial_count = KrispVivaSDKManager.get_reference_count()

        # Release without acquire should be safe (no-op)
        KrispVivaSDKManager.release()

        assert KrispVivaSDKManager.get_reference_count() == initial_count
        mock_krisp_audio.globalDestroy.assert_not_called()

    def test_is_initialized_state(self):
        """Test is_initialized state transitions."""
        # Initially not initialized
        assert not KrispVivaSDKManager.is_initialized()

        # After acquire, should be initialized
        KrispVivaSDKManager.acquire()
        assert KrispVivaSDKManager.is_initialized()

        # After release, should not be initialized
        KrispVivaSDKManager.release()
        assert not KrispVivaSDKManager.is_initialized()


class TestSampleRateConversion:
    """Tests for sample rate conversion utilities."""

    def test_supported_sample_rates(self):
        """Test conversion of all supported sample rates."""
        for rate_hz, krisp_enum in KRISP_SAMPLE_RATES.items():
            result = int_to_krisp_sample_rate(rate_hz)
            assert result == krisp_enum

    def test_unsupported_sample_rate(self):
        """Test that unsupported rates raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            int_to_krisp_sample_rate(22050)  # Not supported

        with pytest.raises(ValueError, match="Unsupported sample rate"):
            int_to_krisp_sample_rate(96000)  # Not supported

    def test_sample_rate_error_message(self):
        """Test that error message includes helpful information."""
        try:
            int_to_krisp_sample_rate(11025)
        except ValueError as e:
            assert "11025" in str(e)
            assert "Supported rates" in str(e)
            # Should list at least some supported rates
            assert "16000" in str(e)

    def test_all_krisp_sample_rates_defined(self):
        """Test that all expected sample rates are in KRISP_SAMPLE_RATES."""
        expected_rates = [8000, 16000, 24000, 32000, 44100, 48000]
        for rate in expected_rates:
            assert rate in KRISP_SAMPLE_RATES


if __name__ == "__main__":
    unittest.main()

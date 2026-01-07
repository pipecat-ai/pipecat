"""Unit tests for Krisp SDK Manager (singleton with reference counting)."""

import pytest

# Skip entire module if krisp_audio is not available (it's a private module)
try:
    from pipecat.audio.krisp_instance import (
        KRISP_SAMPLE_RATES,
        KrispVivaSDKManager,
        int_to_krisp_sample_rate,
    )
except Exception as e:
    pytest.skip(
        f"Krisp audio module not available: {e}. "
        "This is expected in CI environments as krisp_audio is a private module.",
        allow_module_level=True,
    )


class TestKrispVivaSDKManager:
    """Tests for KrispVivaSDKManager singleton."""

    def test_reference_counting(self):
        """Test that SDK manager properly tracks references."""
        # Initial state
        initial_count = KrispVivaSDKManager.get_reference_count()

        # Acquire first reference
        KrispVivaSDKManager.acquire()
        assert KrispVivaSDKManager.get_reference_count() == initial_count + 1
        assert KrispVivaSDKManager.is_initialized()

        # Acquire second reference
        KrispVivaSDKManager.acquire()
        assert KrispVivaSDKManager.get_reference_count() == initial_count + 2
        assert KrispVivaSDKManager.is_initialized()

        # Release first reference
        KrispVivaSDKManager.release()
        assert KrispVivaSDKManager.get_reference_count() == initial_count + 1
        assert KrispVivaSDKManager.is_initialized()

        # Release second reference
        KrispVivaSDKManager.release()
        assert KrispVivaSDKManager.get_reference_count() == initial_count
        # Note: SDK might still be initialized from other tests or previous calls

    def test_multiple_acquire_release_cycles(self):
        """Test multiple acquire/release cycles."""
        initial_count = KrispVivaSDKManager.get_reference_count()

        for _ in range(3):
            KrispVivaSDKManager.acquire()
            assert KrispVivaSDKManager.get_reference_count() > initial_count
            assert KrispVivaSDKManager.is_initialized()
            KrispVivaSDKManager.release()
            assert KrispVivaSDKManager.get_reference_count() == initial_count


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

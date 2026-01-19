#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp Instance manager for pipecat audio."""

import atexit
from threading import Lock

from loguru import logger

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Krisp instance, you need to install krisp_audio.")
    raise Exception(f"Missing module: {e}")


# Mapping of sample rates (Hz) to Krisp SDK SamplingRate enums
KRISP_SAMPLE_RATES = {
    8000: krisp_audio.SamplingRate.Sr8000Hz,
    16000: krisp_audio.SamplingRate.Sr16000Hz,
    24000: krisp_audio.SamplingRate.Sr24000Hz,
    32000: krisp_audio.SamplingRate.Sr32000Hz,
    44100: krisp_audio.SamplingRate.Sr44100Hz,
    48000: krisp_audio.SamplingRate.Sr48000Hz,
}

KRISP_FRAME_DURATIONS = {
    10: krisp_audio.FrameDuration.Fd10ms,
    15: krisp_audio.FrameDuration.Fd15ms,
    20: krisp_audio.FrameDuration.Fd20ms,
    30: krisp_audio.FrameDuration.Fd30ms,
    32: krisp_audio.FrameDuration.Fd32ms,
}


def int_to_krisp_sample_rate(sample_rate: int):
    """Convert integer sample rate to Krisp SDK enum value.

    Args:
        sample_rate: Sample rate in Hz (e.g., 16000, 24000, 48000).

    Returns:
        Corresponding Krisp SDK SampleRate enum value.

    Raises:
        ValueError: If the sample rate is not supported by Krisp SDK.
    """
    if sample_rate not in KRISP_SAMPLE_RATES:
        supported_rates = ", ".join(str(rate) for rate in sorted(KRISP_SAMPLE_RATES.keys()))
        raise ValueError(
            f"Unsupported sample rate: {sample_rate} Hz. Supported rates: {supported_rates} Hz"
        )
    return KRISP_SAMPLE_RATES[sample_rate]


def int_to_krisp_frame_duration(frame_duration_ms: int):
    """Convert integer frame duration to Krisp SDK enum value.

    Args:
        frame_duration_ms: Frame duration in milliseconds (e.g., 10, 20, 30).

    Returns:
        Corresponding Krisp SDK FrameDuration enum value.

    Raises:
        ValueError: If the frame duration is not supported by Krisp SDK.
    """
    if frame_duration_ms not in KRISP_FRAME_DURATIONS:
        supported_durations = ", ".join(
            str(duration) for duration in sorted(KRISP_FRAME_DURATIONS.keys())
        )
        raise ValueError(
            f"Unsupported frame duration: {frame_duration_ms} ms. "
            f"Supported durations: {supported_durations} ms"
        )
    return KRISP_FRAME_DURATIONS[frame_duration_ms]


class KrispVivaSDKManager:
    """Singleton manager for Krisp VIVA SDK with reference counting."""

    _initialized = False
    _lock = Lock()
    _reference_count = 0

    @staticmethod
    def _log_callback(log_message, log_level):
        """Thread-safe callback for Krisp SDK logging."""
        logger.info(f"[{log_level}] {log_message}")

    @classmethod
    def acquire(cls):
        """Acquire a reference to the SDK (initializes if needed).

        Call this when creating a filter instance.

        Raises:
            Exception: If SDK initialization fails (propagated from krisp_audio)
        """
        with cls._lock:
            # Initialize SDK on first acquire
            if cls._reference_count == 0:
                try:
                    krisp_audio.globalInit("", cls._log_callback, krisp_audio.LogLevel.Off)

                    cls._initialized = True

                    SDK_VERSION = krisp_audio.getVersion()
                    logger.debug(
                        f"Krisp Audio Python SDK initialized - Version: "
                        f"{SDK_VERSION.major}.{SDK_VERSION.minor}.{SDK_VERSION.patch}"
                    )

                    # Register cleanup on program exit (failsafe)
                    atexit.register(cls._force_cleanup)

                except Exception as e:
                    cls._initialized = False
                    logger.error(f"Krisp SDK initialization failed: {e}")
                    raise

            cls._reference_count += 1
            logger.debug(f"Krisp SDK reference count: {cls._reference_count}")

    @classmethod
    def release(cls):
        """Release a reference to the SDK (destroys if last reference).

        Call this when destroying a filter instance.
        """
        with cls._lock:
            if cls._reference_count > 0:
                cls._reference_count -= 1
                logger.debug(f"Krisp SDK reference count: {cls._reference_count}")

                # Destroy SDK when last reference is released
                if cls._reference_count == 0 and cls._initialized:
                    try:
                        krisp_audio.globalDestroy()
                        cls._initialized = False
                        logger.debug("Krisp Audio SDK destroyed (all references released)")
                    except Exception as e:
                        logger.error(f"Error during Krisp SDK cleanup: {e}")
                        cls._initialized = False

    @classmethod
    def get_reference_count(cls) -> int:
        """Get the current reference count.

        Returns:
            Current number of active references to the SDK.
        """
        with cls._lock:
            return cls._reference_count

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the SDK is currently initialized.

        Returns:
            True if SDK is initialized, False otherwise.
        """
        with cls._lock:
            return cls._initialized

    @classmethod
    def _force_cleanup(cls):
        """Force cleanup on program exit (failsafe)."""
        with cls._lock:
            if cls._initialized:
                try:
                    logger.warning(
                        f"Force cleaning up Krisp SDK at exit (ref count: {cls._reference_count})"
                    )
                    krisp_audio.globalDestroy()
                    cls._initialized = False
                except Exception as e:
                    logger.error(f"Error during forced Krisp SDK cleanup: {e}")

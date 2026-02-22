#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""STT service latency defaults.

This module contains P99 time-to-final-segment (TTFS) latency values for STT
services. TTFS measures the time from when speech ends to when the final
transcript is received.

These values are used by turn stop strategies to optimize timing. Each STT
service publishes its latency via STTMetadataFrame at pipeline start.

To measure latency for your specific deployment (region, network conditions,
self-hosted instances), use the STT benchmark tool:
https://github.com/pipecat-ai/stt-benchmark

Run the TTFS benchmark for your service and configuration, then pass the
measured value to your STT service constructor:

    stt = DeepgramSTTService(api_key="...", ttfs_p99_latency=0.45)
"""

# Conservative fallback for services without measured values
DEFAULT_TTFS_P99: float = 1.0

# Measured P99 TTFS latency values (in seconds)
ASSEMBLYAI_TTFS_P99: float = 0.42
AWS_TRANSCRIBE_TTFS_P99: float = 1.90
AZURE_TTFS_P99: float = 1.80
CARTESIA_TTFS_P99: float = 0.81
DEEPGRAM_TTFS_P99: float = 0.35
DEEPGRAM_SAGEMAKER_TTFS_P99: float = 0.35
ELEVENLABS_TTFS_P99: float = 2.01
ELEVENLABS_REALTIME_TTFS_P99: float = 0.41
FAL_TTFS_P99: float = 2.07
GLADIA_TTFS_P99: float = 1.49
GOOGLE_TTFS_P99: float = 1.57
GRADIUM_TTFS_P99: float = 1.61
GROQ_TTFS_P99: float = 1.54
HATHORA_TTFS_P99: float = 0.87
OPENAI_TTFS_P99: float = 2.01
OPENAI_REALTIME_TTFS_P99: float = 1.66
SAMBANOVA_TTFS_P99: float = 2.20
SARVAM_TTFS_P99: float = 1.17
SONIOX_TTFS_P99: float = 0.35
SPEECHMATICS_TTFS_P99: float = 0.74

# These services run locally and should be replaced with measured values
NVIDIA_TTFS_P99: float = DEFAULT_TTFS_P99
WHISPER_TTFS_P99: float = DEFAULT_TTFS_P99

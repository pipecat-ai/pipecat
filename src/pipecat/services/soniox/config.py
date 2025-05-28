#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Optional

from pydantic import BaseModel

from pipecat.transcriptions.language import Language


class SonioxInputParams(BaseModel):
    """Real-time transcription settings.

    Attributes:
        languages: List of language codes to use for transcription
        code_switching: Whether to auto-detect language changes during transcription
    """

    model: str = "stt-rt-preview"

    audio_format: Optional[str] = "pcm_s16le"
    num_channels: Optional[int] = 1
    sample_rate: Optional[int] = 16000

    language_hints: Optional[List[Language]] = None
    context: Optional[str] = None

    enable_non_final_tokens: Optional[bool] = True
    max_non_final_tokens_duration_ms: Optional[int] = None

    enable_endpoint_detection: Optional[bool] = True

    client_reference_id: Optional[str] = None

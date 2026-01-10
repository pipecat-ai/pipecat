"""Simplismart Speech-to-Text service implementation using Simplismart's transcription API."""

from typing import Optional
import base64
from pipecat.transcriptions.language import Language
import httpx
from pydantic import BaseModel
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription

class SimplismartSTTService(BaseWhisperSTTService):
    """Simplismart Speech-to-Text service that generates text from audio.

    Uses Simplismart's transcription API to convert audio to text. Requires an Simplismart API key
    set via the api_key parameter
    """

    class InputParams(BaseModel):
        language: Optional[Language] = None
        task: str = "transcribe"
        without_timestamps: bool = True
        vad_model: str = "frame"
        vad_filter: bool = True
        word_timestamps: bool = False
        vad_onset: Optional[float] = 0.5
        vad_offset: Optional[float] = None
        min_speech_duration_ms: int = 0
        max_speech_duration_s: float = 30
        min_silence_duration_ms: int = 2000
        speech_pad_ms: int = 400
        diarization: bool = False
        initial_prompt: Optional[str] = None
        hotwords: Optional[str] = None
        num_speakers: int = 0
        compression_ratio_threshold: Optional[float] = 2.4
        beam_size: int = 4
        temperature: float = 0.0
        multilingual: bool = False
        max_tokens: Optional[float] = 400
        log_prob_threshold: Optional[float] = -1.0
        length_penalty: int = 1
        repetition_penalty: float = 1.01
        strict_hallucination_reduction: bool = False


    def __init__(
        self,
        *,
        model: str = "whisper",
        api_key: str,
        base_url: str,
        params: Optional[InputParams] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """
        Construct a new instance of the SimplismartSTTService.

        This initializes the Speech-to-Text service that interacts with the Simplismart API for audio transcription.

        Args:
            model: The ASR model to use for transcription (e.g., "whisper"). Defaults to "whisper".
            api_key: Bearer API key for authenticating requests to the Simplismart API.
            base_url: The base URL endpoint for the Simplismart transcription API.
            params: (Optional) An InputParams object for extra API options such as language, VAD, speaker diarization, etc.
            sample_rate: (Optional) Audio sample rate in Hz, or None for auto-detection.
            **kwargs: Additional arguments forwarded to the parent STTService class.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        self._base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
        self._params = params or SimplismartSTTService.InputParams()
        self.sr = sample_rate

    async def _transcribe(self, audio: bytes) -> Transcription:
        # Build kwargs dict with only set parameters

        audio_b64 = base64.b64encode(audio).decode("utf-8")
        payload = self._params.model_dump()
        payload["audio_data"] = audio_b64

        response = httpx.post(self._base_url, json=payload, headers = self.headers)

        response_json = response.json()
        text = response_json["transcription"]
        text = "".join(text)

        return Transcription(
            text=text
        )
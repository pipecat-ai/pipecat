#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, List, Optional, Union, Iterator

from loguru import logger
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import STTService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use nvidia riva TTS or STT, you need to `pip install pipecat-ai[riva]`. Also, set `NVIDIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class FastpitchTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[str] = "en-US"

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "English-US.Female-1",
        sample_rate_hz: int = 24000,
        # nvidia riva calls this 'function-id'
        model: str = "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate_hz, **kwargs)
        self._api_key = api_key

        self.set_model_name("fastpitch-hifigan-tts")
        self.set_voice(voice_id)

        self.voice_id = voice_id
        self.sample_rate_hz = sample_rate_hz
        self.language_code = params.language
        self.nchannels = 1
        self.sampwidth = 2
        self.quality = None

        metadata = [
            ["function-id", model],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self.service = riva.client.SpeechSynthesisService(auth)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        try:
            custom_dictionary_input = {}
            responses = self.service.synthesize_online(
                text,
                self.voice_id,
                self.language_code,
                sample_rate_hz=self.sample_rate_hz,
                audio_prompt_file=None,
                quality=20 if self.quality is None else self.quality,
                custom_dictionary=custom_dictionary_input,
            )

            for resp in responses:
                await self.stop_ttfb_metrics()

                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate_hz,
                    num_channels=self.nchannels,
                )
                yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()


class ParakeetSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        # nvidia calls this 'function-id'
        model: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key

        self.set_model_name("parakeet-ctc-1.1b-asr")

        input_device = 0
        list_devices = False
        profanity_filter = False
        automatic_punctuation = False
        no_verbatim_transcripts = False
        language_code = "en-US"
        model_name = ""
        boosted_lm_words = None
        boosted_lm_score = 4.0
        speaker_diarization = False
        diarization_max_speakers = 3
        start_history = -1
        start_threshold = -1.0
        stop_history = -1
        stop_threshold = -1.0
        stop_history_eou = -1
        stop_threshold_eou = -1.0
        custom_configuration = ""
        ssl_cert = None
        use_ssl = True
        sample_rate_hz: int = 16000
        file_streaming_chunk = 1600

        metadata = [
            ["function-id", model],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self.asr_service = riva.client.ASRService(auth)

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language_code,
                model="",
                max_alternatives=1,
                profanity_filter=profanity_filter,
                enable_automatic_punctuation=automatic_punctuation,
                verbatim_transcripts=not no_verbatim_transcripts,
                sample_rate_hertz=sample_rate_hz,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        self.config = config
        riva.client.add_word_boosting_to_config(config, boosted_lm_words, boosted_lm_score)
        riva.client.add_endpoint_parameters_to_config(
            config,
            start_history,
            start_threshold,
            stop_history,
            stop_history_eou,
            stop_threshold,
            stop_threshold_eou,
        )
        riva.client.add_custom_configuration_to_config(config, custom_configuration)

        # this doesn't work, but something like this perhaps? part 1
        self.audio = []
        self.responses = self.asr_service.streaming_response_generator(
            audio_chunks=[self.audio],
            streaming_config=self.config,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        # this doesn't work, but something like this perhaps? part 2
        self.audio.append(audio)

        # need to start to run this generator only once somewhere...
        # 'start' function doesn't work...
        # something about the event loop...
        # maybe an audio buffer... though my attempt at that didn't work either
        for response in self.responses:
            if not response.results:
                continue
            partial_transcript = ""
            for result in response.results:
                if result:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    if transcript:
                        language = None
                        if len(transcript) > 0:
                            await self.stop_ttfb_metrics()
                            if result.is_final:
                                await self.stop_processing_metrics()
                                yield TranscriptionFrame(
                                    transcript, "", time_now_iso8601(), language
                                )
                            else:
                                yield InterimTranscriptionFrame(
                                    transcript, "", time_now_iso8601(), language
                                )
        yield None

    async def _on_speech_started(self, *args, **kwargs):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

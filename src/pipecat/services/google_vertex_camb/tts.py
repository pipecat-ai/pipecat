#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import os
import time
import uuid
import warnings
import io
from typing import AsyncGenerator, List, Optional, Union, Literal

import aiohttp
import soundfile as sf
import numpy as np

from loguru import logger
from pydantic import BaseModel
from google.cloud import aiplatform

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# See .env.example for Camb.AI configuration needed

# Mars7 supported languages constant
Mars7Language = Literal["de-de", "en-gb", "en-us", "es-us", "es-es", "fr-ca", "fr-fr", "ja-jp", "ko-kr", "zh-cn"]
NUM_CHANNELS = 1

class GoogleVertexCambClient:
    def __init__(
        self,         
        project_id: str,
        location: str,
        endpoint_id: str,
        credentials_path: str
    ):
        self._project_id = project_id
        self._location = location
        self._endpoint_id = endpoint_id
        self._credentials_path = credentials_path
        self._session = None
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable must be set "
                "with path to service account key file."
            )

        try:
            aiplatform.init(
                project=project_id,
                location=location,
            )
            self.endpoint = aiplatform.Endpoint(endpoint_id)
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI client: {e}")
    async def _get_session(self):
        if self._session is None or self._session.closed:
            # Set timeout for all requests
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

class GoogleVertexCambTTSService(TTSService):
    class InputParams(BaseModel):
        reference_audio_path: str
        reference_text: Optional[str] = None
        language: Mars7Language = "en-us"

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        endpoint_id: str,
        credentials_path: str,
        sample_rate: Optional[int] = None,
        params: InputParams,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._settings = {
            "reference_audio_path": params.reference_audio_path,
            "reference_text": params.reference_text,
            "language": params.language,
            "output_format": {"sample_rate": sample_rate or 44100},  # MARS7 outputs at 44.1kHz
        }
        self._client = GoogleVertexCambClient(
            project_id=project_id,
            location=location,
            endpoint_id=endpoint_id,
            credentials_path=credentials_path,
        )
        
    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Keep the native MARS7 sample rate
        self._settings["output_format"]["sample_rate"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.close()

    def _convert_flac_to_pcm(self, flac_data: bytes) -> tuple[bytes, int]:
        """Convert FLAC audio data to raw PCM and return the sample rate."""
        try:
            # Read FLAC data from memory
            with io.BytesIO(flac_data) as flac_io:
                data, file_sample_rate = sf.read(flac_io)
                
                # Ensure mono audio
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Convert to 16-bit PCM
                if data.dtype != np.int16:
                    # Scale float data to int16 range
                    if data.dtype == np.float32 or data.dtype == np.float64:
                        data = (data * 32767).astype(np.int16)
                    else:
                        data = data.astype(np.int16)
                
                # Convert to bytes
                return data.tobytes(), int(file_sample_rate)
        except Exception as e:
            logger.error(f"Error converting FLAC to PCM: {e}")
            raise

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()
            try:
                with open(self._settings["reference_audio_path"], "rb") as f:
                    audio_ref = base64.b64encode(f.read()).decode("utf-8")
            except FileNotFoundError:
                raise ValueError(
                    f"Reference audio file not found: {self._settings['reference_audio_path']}"
                )
            except Exception as e:
                raise ValueError(f"Error reading reference audio file: {e}")
            instances =  {
                "text": text,
                "language": self._settings["language"],
                "audio_ref": audio_ref
            }

            if self._settings["reference_text"] is not None:
                instances["ref_text"] = self._settings["reference_text"]
            yield TTSStartedFrame()
            
            response = await self._client.endpoint.predict_async(instances=[instances])
            predictions = response.predictions

            if not predictions or len(predictions) == 0:
                raise RuntimeError("No audio predictions returned from the model")
            flac_data = base64.b64decode(predictions[0])
            # Convert FLAC to PCM
            try:
                audio_data, file_sample_rate = self._convert_flac_to_pcm(flac_data)
                logger.debug(f"Converted FLAC to PCM, size: {len(audio_data)} bytes, sample rate: {file_sample_rate}Hz")
                # Use the file's native sample rate (should be 44.1kHz for MARS7)
                actual_sample_rate = file_sample_rate
            except Exception as e:
                logger.error(f"Failed to convert FLAC to PCM: {e}")
                # Fall back to the original data
                audio_data = flac_data
                actual_sample_rate = self.sample_rate
                logger.warning("Using original FLAC data as fallback")

            await self.start_tts_usage_metrics(text)
            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=actual_sample_rate,
                num_channels=NUM_CHANNELS,
            )

            yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

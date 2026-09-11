"""
Orpheus TTS service for Pipecat.

Streams <custom_token_N> SNAC tokens from an Orpheus GGUF
(OpenAI-compatible /v1/completions endpoint), then decodes them to audio locally
with the SNAC vocoder. 
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from loguru import logger
from openai import AsyncOpenAI
from snac import SNAC

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

ORPHEUS_SAMPLE_RATE = 24000
PCM_INT16_MAX: int = 32767
TOKEN_PATTERN = re.compile(r"<custom_token_(\d+)>")
PROMPT_TEMPLATE: str = "<|audio|>{voice}: {text}<|eot_id|>"

# One Orpheus frame is 7 tokens.
TOKENS_PER_FRAME: int = 7
SNAC_CODEBOOK_SIZE: int = 4096
CUSTOM_TOKEN_OFFSET: int = 10

# Orpheus/SNAC decoding uses a 4-frame context window.
FRAMES_PER_CONTEXT_WINDOW: int = 4
TOKENS_PER_CONTEXT_WINDOW: int = TOKENS_PER_FRAME * FRAMES_PER_CONTEXT_WINDOW

# In the 24 kHz SNAC layout used by Orpheus, one frame is 2048 samples.
SNAC_SAMPLES_PER_FRAME: int = 2048
SNAC_SAMPLES_FRAME_TWO: int = SNAC_SAMPLES_PER_FRAME
SNAC_SAMPLES_FRAME_THREE: int = SNAC_SAMPLES_PER_FRAME*2

# We emit the second frame of the 4-frame decoded context window.
SNAC_EMIT_FRAME_INDEX: int = 1
SNAC_EMIT_START_SAMPLE:int = SNAC_EMIT_FRAME_INDEX * SNAC_SAMPLES_PER_FRAME
SNAC_EMIT_END_SAMPLE:int = SNAC_EMIT_START_SAMPLE + SNAC_SAMPLES_PER_FRAME



@staticmethod
def _turn_token_into_id(token_number: int, index: int) -> int | None:
    """Parse a '<custom_token_N>' token number into a SNAC code id."""
    codebook_index = index % TOKENS_PER_FRAME
    code_id = token_number - CUSTOM_TOKEN_OFFSET - (codebook_index * SNAC_CODEBOOK_SIZE)

    if code_id < 0 or code_id >= SNAC_CODEBOOK_SIZE:
        return None
    else:
        return code_id

class OrpheusVoice(StrEnum):
    """Supported voices for the Orpheus TTS model.

    Parameters:
        TARA: Tara voice.
        LEAH: Leah voice.
        JESS: Jess voice.
        LEO: Leo voice.
        DAN: Dan voice.
        MIA: Mia voice.
        ZAC: Zac voice.
        ZOE: Zoe voice.
    """
     
    TARA = "tara"
    LEAH = "leah"
    JESS = "jess"
    LEO = "leo"
    DAN = "dan"
    MIA = "mia"
    ZAC = "zac"
    ZOE = "zoe"

@dataclass
class OrpheusTTSSettings(TTSSettings):
    """Settings for the Orpheus TTS service.

    Parameters:
        language: Language parameter (unused by Orpheus, defaults to None).
        sample_rate: Audio sample rate in Hz.
        voice: The voice to use for synthesis.
        model: The model identifier to pass to the OpenAI-compatible endpoint.
        snac_device: Device to run the SNAC vocoder on ('auto', 'cpu', 'cuda', 'mps').
        temperature: Sampling temperature for the LLM.
        top_p: Top-p sampling parameter for the LLM.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty applied to repeated tokens (passed via extra_body).
    """
    # TTSSettings base provides: model, voice, language.
    # Orpheus does not use language directly, so default it to None.
    language: str | None = None
    sample_rate: int = ORPHEUS_SAMPLE_RATE
    voice: OrpheusVoice = OrpheusVoice.TARA
    model: str = "orpheus-3b-0.1-ft"
    snac_device: str = "auto"
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 8192
    repetition_penalty: float = 1.1



class OrpheusTTSService(TTSService):
    """Orpheus TTS service for Pipecat.

    Streams `<custom_token_N>` SNAC tokens from an Orpheus GGUF model 
    (via an OpenAI-compatible `/v1/completions` endpoint) and decodes them 
    into raw audio locally using the SNAC vocoder.
    """
    
    Settings = OrpheusTTSSettings

    
    def __init__(self, *, api_key="not-needed", base_url="http://127.0.0.1:1235/v1",
                settings: "OrpheusTTSService.Settings | None" = None, **kwargs):
        default_settings = self.Settings()          # dataclass defaults 
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(sample_rate=default_settings.sample_rate,
                        push_start_frame=True, push_stop_frames=True,
                        settings=default_settings, **kwargs)



        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        # Load the SNAC vocoder once, on the configured device.
        self._snac_device = self._resolve_snac_device(self._settings.snac_device)
        logger.debug(f"{self}: loading SNAC vocoder on '{self._snac_device}'")
        self._snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self._snac_device)

    def _resolve_snac_device(self, requested: str) -> str:
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return requested

    def can_generate_metrics(self) -> bool:
        return True

    def _format_prompt(self, text: str, voice: str) -> str:
        return PROMPT_TEMPLATE.format(voice=voice, text=text)

    # ---- Stream <custom_token_N> tokens from llama.cpp /v1/completions ----
    async def _token_stream(self, text: str, voice: str) -> AsyncGenerator[int, None]:
        
        
        prompt = self._format_prompt(text, voice)


        stream = await self._client.completions.create(
            model=self._settings.model,
            prompt=prompt,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            stream=True,
            stop=["<custom_token_3>"],
            # llama.cpp-specific; not a standard OpenAI param
            extra_body={"repeat_penalty": self._settings.repetition_penalty},
        )
        buffer = ""
        async for chunk in stream:
            if chunk.choices is not None and len(chunk.choices) > 0:
                token_text = chunk.choices[0].text
                if token_text is not None and token_text != "":
                    buffer += token_text
                    matches = list(TOKEN_PATTERN.finditer(buffer))
                    if matches:
                        for m in matches:
                            yield int(m.group(1))
                        buffer = buffer[matches[-1].end():]
                    
                    partial = buffer.rfind("<")
                    if partial != -1:
                        buffer = buffer[partial:]
                    else:
                        buffer = ""
        


    def _build_snac_codes(self, multiframe):
        if len(multiframe) < TOKENS_PER_FRAME:
            return None
        else:
            dev = self._snac_device
            num_frames = len(multiframe) // TOKENS_PER_FRAME
            frame = multiframe[:num_frames * TOKENS_PER_FRAME]

            frame_tensor = torch.tensor(frame, dtype=torch.int32, device=dev)
            frame_2d = frame_tensor.view(num_frames, TOKENS_PER_FRAME)

            #The mapping per frame of 7 tokens is fixed:
            #position 0 → codes_0
            #positions 1, 4 → codes_1
            #positions 2, 3, 5, 6 → codes_2

            codes_0 = frame_2d[:, 0].contiguous()                   # 1 value per frame
            codes_1 = frame_2d[:, [1, 4]].reshape(-1).contiguous()   # 2 values per frame
            codes_2 = frame_2d[:, [2, 3, 5, 6]].reshape(-1).contiguous()  # 4 values per frame

            codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
            return codes
    
    # ---- Decode tokens to audio via SNAC (blocking work runs in a thread) ----
    async def _decode_tokens(self, token_gen) -> AsyncGenerator[bytes, None]:
        

        buffer = []
        count = 0
        async for token_number in token_gen:
            token = _turn_token_into_id(token_number, count)
            if token is not None:
                buffer.append(token)
                count += 1
                if count % TOKENS_PER_FRAME == 0 and count >= TOKENS_PER_CONTEXT_WINDOW:
                    buffer_to_proc = buffer[-TOKENS_PER_CONTEXT_WINDOW:]
                    audio = await asyncio.to_thread(self._convert_to_audio, buffer_to_proc, count)
                    if audio is not None:
                        yield audio

    def _convert_to_audio(self, token_window: list[int], count: int) -> bytes | None:
        """
        Convert a 28-token Orpheus window into raw int16 PCM bytes.
        """
        codes = self._build_snac_codes(token_window)

        if codes is None:
            return None
        else:
            with torch.inference_mode():
                audio_hat = self._snac.decode(codes)

            # The decoder produces overlapping audio.
            # We take the stable middle section.
            audio_slice = audio_hat[:, :, SNAC_SAMPLES_FRAME_TWO:SNAC_SAMPLES_FRAME_THREE]
            audio_int16 = (audio_slice * PCM_INT16_MAX).to(torch.int16)

            return audio_int16.cpu().numpy().tobytes()
    

    # ---- Pipecat entry point ----
    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        voice = self._settings.voice

        try:
            await self.start_tts_usage_metrics(text)
            token_gen = self._token_stream(text, voice)
            async for audio in self._decode_tokens(token_gen):
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
        except Exception as e:
            logger.exception(f"{self}: error running TTS")
            yield ErrorFrame(error=f"Orpheus TTS error: {e}")
# simple_dialin.py
import argparse
import asyncio
import os
import sys
import json
import io
import wave
from datetime import datetime
import requests

from dotenv import load_dotenv
from loguru import logger

from call_connection_manager import CallConfigManager, SessionManager
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndTaskFrame, TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame,
    LLMTextFrame, UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame,
    TranscriptionFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.azure.stt import AzureSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport
from pipecat.services.tts_service import TTSService

#––– Setup –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
load_dotenv(override=True)
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Shared dict for timestamps & latencies
metrics = {}

# Event keys
STT_START = "stt_start_event_ts"
LLM_START = "llm_start_event_ts"
TTS_START = "tts_start_event_ts"
TTS_END   = "tts_end_event_ts"

def log_event(name):
    metrics[name] = datetime.now()
    logger.info(f"[{metrics[name].isoformat()}] Event: {name}")

def compute_and_log(start_key, end_key, metric_key, label):
    if start_key in metrics and end_key in metrics:
        delta = (metrics[end_key] - metrics[start_key]).total_seconds()
        metrics[metric_key] = delta
        logger.info(f"[{metrics[end_key].isoformat()}] {label.upper()}_LATENCY: {delta:.2f}s")
        return delta
    return 0

#––– Logging STT Service ––––––––––––––––––––––––––––––––––––––––––––––––
class LoggingSTTService(AzureSTTService):
    async def process_frame(self, frame, direction):
        # On user stops speaking → mark STT start
        if (direction == FrameDirection.DOWNSTREAM and
            isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame))):
            metrics.clear()
            log_event(STT_START)

        # run default STT
        await super().process_frame(frame, direction)

        # On transcription → log what user said
        if (direction == FrameDirection.UPSTREAM and
            isinstance(frame, TranscriptionFrame) and frame.text.strip()):
            logger.info(f"User said: '{frame.text.strip()}'")

#––– Logging LLM Service ––––––––––––––––––––––––––––––––––––––––––––––––
class LoggingLLMService(OpenAILLMService):
    async def process_frame(self, frame, direction):
        # On context delivered → mark LLM start and STT latency
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, OpenAILLMContextFrame):
            log_event(LLM_START)
            compute_and_log(STT_START, LLM_START, "stt_latency", "stt")
            cnt = len(frame.context.messages) if frame.context else "N/A"
            logger.info(f"LLM received context (messages: {cnt}).")

        # run default LLM
        await super().process_frame(frame, direction)

        # On LLM output → log what bot responded
        if (direction == FrameDirection.DOWNSTREAM and
            isinstance(frame, LLMTextFrame) and frame.text.strip()):
            logger.info(f"Bot responds: '{frame.text.strip()}'")

#––– Logging TTS Service –––––––––––––––––––––––––––––––––––––––––––––––
class LoggingTTSService(TTSService):
    def __init__(self, api_key=None, voice_id=None, sample_rate=24000, speed=1.0):
        super().__init__(sample_rate=sample_rate)
        self.api_key = api_key or os.getenv("SMALLEST_API_KEY")
        self.voice_id = voice_id or os.getenv("SMALLEST_VOICE_ID")
        self.speed = speed

    def _chunk_text(self, text: str, max_len: int = 500):
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text); break
            cut = text.rfind('.', 0, max_len) or max_len
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()
        return chunks

    def _synthesize(self, text: str) -> bytes:
        url = "https://waves-api.smallest.ai/api/v1/lightning-v2/get_speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        pcm = []
        for chunk in self._chunk_text(text):
            resp = requests.post(url, json={
                "text": chunk,
                "voice_id": self.voice_id,
                "add_wav_header": True,
                "sample_rate": self.sample_rate,
                "speed": self.speed,
                "language": "en",
                "consistency": 0.5,
                "similarity": 0.0,
                "enhancement": 1,
            }, headers=headers)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
            with wave.open(buf, 'rb') as wf:
                pcm.append(wf.readframes(wf.getnframes()))
        return b"".join(pcm)

    async def run_tts(self, text: str):
        # TTS start → log LLM latency
        log_event(TTS_START)
        compute_and_log(LLM_START, TTS_START, "llm_latency", "llm")
        yield TTSStartedFrame()

        # Synthesize
        audio = await asyncio.get_running_loop().run_in_executor(None, self._synthesize, text)

        # TTS end → compute TTS latency
        log_event(TTS_END)
        tts_lat = compute_and_log(TTS_START, TTS_END, "tts_latency", "tts")

        # Compute total = stt + llm + tts
        st = metrics.get("stt_latency", 0)
        ll = metrics.get("llm_latency", 0)
        total = st + ll + tts_lat
        logger.info(f"TOTAL_LATENCY (stt+llm+tts): {total:.2f}s")

        # Emit audio frames
        yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1)
        yield TTSStoppedFrame()

#––– Main ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
async def main(room_url: str, token: str, body: object):
    # parse config
    cfg = CallConfigManager.from_json_string(body) if isinstance(body, str) else CallConfigManager(body or {})
    test_mode = cfg.is_test_mode()
    dialin = cfg.get_dialin_settings()
    session_manager = SessionManager()

    base = {
        "api_url":             os.getenv("DAILY_API_URL"),
        "api_key":             os.getenv("DAILY_API_KEY"),
        "audio_in_enabled":    True,
        "audio_out_enabled":   True,
        "video_out_enabled":   False,
        "vad_analyzer":        SileroVADAnalyzer(),
        "transcription_enabled": False,
    }
    tp = DailyParams(**base) if test_mode else DailyParams(**base, dialin_settings=DailyDialinSettings(
        call_id=dialin.get("call_id"),
        call_domain=dialin.get("call_domain","")
    ))

    transport = DailyTransport(room_url, token, "Dial-in Bot", tp)

    stt = LoggingSTTService(
        api_key=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
        language=Language.EN_IN,
    )
    llm = LoggingLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
    )
    tts = LoggingTTSService(
        api_key=os.getenv("SMALLEST_API_KEY"),
        voice_id=os.getenv("SMALLEST_VOICE_ID"),
        sample_rate=int(os.getenv("SMALLEST_SAMPLE_RATE","24000")),
        speed=float(os.getenv("SMALLEST_VOICE_SPEED","1.0")),
    )

    # hang-up function
    async def terminate_call(params: FunctionCallParams):
        session_manager.call_flow_state.set_call_terminated()
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    fn = FunctionSchema(name="terminate_call", description="Hang up", properties={}, required=[])
    tools = ToolsSchema(standard_tools=[fn])
    llm.register_function("terminate_call", terminate_call)

    # build pipeline
    system = os.getenv("BOT_SYSTEM_PROMPT","You are a helpful phone agent.")
    ctx = OpenAILLMContext([cfg.create_system_message(system)], tools)
    agg = llm.create_context_aggregator(ctx)

    pipeline = Pipeline([
        transport.input(),
        stt,
        agg.user(),
        llm,
        tts,
        transport.output(),
        agg.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_join(_, p):
        await transport.capture_participant_transcription(p["id"])
        await task.queue_frames([agg.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_leave(_, p, reason):
        await task.cancel()

    await PipelineRunner().run(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u","--url",   required=True)
    parser.add_argument("-t","--token", required=True)
    parser.add_argument("-b","--body",  default="{}")
    args = parser.parse_args()

    logger.info(f"START {datetime.now().isoformat()} URL={args.url}")
    load_dotenv(override=True)
    body = json.loads(args.body) if args.body else {}
    asyncio.run(main(args.url, args.token, body))

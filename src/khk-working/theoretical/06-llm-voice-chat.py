from dailyai.services.transport.DailyTransport import DailyTransportService
from dailyai.services.llm.AzureLLMService import AzureLLMService
from dailyai.services.tts.AzureTTSService import AzureTTSService
from dailyai.services.utils import Tee
from dailyai.services.utils import ReadySoundWav

initial_prompt = "You are a helpful assistant. Introduce yourself and ask how you can be helpful."

llm_messages = [{
    "role": "system",
    "content": initial_prompt
}]


transport = None
llm = None
tts = None
mic = None
transcription = None


def main():
    global transport
    global llm
    global tts
    global mic
    global transcription

    transport = DailyTransportService()
    llm = AzureLLMService()
    tts = AzureTTSService()

    # using Moishe's combined output queue rather than an audio-only queue
    mic = transport.create_output_queue(audio=True, video=False)

    llm.set_output(Tee(tts, accumulate_assistant_messages))
    tts.set_output(mic)

    # DailyTransport implements transcription internally. we'll grab a handle to this
    # Transcription service, configure it to use silence-based endpointing, and
    # set the silence interval to 1.5 seconds
    transcription = transport.transcription_service()
    transcription.configure(endpointing_pause=1.5)

    transport.on("error", lambda e: print(e))
    transport.on("joined-meeting", llm_prompt)
    transport.start()


def llm_prompt():
    llm.run_llm(
        """You are a friendly assistant. Introduce yourself and ask how you can be helpful""")
    mic.once("audio-queue-empty", listen)


def listen():
    mic.queue(ReadySoundWav)
    # ignore any transcription results that come in before we're ready
    _ = transcription.read()
    user_text_input = transcription.read_until_silence()
    llm_messages.push({
        "role": "user",
        "content": user_text_input
    })
    llm_prompt()


def accumulate_assistant_messages(completed_inference_text):
    llm_messages.push({
        "role": "assistant",
        "content": completed_inference_text
    })

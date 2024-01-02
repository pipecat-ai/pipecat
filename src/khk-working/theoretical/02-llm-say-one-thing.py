
from dailyai.services.transport.DailyTransport import DailyTransportService
from dailyai.services.llm.AzureLLMService import AzureLLMService
from dailyai.services.tts.AzureTTSService import AzureTTSService

transport = None
mic = None
llm = None
tts = None


def main():
    global transport
    global mic
    global llm
    global tts

    transport = DailyTransportService()
    llm = AzureLLMService()
    tts = AzureTTSService()
    mic = transport.audio_queue()
    tts.set_output(mic)

    # similarly, we can tell the llm to pipe infeference output to our tts
    # service. the design idea here is that any time we call llm.run_llm()
    # we are creating a cancelable inference call, and somehow behind the
    # scenes the full pipeline from the llm to the tts service to the
    # transport's audio queue is managed in such a way as to be
    # introspectible and cancelable. also, instead of piping the
    # output to the tts service directly, we could pipe it through an
    # adapter object that does chunking or processing or whatever.
    llm.set_output(tts)

    transport.on("error", lambda e: print(e))
    transport.on("joined-meeting", make_one_inference_call)
    transport.start()


def make_one_inference_call():
    # ask our llm to say one thing, then leave
    llm.run_llm("tell me a joke about llamas")
    transport.on("audio-queue-empty", shutdown)


def shutdown():
    transport.stop()
    tts.close()

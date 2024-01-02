
from dailyai.services.transport.DailyTransport import DailyTransportService
from dailyai.services.tts.AzureTTSService import AzureTTSService


transport = None
tts = None


def main():
    global transport
    global tts

    # create a transport service object using environment variables for
    # the transport service's API key, room url, and any other configuration.
    # services can all define and document the environment variables they use.
    # services all also take an optional config object that is used instead of
    # environment variables.
    #
    # the abstract transport service APIs presumably can map pretty closely
    # to the daily-python basic API
    transport = DailyTransportService()

    # similarly, create a tts service
    tts = AzureTTSService()

    # ask the transport to create a local audio "device"/queue for
    # chunks of audio to play sequentially. the "mic" object is a handle
    # we can use to inspect and control the queue if we need to. in this
    # case we will pipe into this queue from the tts service
    mic = transport.audio_queue()
    tts.set_output(mic)

    transport.on("error", lambda e: print(e))
    transport.on("joined-meeting", say_one_thing)
    transport.start()


def say_one_thing():
    # say one thing, then leave
    tts.run_tts("hello world")
    transport.on("audio-queue-empty", shutdown)


def shutdown():
    transport.stop()
    tts.close()

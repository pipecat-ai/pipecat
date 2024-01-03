from dailyai.services.transport.DailyTransport import DailyTransportService
from dailyai.services.llm.AzureLLMService import AzureLLMService
from dailyai.services.tts.AzureTTSService import AzureTTSService
from dailyai.services.genimage.AzureDalleService import AzureDalleService
from dailyai.services.utils.AudioImageSynchronizedPair import AudioImageSynchronizedPair

transport = None
llm = None
tts = None
dalle = None
mic = None
cam = None


def main():
    global transport
    global llm
    global tts
    global dalle

    transport = DailyTransportService()
    llm = AzureLLMService()
    tts = AzureTTSService()
    dalle = AzureDalleService()

    # set up mic and cam. but don't wire up automatic output to the mic
    # and cam from our AI services because we need to manage synchronization
    # of image/speech pairings
    mic = transport.create_audio_queue()
    cam = transport.create_video_queue()

    transport.on("error", lambda e: print(e))
    transport.on("joined-meeting", narrate_calendar_images)
    transport.start()


def narrate_calendar_images():
    # let's loop over the months of the year. for each month name, we will have
    # our llm generate a description of a nice photograph for that month's page
    # in a calendar.
    #
    # then we'll take the text description and:
    #  1. turn it into speech that we send into the session as audio
    #  2. turn it into an image that we send into the session as video
    # we want the audio and video to be synchronized, so we'll use a helper
    # class to manage that.
    #
    # the first `run_llm()` call defines a lambda to process its output.
    #
    # the design idea here is that output can be piped into a function that
    # takes inference completion text as its argument. *or* output can be
    # piped into an object that has more options (maybe a callback for streaming
    # results, or a callback for inference completion, or both).
    #
    # note that we might queue up the month outputs out of order, but that's
    # okay for this demo
    #
    for month in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]:
        synchronizer = AudioImageSynchronizedPair(
            audio_output=mic, video_output=cam)
        llm.run_llm(
            f""""
            Describe a nature photograph suitable for use in a calendar,
            for the month of {month}. Include only the image description
            with no preamble.
            """,
            output=lambda inference_text: (
                dalle.generate_image(inference_text, output=synchronizer),
                tts.run_tts(inference_text, output=synchronizer)
            ),
        )


# the AudioImageSynchronizedPair class seems useful enough that I've listed
# it above as a standard utility we can import. but here's a theoretical
# implementation

class TheoreticalAudioImageSynchronizedPair:
    def __init__(self, audio_output, video_output):
        self.audio_output = audio_output
        self.video_output = video_output
        self.image = None
        self.audio = None

    def image_generation_complete(self, image):
        self.image = image
        self._maybe_send()

    def tts_complete(self, audio):
        self.audio = audio
        self._maybe_send()

    def _maybe_send(self):
        if self.image is not None and self.audio is not None:
            self.video_output.queue_frame(self.image)
            self.audio_output.queue_audio(self.audio)


def shutdown():
    transport.stop()
    tts.close()

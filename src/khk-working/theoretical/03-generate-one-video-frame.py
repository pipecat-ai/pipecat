from dailyai.services.transport.DailyTransport import DailyTransportService
from dailyai.services.genimage.AzureDalleService import AzureDalleService

dalle = None


def main():
    global dalle

    transport = DailyTransportService()
    dalle = AzureDalleService()

    # create_video_queue() could presumably take configuration parameters that
    # correspond to Daily video settings (resolution, framerate, target
    # bitrate, etc.)
    cam = transport.create_video_queue()
    dalle.set_output(cam)

    transport.on("error", lambda e: print(e))
    transport.on("joined-meeting", say_one_thing)
    transport.start()


def say_one_thing():
    # make one image, send it to the video queue, then just hang out.
    # for simplicity we have not implemented graceful shutdown :-)
    dalle.generate_image("an astronaut riding a skateboard")

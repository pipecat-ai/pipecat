#
# This demo will join a Daily meeting and send the audio from a WAV file into
# the meeting. It uses the asyncio library.
#
# Usage: python3 hold_music.py -m MEETING_URL -i FILE.wav
#

import argparse
import asyncio
import signal
import wave

from daily import *

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


class AsyncSendWavApp:
    def __init__(self, input_file_name, sample_rate, num_channels):
        self.__mic_device = Daily.create_microphone_device(
            "my-mic",
            sample_rate=sample_rate,
            channels=num_channels,
            non_blocking=True,
        )

        self.__client = CallClient()

        self.__client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "unsubscribed"}}
        )

        self.__app_error = None

        self.__start_event = asyncio.Event()
        self.__task = asyncio.get_running_loop().create_task(self.send_wav_file(input_file_name))

    async def run(self, meeting_url, meeting_token):
        (data, error) = await self.join(meeting_url, meeting_token)

        if error:
            print(f"Unable to join meeting: {error}")
            self.__app_error = error

        self.__start_event.set()

        await self.__task

    async def join(self, meeting_url, meeting_token):
        future = asyncio.get_running_loop().create_future()

        def join_completion(data, error):
            future.get_loop().call_soon_threadsafe(future.set_result, (data, error))

        self.__client.join(
            meeting_url,
            meeting_token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {"isEnabled": True, "settings": {"deviceId": "my-mic"}},
                }
            },
            completion=join_completion,
        )

        return await future

    async def leave(self):
        future = asyncio.get_running_loop().create_future()

        def leave_completion(error):
            future.get_loop().call_soon_threadsafe(future.set_result, error)

        self.__client.leave(completion=leave_completion)

        await future

        self.__client.release()

        self.__task.cancel()
        await self.__task

    async def write_frames(self, frames):
        future = asyncio.get_running_loop().create_future()

        def write_completion(count):
            future.get_loop().call_soon_threadsafe(future.set_result, count)

        self.__mic_device.write_frames(frames, completion=write_completion)

        await future

    async def send_wav_file(self, file_name):
        await self.__start_event.wait()

        if self.__app_error:
            print(f"Unable to send WAV file!")
            return

        try:
            wav = wave.open(file_name, "rb")

            sent_frames = 0
            total_frames = wav.getnframes()
            sample_rate = wav.getframerate()
            while sent_frames < total_frames:
                # Read 100ms worth of audio frames.
                frames = wav.readframes(int(sample_rate / 10))
                if len(frames) > 0:
                    await self.write_frames(frames)
                    sent_frames += sample_rate / 10
        except asyncio.CancelledError:
            pass


async def sig_handler(app):
    print("Ctrl-C detected. Exiting!")
    await app.leave()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meeting", required=True, help="Meeting URL")
    parser.add_argument("-t", "--token", required=True, help="Meeting token")
    parser.add_argument("-i", "--input", required=True, help="WAV input file")
    parser.add_argument(
        "-c", "--channels", type=int, default=NUM_CHANNELS, help="Number of channels"
    )
    parser.add_argument("-r", "--rate", type=int, default=SAMPLE_RATE, help="Sample rate")

    args = parser.parse_args()

    Daily.init()

    app = AsyncSendWavApp(args.input, args.rate, args.channels)

    loop = asyncio.get_running_loop()

    loop.add_signal_handler(signal.SIGINT, lambda *args: asyncio.create_task(sig_handler(app)))

    await app.run(args.meeting, args.token)


if __name__ == "__main__":
    asyncio.run(main())

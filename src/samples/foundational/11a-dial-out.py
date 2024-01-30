import argparse
import asyncio
import os
import wave
import requests
import time
import urllib.parse

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.queue_aggregators import LLMContextAggregator
from dailyai.services.ai_services import AIService, FrameLogger
from dailyai.queue_frame import QueueFrame, AudioQueueFrame, LLMResponseEndQueueFrame, LLMMessagesQueueFrame
from typing import AsyncGenerator

sounds = {}
sound_files = [
    'ding1.wav',
    'ding2.wav'
]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the image file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the image and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = audio_file.readframes(-1)




class OutboundSoundEffectWrapper(AIService):
    def __init__(self):
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, LLMResponseEndQueueFrame):
            yield AudioQueueFrame(sounds["ding1.wav"])
            # In case anything else up the stack needs it
            yield frame
        else:
            yield frame

class InboundSoundEffectWrapper(AIService):
    def __init__(self):
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, LLMMessagesQueueFrame):
            yield AudioQueueFrame(sounds["ding2.wav"])
            # In case anything else up the stack needs it
            yield frame
        else:
            yield frame


async def main(room_url: str, token, phone):
    global transport
    global llm
    global tts

    transport = DailyTransportService(
        room_url,
        token,
        "Respond bot",
        5,
    )
    transport.mic_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_enabled = False

    llm = AzureLLMService()
    tts = AzureTTSService()

    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport):
        await tts.say("Hi, I'm listening!", transport.send_queue)
        await transport.send_queue.put(AudioQueueFrame(sounds["ding1.wav"]))
    async def handle_transcriptions():
        messages = [
            {"role": "system", "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way."},
        ]

        tma_in = LLMContextAggregator(
            messages, "user", transport.my_participant_id
        )
        tma_out = LLMContextAggregator(
            messages, "assistant", transport.my_participant_id
        )
        out_sound = OutboundSoundEffectWrapper()
        in_sound = InboundSoundEffectWrapper()
        fl = FrameLogger("LLM Out")
        fl2 = FrameLogger("Transcription In")
        await out_sound.run_to_queue(
            transport.send_queue,
            tts.run(
                fl.run(
                    tma_out.run(
                        llm.run(
                            fl2.run(
                                in_sound.run(
                                    tma_in.run(
                                        transport.get_receive_frames()
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    @transport.event_handler("on_participant_joined")
    async def pax_joined(transport, pax):
        print(f"PARTICIPANT JOINED: {pax}")
        
    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        if (state == "joined"):
            if (phone):
                transport.dialout(phone)


    transport.transcription_settings["extra"]["punctuate"] = True

    await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-k",
        "--apikey",
        type=str,
        required=True,
        help="Daily API Key (needed to create token)",
    )

    parser.add_argument("-p", "--phone", type=str, required=False, help="A phone number to call when the bot joins the room")

    args, unknown = parser.parse_known_args()

    # Create a meeting token for the given room with an expiration 1 hour in the future.
    room_name: str = urllib.parse.urlparse(args.url).path[1:]
    expiration: float = time.time() + 60 * 60

    res: requests.Response = requests.post(
        f"https://api.staging.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {args.apikey}"},
        json={
            "properties": {"room_name": room_name, "is_owner": True, "exp": expiration}
        },
    )

    if res.status_code != 200:
        raise Exception(f"Failed to create meeting token: {res.status_code} {res.text}")

    token: str = res.json()["token"]
    asyncio.run(main(args.url, token, args.phone))

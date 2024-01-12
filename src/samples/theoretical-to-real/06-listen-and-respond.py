import argparse
import asyncio
import requests
import time
import urllib.parse

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.queue_frame import QueueFrame, FrameType

async def main(room_url:str, token):
    global transport
    global llm
    global tts

    transport = DailyTransportService(
        room_url,
        token,
        "Respond bot",
        1,
    )
    transport.mic_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_enabled = False

    llm = AzureLLMService()
    tts = AzureTTSService()

    async def handle_transcriptions():
        messages = [
            {"role": "system", "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way."},
        ]

        sentence = ""
        async for message in transport.get_transcriptions():
            if message["session_id"] == transport.my_participant_id:
                continue

            # todo: we could differentiate between transcriptions from different participants
            sentence += message["text"]
            if sentence.endswith((".", "?", "!")):
                messages.append({"role": "user", "content": sentence})
                sentence = ''

                full_response = ""
                async for response in llm.run_llm_async_sentences(messages):
                    full_response += response
                    async for audio in tts.run_tts(response):
                        transport.output_queue.put(QueueFrame(FrameType.AUDIO_FRAME, audio))

                messages.append({"role": "assistant", "content": full_response})

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

    args, unknown = parser.parse_known_args()

    # Create a meeting token for the given room with an expiration 1 hour in the future.
    room_name: str = urllib.parse.urlparse(args.url).path[1:]
    expiration: float = time.time() + 60 * 60

    res: requests.Response = requests.post(
        f"https://api.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {args.apikey}"},
        json={
            "properties": {"room_name": room_name, "is_owner": True, "exp": expiration}
        },
    )

    if res.status_code != 200:
        raise Exception(f"Failed to create meeting token: {res.status_code} {res.text}")

    token: str = res.json()["token"]

    asyncio.run(main(args.url, token))

import argparse
import asyncio
import requests
import time
import urllib.parse

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.output_queue import OutputQueueFrame, FrameType

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

    transcribed_message = ""
    transcription_timeout = None
    """
    @transport.event_handler("on_participant_joined")
    async def on_joined(transport, participant):
        if participant["id"] == transport.my_participant_id:
            return

        async for audio_chunk in tts.run_tts("If you say something, I will respond."):
            transport.output_queue.put(OutputQueueFrame(FrameType.AUDIO_FRAME, audio_chunk))

    @transport.event_handler("on_transcription_message")
    async def on_transcription_message(transport, message) -> None:
        nonlocal transcribed_message
        nonlocal transcription_timeout
        print(message)
        if message["session_id"] != transport.my_participant_id:
            transcribed_message += message['text']

            print("message received", transcribed_message)

    @transport.event_handler("on_transcription_error")
    def on_transcription_error(transport, status) -> None:
        print("transcription error", status)

    @transport.event_handler("on_transcription_started")
    def on_transcription_started(transport, status) -> None:
        print("transcription started", status)
    """
    #await transport.run()
    transport.run()


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

    args: argparse.Namespace = parser.parse_args()

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

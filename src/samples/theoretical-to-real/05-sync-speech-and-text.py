import argparse
import asyncio

from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.services.azure_ai_services import AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.open_ai_services import OpenAILLMService, OpenAIImageGenService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.daily_transport_service import DailyTransportService

async def main(room_url):
    meeting_duration_minutes = 5
    transport = DailyTransportService(
        room_url,
        None,
        "Month Narration Bot",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True
    transport.camera_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_width = 1024
    transport.camera_height = 1024

    llm = OpenAILLMService()
    tts = ElevenLabsTTSService()
    dalle = OpenAIImageGenService()

    async def get_all_audio(text):
        all_audio = bytearray()
        async for audio in tts.run_tts(text):
            all_audio.extend(audio)

        return all_audio

    async def show_month(month):
        inference_text = await llm.run_llm(
            [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit your description to 1 sentence."
                }
            ]
        )

        (image, audio) = await asyncio.gather(
            *[dalle.run_image_gen(inference_text, "1024x1024"), get_all_audio(inference_text)]
        )
        transport.output_queue.put(
            [
                OutputQueueFrame(FrameType.IMAGE_FRAME, image[1]),
                OutputQueueFrame(FrameType.AUDIO_FRAME, audio),
            ]
        )

    async def show_all_months():
        # for now just two to avoid 429s with Azure
        months: list[str] = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        await asyncio.gather(*[show_month(month) for month in months])

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        if participant["id"] == transport.my_participant_id:
            return

        await show_all_months()

        # wait for the output queue to be empty, then leave the meeting
        transport.output_queue.join()
        transport.stop()

    await transport.run()
    print("Done")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args: argparse.Namespace = parser.parse_args()

    asyncio.run(main(args.url))

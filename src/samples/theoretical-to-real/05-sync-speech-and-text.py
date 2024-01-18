import argparse
import asyncio

from asyncio.queues import Queue
import re

from dailyai.queue_frame import QueueFrame, FrameType
from dailyai.services.ai_services import SentenceAggregator
from dailyai.services.azure_ai_services import AzureLLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.open_ai_services import OpenAIImageGenService
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.fal_ai_services import FalImageGenService

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

    llm = AzureLLMService()
    dalle = FalImageGenService()
    tts = ElevenLabsTTSService(voice_id="ErXwobaYiN019PkySvjV")
    #dalle = OpenAIImageGenService(image_size="1024x1024")

    # Get a complete audio chunk from the given text. Splitting this into its own
    # coroutine lets us ensure proper ordering of the audio chunks on the send queue.
    async def get_all_audio(text):
        all_audio = bytearray()
        async for audio in tts.run_tts(text):
            all_audio.extend(audio)

        return all_audio

    async def get_month_data(month):
        image_text = ""
        tts_tasks = []
        first_sentence = True
        messages = [
            {
                "role": "system",
                "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
            }
        ]

        async for frame in SentenceAggregator().run(llm.run([QueueFrame(FrameType.LLM_MESSAGE, messages)])):
            if type(frame.frame_data) != str:
                raise Exception("LLM service requires a string for the data field")

            sentence: str = frame.frame_data
            image_text += sentence

            if first_sentence:
                sentence = f"{month}: {sentence}"
            else:
                first_sentence = False

            tts_tasks.append(get_all_audio(sentence))

        tts_tasks.insert(0, dalle.run_image_gen(image_text))

        print(f"waiting for tasks to finish for {month}")
        data = await asyncio.gather(
            *tts_tasks
        )

        print(f"done gathering tts tasks for {month}")

        return {
            "month": month,
            "text": image_text,
            "image": data[0][1],
            "audio": data[1:],
        }

    months: list[str] = [
        "January",
        "February",
        "March"]
    """
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
    """

    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport):
        # This will play the months in the order they're completed. The benefit
        # is we'll have as little delay as possible before the first month, and
        # likely no delay between months, but the months won't display in order.
        for month_data_task in asyncio.as_completed(month_tasks):
            data = await month_data_task
            await transport.send_queue.put(
                [
                    QueueFrame(FrameType.IMAGE, data["image"]),
                    QueueFrame(FrameType.AUDIO, data["audio"][0]),
                ]
            )
            for audio in data["audio"][1:]:
                await transport.send_queue.put(QueueFrame(FrameType.AUDIO, audio))

        # wait for the output queue to be empty, then leave the meeting
        transport.stop_when_done()

    month_tasks = [asyncio.create_task(get_month_data(month)) for month in months]

    await transport.run()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()

    asyncio.run(main(args.url))

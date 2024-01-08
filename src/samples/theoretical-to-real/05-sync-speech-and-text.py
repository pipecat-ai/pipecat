import asyncio

from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.services.azure_ai_services import AzureTTSService
from dailyai.services.open_ai_services import OpenAILLMService, OpenAIImageGenService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.daily_transport_service import DailyTransportService

async def main(room_url, token):
    class Sample05Transport(DailyTransportService):
        def on_participant_joined(self, participant):
            super().on_participant_joined(participant)
            asyncio.run(show_all_months())

    meeting_duration_minutes = 4
    transport = Sample05Transport(
        room_url,
        token,
        "Month Narration Bot",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True
    transport.camera_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_width = 1024
    transport.camera_height = 1024

    llm = OpenAILLMService()
    tts = DeepgramTTSService()
    dalle = OpenAIImageGenService()

    async def get_all_audio(text):
        all_audio = bytearray()
        async for audio in tts.run_tts(text):
            all_audio.extend(audio)

        return all_audio

    async def show_month(month):
        print(f"Running llm for {month}")
        inference_text = await llm.run_llm(
            [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble."
                }
            ]
        )
        print(f"got llm for {month}, {inference_text}")

        (image, audio) = await asyncio.gather(
            *[dalle.run_image_gen(inference_text, "1024x1024"), get_all_audio(inference_text)]
        )
        print(f"Got audio and video for {month}", image[0])
        transport.output_queue.put(
            [
                OutputQueueFrame(FrameType.IMAGE_FRAME, image[1]),
                OutputQueueFrame(FrameType.AUDIO_FRAME, audio),
            ]
        )

    async def show_all_months():
        # for now just two to avoid 429s with Azure
        months = [
            "January",
            "February",]
        """
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
        """
        print("gathering")
        await asyncio.gather(*[show_month(month) for month in months])
        print("done")

    try:
        transport.run()
        await asyncio.sleep(meeting_duration_minutes * 60)
    except Exception as e:
        print("Exception", e)
    finally:
        print("finally")
        transport.stop()
    print("Done")

if __name__=="__main__":
    asyncio.run(main("https://chad-hq.daily.co/howdy", None))

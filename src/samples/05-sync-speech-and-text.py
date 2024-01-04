import asyncio

from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService, AzureImageGenServiceREST
from dailyai.services.daily_transport_service import DailyTransportService

async def main(room_url, token):
    class Sample05Transport(DailyTransportService):
        def on_participant_joined(self, participant):
            super().on_participant_joined(participant)

    meeting_duration_minutes = 4
    transport = Sample05Transport(
        room_url,
        token,
        "Simple Bot",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True
    transport.camera_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_width = 1024
    transport.camera_height = 1024

    llm = AzureLLMService()
    tts = AzureTTSService()
    dalle = AzureImageGenServiceREST()

    inference_text_process = llm.run_llm(
        [
            {
                "role": "system",
                "content": f"Describe a nature photograph suitable for use in a calendar, for the month of January. Include only the image description with no preamble."
            }
        ]
    )

    try:
        transport.run()

        inference_text = await inference_text_process

        tts_iterator = tts.run_tts(inference_text)
        (image, audio) = await asyncio.gather(
            *[dalle.run_image_gen(inference_text, "1024x1024"), anext(tts_iterator)]
        )
        transport.output_queue.put(OutputQueueFrame(FrameType.IMAGE_FRAME, image[1]))
        transport.output_queue.put(OutputQueueFrame(FrameType.AUDIO_FRAME, audio))
        async for audio in tts_iterator:
            transport.output_queue.put(
                OutputQueueFrame(FrameType.AUDIO_FRAME, audio)
            )

        await asyncio.sleep(meeting_duration_minutes * 60)
    finally:
        transport.stop()
    print("Done")

if __name__=="__main__":
    asyncio.run(main("https://moishe.daily.co/Lettvins", None))

import asyncio

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.queue_aggregators import LLMAssistantContextAggregator, LLMContextAggregator, LLMUserContextAggregator
from samples.foundational.support.runner import configure

async def main(room_url: str, token):
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

    async def handle_transcriptions():
        messages = [
            {"role": "system", "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way."},
        ]

        tma_in = LLMUserContextAggregator(messages, transport.my_participant_id)
        tma_out = LLMAssistantContextAggregator(messages, transport.my_participant_id)
        await tts.run_to_queue(
            transport.send_queue,
            tma_out.run(
                llm.run(
                    tma_in.run(
                        transport.get_receive_frames()
                    )
                )
            )
        )

    transport.transcription_settings["extra"]["punctuate"] = True
    await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

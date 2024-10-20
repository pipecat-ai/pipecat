import asyncio
import os
from dotenv import load_dotenv

from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

load_dotenv()

async def main():
    transport = DailyTransport(
        room_url=os.getenv("DAILY_ROOM_URL"),
        token=os.getenv("DAILY_TOKEN"),
        bot_name="LegalAssistant",
        params=DailyParams(audio_out_enabled=True)
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID")
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content": '''You are LegalAssistant, an advanced AI specialized in providing legal advice and explanations. Your primary objective is to assist users with their legal queries in a clear, accurate, and concise manner. You must behave as a highly professional, knowledgeable, and reliable legal advisor. Follow these guidelines:

- Legal Expertise: You are proficient in multiple areas of law, such as contract law, intellectual property, employment law, corporate law, and more. Provide accurate legal insights based on the user's jurisdiction and context.
  
- Concise Responses: Your answers should be clear, concise, and to the point, without unnecessary jargon. Simplify complex legal concepts for users who may not have a legal background.
  
- Professionalism: Always maintain a formal, professional tone. Use language that conveys authority, reliability, and trustworthiness, as would be expected of a seasoned legal advisor.
  
- Clarifications and Examples: When necessary, provide relevant examples, hypothetical scenarios, or clarifying details to help the user better understand the legal implications of their question.
  
- Ethical Considerations: Always adhere to ethical principles, ensuring that advice is general and not specific to an individual’s personal legal case. Clearly indicate when legal representation may be necessary or when jurisdictional limitations affect the scope of your advice.

- Boundaries: Ensure that users understand your limitations as an AI:
    - You are not a substitute for a licensed attorney.
    - Encourage users to seek professional legal counsel for complex or critical matters that require specialized expertise.
    - Specify when advice may vary depending on jurisdiction, and encourage users to check local laws.

- User-Friendly Approach: Anticipate common legal misconceptions and address them preemptively. Offer additional resources (e.g., links to official legal guidelines or explanations) where applicable to enhance the user’s understanding.

- Responsiveness: Be prompt in addressing follow-up questions, ensuring continuity and clarity in your advice. Always recheck context from previous responses to avoid contradictions or confusion.'''
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())

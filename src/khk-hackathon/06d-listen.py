from datetime import datetime
import asyncio
import aiohttp
import os
import sys
from dailyai.conversation_wrappers import InterruptibleConversationWrapper

from dailyai.queue_frame import StartStreamQueueFrame, TranscriptionQueueFrame, TextQueueFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.fireworks_ai_services import FireworksLLMService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.ai_services import FrameLogger

from dailyai.services.fal_ai_services import FalImageGenService

from examples.foundational.support.runner import configure


command_line_prompt = ' '.join(sys.argv[1:])

system_prompt = """
You are a friendly robot character with a cartoon body with head, torso, arms, feet,
and legs.

You can change your appearance using the `change_appearance` function call.
You can add or remove items from your body, change
your color, and more. You can use function calling to change your appearance.

When changing your appearance, please create a prompt as an argument to the function.
The prompt will help the image generation model
create a new appearance for you. Include as much detail as possible. Include the
keywords "robot", "friendly", "cartoon", "smiling", "happy", "animated". 
The initial image prompt you are adding to or changing is
"A friendly cartoon robot, smiling and happy, animated."

Do not include the image model prompt in your response. The prompt must be passed to the function
as a parameter. 
"""

do_not_respond_function = {
    "name": "do_not_respond",
    "description": "Call this function when the users are not talking to the robot.",
    "parameters": {
        "type": "object",
        "properties": {
            "transcribed_text": {
                "type": "string",
                "description": "The transcribed text from the users."
            }
        }
    }
}

change_appearance_function = {
    "name": "change_appearance",
    "description": "Call this function when the users want you to change your appearance.",
    "parameters": {
        "type": "object",
        "properties": {
            "appearance": {
                "type": "string",
                "description": "The new appearance for the robot, in the form of a prompt for an generative AI diffusion model."
            }
        }
    }
}

tools = [
    {
        "type": "function",
        "function": do_not_respond_function
    },
    {
        "type": "function",
        "function": change_appearance_function
    }
]


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        context = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            duration_minutes=30,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
            # TODO-CB: Should this be VAD enabled or something?
            speaker_enabled=True,
            context=context
        )

        imagegen = FalImageGenService(
            image_size="512x512",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"))

        async def change_appearance(appearance):
            await asyncio.create_task(
                imagegen.run_to_queue(
                    transport.send_queue, [
                        TextQueueFrame(appearance)]))

        llm = FireworksLLMService(
            context=context,
            api_key=os.getenv("FIREWORKS_API_KEY"),
            model="accounts/fireworks/models/firefunction-v1",
            # TODO - how can we modify tools list on the fly?
            tools=tools,
            change_appearance=change_appearance,
            transport=transport
        )
        tts = DeepgramTTSService(aiohttp_session=session, api_key=os.getenv(
            "DEEPGRAM_API_KEY"), voice=os.getenv("DEEPGRAM_VOICE"))
        fl = FrameLogger("just outside the innermost layer")

        async def run_response(in_frame):
            await tts.run_to_queue(
                transport.send_queue,
                # tma_out.run(
                llm.run(
                    # tma_in.run(
                    fl.run(
                        [StartStreamQueueFrame(), in_frame]
                    )
                    # )
                )
                # ),
            )

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await change_appearance("A friendly cartoon robot, smiling and happy, animated.")
            return

            await tts.say("Hi, I'm listening!", transport.send_queue)
            await asyncio.sleep(1)

            await transport.receive_queue.put(UserStartedSpeakingFrame())
            await asyncio.sleep(0.1)

            transport.on_transcription_message({
                "text": command_line_prompt,
                "participantId":  "cb65b845-aac0-4fc8-987d-2e7ce3c7d8f0",
                "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            })
# putting the frame into the queue directly doesn't seem to work
#            await transport.receive_queue.put(
#                TranscriptionQueueFrame(
#                    "tell me a joke.",
#                    "cb65b845-aac0-4fc8-987d-2e7ce3c7d8f0",
#                    datetime.utcnow().strftime(
#                        '%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
#                ))
            await asyncio.sleep(0.1)
            await transport.receive_queue.put(UserStoppedSpeakingFrame())

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True

        await asyncio.gather(transport.run(), transport.run_conversation(run_response))


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

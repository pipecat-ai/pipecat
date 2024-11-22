#### run from in pipecat/examples/foundational


#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import random
import sys

import aiohttp
import requests
from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
PARTICIPANT_ID = [""]  # this will get filled in later

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
print(f"_____bigly.py * DEEPGRAM_API_KEY: {DEEPGRAM_API_KEY}")
DAILY_API_KEY = os.getenv("DAILY_BOTS_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

DAILY_ROOM_NAME = "bigs"  #### ENSURE THIS ROOM HAS enable_dialout SET (and exp)

TO = "+12097135125"  # Daily bot that pretends to be a customer asking about solar panels
# FROM_DAILY_CALLER_ID = "+13373378501"

QUESTION_AFFIRMATION = [
    "Excellent question!",
    "That's a solid question!",
    "Good question!",
    "Great question!",
    "I am glad you asked that!",
]
VOICEMAIL_EXAMPLES = [
    "We are sorry, there is noone available to take your call...",
    "Please leave a message for ",
    "Please leave your name and phone and I'll get back to you as soon as I can",
    "Your call has been forwarded to voicemail, the person you are trying to reach is not available",
    "The person you're trying to reach is not available",
    "Hey, it's (user's name) leave a message",
    "Hey you reached (user's name), please leave a message",
    "Hi/Hey I'm not available please leave a message",
    "The number you're trying to reach...",
    "I'll get back to you as soon as possible",
    "This voicemail is not receiving any messages",
    "My VM is full so If you need to reach me, please text me",
    "Leave a message and I'll call you back",
    "You've reached my cell phone, I'm not available",
    "We are sorry, there is noneone available to take your call...",
]


class DebugProcessor(FrameProcessor):
    def __init__(self, name, **kwargs):
        self._name = name
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # if not (
        #     isinstance(frame, InputAudioRawFrame)
        #     or isinstance(frame, BotSpeakingFrame)
        #     or isinstance(frame, UserStoppedSpeakingFrame)
        #     or isinstance(frame, TTSAudioRawFrame)
        #     or isinstance(frame, TextFrame)
        # ):
        #     logger.debug(f"--- DebugProcessor {self._name}: {frame} {direction}")
        # StopInterruptionFrame
        if isinstance(frame, StopInterruptionFrame) or isinstance(frame, StartInterruptionFrame):
            logger.debug(f"--- DebugProcessor {self._name}: {frame} {direction}")

        await self.push_frame(frame, direction)


async def configure(aiohttp_session: aiohttp.ClientSession):
    (url, token, _) = await configure_with_args(aiohttp_session)
    return (url, token)


async def configure_with_args(
    aiohttp_session: aiohttp.ClientSession, parser: argparse.ArgumentParser | None = None
):
    if not parser:
        parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-k",
        "--apikey",
        type=str,
        required=False,
        help="Daily API Key (needed to create an owner token for the room)",
    )

    args, unknown = parser.parse_known_args()

    #####
    url = f"https://pc-5b722fad4e9b47df8faa50cf3626267d.daily.co/{DAILY_ROOM_NAME}"
    # url = f"https://biglysales-team.daily.co/{DAILY_ROOM_NAME}"
    key = DAILY_API_KEY
    print(f"_____bigly.py * key: {key}")

    if not url:
        raise Exception(
            "No Daily room specified. use the -u/--url option from the command line, or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL."
        )

    if not key:
        raise Exception(
            "No Daily API key specified. use the -k/--apikey option from the command line, or set DAILY_API_KEY in your environment to specify a Daily API key, available from https://dashboard.daily.co/developers."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    # Create a meeting token for the given room with an expiration 1 hour in
    # the future.
    expiry_time: float = 60 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)
    print(f"_____bigly.py * token: {token}")

    return (url, token, args)


async def cancel_transport_input_task(transport_input) -> None:
    transport_input._audio_task.cancel()
    await transport_input._audio_task


def get_cartesia_static_response(text: str, voice_id: str, model_id: str, **kwargs) -> bytes:
    """
    Makes an API call to Cartesia to generate TTS audio bytes.

    Args:
        text (str): The transcript text.
        voice_id (str): The ID of the voice to be used.
        model_id (str): The model ID for the TTS request.
        **kwargs: Additional parameters like output format.

    Returns:
        bytes: The audio bytes returned by the API.

    Raises:
        ValueError: If the API request fails.
    """
    try:
        output_format = kwargs.get("output_format")
        response = requests.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                "X-API-Key": CARTESIA_API_KEY,
                "Cartesia-Version": "2024-06-10",
                "Content-Type": "application/json",
            },
            json={
                "model_id": model_id,
                "transcript": text,
                "voice": {"mode": "id", "id": voice_id},
                "output_format": output_format,
            },
        )
        response.raise_for_status()
        logger.info("Cartesia TTS response cached.")
        return response.content
    except Exception as error:
        logger.opt(exception=True).error("Error Occurred while getting Audio Bytes from Cartesia")
        raise error


async def say_agent_response(
    end_response,
    transport,
    audio_end_buffer: float,
    cancel_input_audio: bool = True,
) -> None:
    """
    Sends the agent's audio response via the specified transport and adds a buffer delay after playback.
    Optionally cancels any ongoing audio input tasks before playing the response.

    Args:
        end_response (bytes): The audio response to be played.
        transport (Union[DailyTransport, FastAPIWebsocketTransport]): The transport handling audio I/O.
        audio_end_buffer (int): Time (in seconds) to wait after playing the audio.
        cancel_input_audio (bool, optional): Whether to cancel ongoing audio input tasks. Defaults to True.

    Retries:
        Retries up to 3 times in case of an exception, with a 2-second delay between attempts.
    """
    try:
        if cancel_input_audio and transport:
            logger.info("Canceling the Audio Input Task")
            await cancel_transport_input_task(transport._input)
        await transport._output.write_raw_audio_frames(end_response)
        await asyncio.sleep(audio_end_buffer)
    except AttributeError:
        logger.info("Audio Input Transport already cancelled")
    except Exception as error:
        logger.opt(exception=True).error("Error Occurred while Uttering Agent Response.")
        raise error


async def get_meeting_dates(
    function_name: str,
    tool_call_id: str,
    _args,
    llm,
    context: OpenAILLMContext,
    result_callback,
) -> None:
    logger.info("Invoking `get_meeting_dates` tool with argument {_args}", _args=_args)
    await result_callback("""
    "1- 8AM Eastern on November 16 2024\n\n2- 9AM Eastern on November 16 2024\n\n3- 10AM Eastern on November 16 2024\n\n4- 11AM Eastern on November 16 2024\n\n5- 12PM Eastern on November 16 2024\n\n6- 1PM Eastern on November 16 2024\n\n7- 2PM Eastern on November 16 2024\n\n8- 3PM Eastern on November 16 2024\n\n9- 4PM Eastern on November 16 2024\n\n10- 5PM Eastern on November 16 2024\n\n11- 6PM Eastern on November 16 2024\n\n12- 7PM Eastern on November 16 2024\n\n"
    """)


async def transfer_call(
    function_name: str,
    tool_call_id: str,
    _args,
    llm,
    context: OpenAILLMContext,
    result_callback,
) -> None:
    print(f"_____bigly.py * transfer_call * _args: {_args}")
    await result_callback("CALL TRANSFERED")


async def voicemail(
    function_name: str, tool_call_id: str, _args, llm, context, result_callback, transport
) -> None:
    logger.info("Invoking `voicemail` tool with argument {_args}", _args=_args)
    await transport.stop_dialout(PARTICIPANT_ID[0])
    await result_callback("VOICEMAIL DETECTED")


async def get_knowledge_base(
    function_name: str, tool_call_id: str, _args, llm, context: OpenAILLMContext, result_callback
) -> None:
    logger.info("Invoking `get_knowledge_base` tool with argument {_args}", _args=_args)

    kb_call_reason = _args.get("reason", "GENERAL_QUESTION")
    if kb_call_reason == "GENERAL_QUESTION":
        # await llm.push_frame("aldkjfls oaogi8ovs(*YVSDY*( &*Tqr))")
        await llm.push_frame(TextFrame(random.choice(QUESTION_AFFIRMATION)))
    await result_callback(
        """Sure! Here are some random facts that could be associated with a solar panel company: 1. **History**: The company was founded in 2008 by a group of renewable energy enthusiasts who wanted to make solar power more accessible to homeowners and businesses. 2. **Headquarters**: Their headquarters is a net-zero energy building powered entirely by their own solar panels, showcasing their commitment to sustainability. 3. **Products**: They produce three main types of solar panels: monocrystalline, polycrystalline, and thin-film, catering to different customer needs and budgets. 4. **Innovation**: The company holds patents for advanced solar cell technology that increases efficiency by 20% compared to industry standards. 5. **Global Reach**: They have installed solar systems in over 40 countries and have manufacturing plants on three continents.6. **Community Impact**: For every 100 solar panels sold, they donate a panel to a school or community center in underprivileged areas. 7. **Workforce**: The company employs over 5,000 people, 40% of whom are in research and development roles. 8. **Recognition**: They won the “Green Energy Innovator of the Year” award in 2022 for their work on solar panels made from recycled materials. 9. **Sustainability**: Their panels are designed to last 25+ years and are 95% recyclable at the end of their life cycle. 10. **Customer Perks**: They offer a 25-year warranty and real-time monitoring systems that allow users to track energy production via an app. 11. **Mission Statement**: "Empowering the world with clean energy, one panel at a time." 12. **Energy Production**: The combined output of all their installations generates enough electricity to power over 2 million homes annually. 13. **R&D Efforts**: They are actively working on integrating solar panels into everyday items like backpacks and electric vehicles. 14. **Solar Farms**: The company has partnered with governments to develop large-scale solar farms, including one that spans over 10,000 acres. 15. **Future Goals**: By 2030, they aim to make solar power the most affordable energy source worldwide.  Would you like these customized for a specific scenario?"""
    )
    # raise RuntimeError("<><><><><>get_knowledge_base")


async def end_call(
    function_name: str,
    tool_call_id: str,
    _args,
    llm,
    context: OpenAILLMContext,
    result_callback,
    voice_provider: str,
    voice_id: str,
    transport: DailyTransport,
) -> None:
    logger.info("Invoking `end_call` tool with argument {_args}", _args=_args)
    end_call_sentence = _args.get("end_call_sentence", "Thank you for your time have a great day")
    estimated_time_end_call_sentence = 15
    logger.info(
        "Estimated End Call Sentence Time is {estimated_time_end_call_sentence}",
        estimated_time_end_call_sentence=estimated_time_end_call_sentence,
    )
    await llm.push_frame(TextFrame(end_call_sentence))
    await transport.stop_dialout(PARTICIPANT_ID[0])
    await result_callback("CALL ENDED BY ASSISTANT")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            audio_passthrough=False,
        )
        transport = DailyTransport(
            room_url,
            token,
            "Test-Bot",
            DailyParams(
                api_key=DAILY_API_KEY,
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_audio_passthrough=True,
                vad_enabled=True,
                transcription_enabled=False,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2)),
            ),
        )
        print(f"_____bigly.py * transport: {transport}")

        tts = CartesiaTTSService(
            api_key=CARTESIA_API_KEY,
            voice_id="7360f116-6306-4e9a-b487-1235f35a0f21",
        )

        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")
        llm.register_function("get_knowledge_base", get_knowledge_base)
        # llm.register_function(
        #     "end_call",
        #     partial(
        #         end_call,
        #         transport=transport,
        #         voice_provider="cartesia",
        #         voice_id="7360f116-6306-4e9a-b487-1235f35a0f21",
        #     ),
        # )
        # llm.register_function("transfer_call", transfer_call)
        # llm.register_function("voicemail", partial(voicemail, transport=transport))
        llm.register_function("get_meeting_dates", get_meeting_dates)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_knowledge_base",
                    "description": """""
                Used to find information from the knowledge base. Use this tool in the following scenarios:
                    - When the user asks questions about the company.
                    - If you need to convince the user to purchase solar panels.
                    
                IMPORTANT: ALWAYS call this tool after EVERY question the user asks.
            """,
                    "strict": True,  # type: ignore[typeddict-unknown-key]
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question the user asked.",
                            },
                            "reason": {
                                "type": "string",
                                "enum": ["CONVINCE", "GENERAL_QUESTION"],
                                "description": "Reason why you are using this tool. This can only be one of the category: `CONVINCE`, `GENERAL_QUESTION`.\
                            `CONVINCE` would be in case you are convincing the user otherwise it is `GENERAL_QUESTION`",
                            },
                        },
                        "additionalProperties": False,
                        "required": ["question", "reason"],
                    },
                },
            ),
            # ChatCompletionToolParam(
            #     type="function",
            #     function={
            #         "name": "end_call",
            #         "description": "Use this tool to end the call",
            #         "strict": True,  # type: ignore[typeddict-unknown-key]
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "end_call_sentence": {
            #                     "type": "string",
            #                     "description": "The End Call Sentence that needs to be utter by the AI",
            #                 }
            #             },
            #             "additionalProperties": False,
            #             "required": ["end_call_sentence"],
            #         },
            #     },
            # ),
            # ChatCompletionToolParam(
            #     type="function",
            #     function={
            #         "name": "transfer_call",
            #         "description": "Use this tool to transfer the call",
            #         "strict": True,  # type: ignore[typeddict-unknown-key]
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "transfer_number_sentence": {
            #                     "type": "string",
            #                     "description": "The sentence that AI needs to speak before transfering the call",
            #                 },
            #                 "transfer_number": {
            #                     "type": "string",
            #                     "description": "The number to use to which we need transfer the call",
            #                 },
            #             },
            #             "additionalProperties": False,
            #             "required": ["transfer_number", "transfer_number_sentence"],
            #         },
            #     },
            # ),
            # ChatCompletionToolParam(
            #     type="function",
            #     function={
            #         "name": "voicemail",
            #         "description": f"Use this tool if you reach voicemail. Here is some examples: {'\n'.join(VOICEMAIL_EXAMPLES)}",
            #         "parameters": {},
            #     },
            # ),
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_meeting_dates",
                    "description": "Use this tool to get the meeting dates in order to schedule a meeting date with the user. The output of this\
                        tool are slots which you need to recommend to the user. The slots are given in numerical list in ascending order\
                            (earlier slots to later slots)",
                    "parameters": {},
                },
            ),
        ]
        messages = [
            {
                "role": "system",
                "content": """You are a friendly sales person for a solar panel company. Your responses will be converted to audio. Please do not include any special characters in your response other than '!' or '?'. """,
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        dp_post_llm = DebugProcessor("post_llm")
        dp_post_tts = DebugProcessor("post_tts")

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,
                context_aggregator.user(),  # User responses
                llm,  # LLM
                dp_post_llm,  # Debug Processor
                tts,  # TTS
                dp_post_tts,  # Debug Processor
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=False,  ####
                enable_usage_metrics=False,  #####
                report_only_initial_ttfb=True,
            ),
        )

        # @transport.event_handler("on_call_state_updated")
        # async def on_call_state_updated(transport, state: str) -> None:
        #     logger.info("Call State Updated, state: {state}", state=state)

        #     async def _dialout_retry_handler() -> None:
        #         try:
        #             for i in range(3):
        #                 logger.info("Attempting a Dial-Out, Attempt: {attempt}", attempt=i + 1)
        #                 await transport.start_dialout(
        #                     {"phoneNumber": TO, "video": False}
        #                     # {"phoneNumber": TO, "callerId": FROM_DAILY_CALLER_ID, "video": False}
        #                 )
        #                 await asyncio.sleep(15)
        #                 current_participant_count = transport.participant_counts()
        #                 if current_participant_count["present"] >= 2:
        #                     return
        #             raise Exception("Unable to perform a dial-out for Daily-Co ROOM")
        #         except Exception as e:
        #             raise e

        #     async def _dialout_task_exception(task: asyncio.Task) -> None:
        #         if task.exception():
        #             await task.queue_frames([EndFrame()])
        #         else:
        #             logger.info("Dial-out completed successfully.")

        #     def _handle_dialout_completion(task: asyncio.Task) -> None:
        #         asyncio.create_task(_dialout_task_exception(task))

        #     if state == "joined":
        #         task = transport.input().get_event_loop().create_task(_dialout_retry_handler())
        #         task.add_done_callback(_handle_dialout_completion)

        # Event handler for on_first_participant_joined
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant) -> None:
            logger.info(
                "First Participant Joined with ID {participant_id}",
                participant_id=participant["id"],
            )
            # PARTICIPANT_ID[0] = participant["id"]
            # await transport.capture_participant_transcription(participant["id"])

        @transport.event_handler("on_dialout_error")
        async def on_dialout_error(transport, cdata) -> None:
            logger.info("Dial-Out error: {data}", data=cdata)
            await task.queue_frames([LLMMessagesFrame(messages)])

        # Event handler for on_dialout_answered
        # @transport.event_handler("on_dialout_answered")
        # async def on_dialout_answered(transport, cdata) -> None:
        #     logger.info("Dial-Out Answered with data as follow: {data}", data=cdata)
        #     await task.queue_frames([LLMMessagesFrame(messages)])

        # Event handler for on_participant_left
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason) -> None:
            logger.info("Following Participant Left {participant}", participant=participant)
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

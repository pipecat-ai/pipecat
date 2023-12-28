import argparse
from email.mime import image
from re import A
import requests
import time
import urllib.parse

from dailyai.async_processor.async_processor import (
    Response,
    ConversationProcessorCollection,
)
from dailyai.orchestrator import OrchestratorConfig, Orchestrator
from dailyai.message_handler.message_handler import MessageHandler
from dailyai.services.azure_ai_services import AzureTTSService
from dailyai.services.ai_services import AIServiceConfig
from dailyai.services.azure_ai_services import AzureImageGenService, AzureTTSService, AzureLLMService

def configure_ai_services() -> AIServiceConfig:
    return

def add_bot_to_room(room_url, token, expiration) -> None:

    # A simple prompt for a simple sample.
    message_handler = MessageHandler(
    """
        You are a sample bot, meant to demonstrate how to use an LLM with transcription at TTS.
        Answer user's questions and be friendly, and if you can, give some ideas about how someone
        could use a bot like you in a more in-depth way. Because your responses will be spoken,
        try to keep them short and sweet.
    """
    )

    # Use the standard Response classes for the intro and response. Do nothing while waiting
    # or saying goodbye.
    conversation_processors = ConversationProcessorCollection(
        introduction=Response,
        waiting=None,
        response=Response,
        goodbye=None,
    )

    # Use Azure services for the TTS, image generation, and LLM.
    # Note that you'll need to set the following environment variables:
    # - AZURE_SPEECH_SERVICE_KEY
    # - AZURE_SPEECH_SERVICE_REGION
    # - AZURE_CHATGPT_KEY
    # - AZURE_CHATGPT_ENDPOINT
    # - AZURE_CHATGPT_DEPLOYMENT_ID
    #
    # This demo doesn't use image generation, but if you extend it to do so,
    # you'll also need to set:
    # - AZURE_DALLE_KEY
    # - AZURE_DALLE_ENDPOINT
    # - AZURE_DALLE_DEPLOYMENT_ID

    services = AIServiceConfig(
        tts=AzureTTSService(), image=AzureImageGenService(), llm=AzureLLMService()
    )

    orchestrator_config = OrchestratorConfig(
        room_url=room_url,
        token=token,
        bot_name="Simple Bot",
        expiration=expiration,
    )

    orchestrator = Orchestrator(
        orchestrator_config,
        services,
        conversation_processors,
        message_handler,
    )
    orchestrator.start()

    # When the orchestrator's done, we need to shut it down,
    # and the various services and handlers we've created.
    orchestrator.stop()
    message_handler.shutdown()

    services.tts.close()
    services.image.close()
    services.llm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument("-u", "--url", type=str, required=True, help="URL of the Daily room")
    parser.add_argument(
        "-k", "--apikey", type=str, required=True, help="Daily API Key (needed to create token)"
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
        raise Exception(f'Failed to create meeting token: {res.status_code} {res.text}')

    token: str = res.json()['token']

    add_bot_to_room(args.url, token, expiration)

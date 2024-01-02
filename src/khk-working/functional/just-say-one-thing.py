import argparse
import time

from dailyai.orchestrator import OrchestratorConfig, Orchestrator
from dailyai.message_handler.message_handler import MessageHandler
from dailyai.services.ai_services import AIServiceConfig
from dailyai.services.azure_ai_services import AzureTTSService, AzureLLMService


# For now, use Azure service for the TTS. Todo: make tts service
# and tts args (like which voice to use) configurable via command
# line arguments.
# Need the following environment variables:
# - AZURE_SPEECH_SERVICE_KEY
# - AZURE_SPEECH_SERVICE_REGION


def add_bot_to_room(room_url, text) -> None:
    message_handler = MessageHandler(
        "Respond with only the following text: " + text)

    services = AIServiceConfig(
        tts=AzureTTSService(), image=None, llm=AzureLLMService()
    )

    orchestrator_config = OrchestratorConfig(
        room_url=room_url,
        # todo: token should be optional
        token=None,
        bot_name="Minimal Speaking Bot",
        # todo: expiration should be optional
        expiration=time.time() + 10
    )

    orchestrator = Orchestrator(
        orchestrator_config,
        services,
        message_handler,
    )

    orchestrator.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say one phrase and exit")
    parser.add_argument("-u", "--url", type=str,
                        required=True, help="URL of the Daily room")

    parser.add_argument(
        "-t", "--text", type=str, required=True, help="text to send into the session as speech"
    )

    args: argparse.Namespace = parser.parse_args()

    add_bot_to_room(args.url, args.text)

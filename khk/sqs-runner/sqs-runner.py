# - block while running bot
# - watchdog timer
# - build docker and deploy to eks

import boto3
import json
import subprocess
import signal
import os
import time

from pydantic import BaseModel, ValidationError
from typing import Optional

from bot import BotSettings

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomObject, DailyRoomProperties, DailyRoomParams

from dotenv import load_dotenv
load_dotenv(override=True)

# SQS queue URL
QUEUE_URL = 'https://sqs.us-west-2.amazonaws.com/955740203061/khk-sqs-launch-day-demos'

# The program to be spawned
SUBPROCESS_PROGRAM = 'your_subprocess_program.py'

# Timeout in seconds
TIMEOUT = 620

# ------------ Configuration ------------ #

MAX_SESSION_TIME = 5 * 60  # 5 minutes
REQUIRED_ENV_VARS = ['DAILY_API_KEY', 'CARTESIA_API_KEY']

daily_rest_helper = DailyRESTHelper(
    os.getenv("DAILY_API_KEY", ""),
    os.getenv("DAILY_API_URL", 'https://api.daily.co/v1'))


class RunnerSettings(BaseModel):
    prompt: Optional[
        str] = "You are a fast, low-latency chatbot. Your goal is to demonstrate voice-driven AI capabilities at human-like speeds. The technology powering you is Daily for transport, Groq for AI inference, Llama 3 (70-B version) LLM, and Deepgram for speech-to-text and text-to-speech. You are running on servers in Oregon. Respond to what the user said in a creative and helpful way, but keep responses short and legible. Ensure responses contain only words. Check again that you have not included special characters other than '?' or '!'."
    deepgram_voice: Optional[str] = os.getenv("DEEPGRAM_VOICE")
    openai_model: Optional[str] = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    test: Optional[bool] = None

# ----------------- API ----------------- #


def setup_sqs():
    """Set up the SQS client."""
    return boto3.client(
        'sqs',
        region_name='us-west-2',
        # use an iam user instead of role because passing a role into an eks pod requires
        # adding an add-on and we don't want to change the eks cluster configuration if we
        # can avoid it
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )


def receive_message(sqs):
    """Receive a message from the SQS queue."""
    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20  # Long polling
    )

    messages = response.get('Messages', [])
    if messages:
        return messages[0]
    return None


def delete_message(sqs, receipt_handle):
    """Delete a message from the queue after processing."""
    sqs.delete_message(
        QueueUrl=QUEUE_URL,
        ReceiptHandle=receipt_handle
    )


# def run_subprocess(message_body):
#     """Run the subprocess with the message data."""
#     process = subprocess.Popen(['python', SUBPROCESS_PROGRAM, message_body])

#     start_time = time.time()
#     while time.time() - start_time < TIMEOUT:
#         if process.poll() is not None:
#             # Process has finished
#             return True
#         time.sleep(1)

#     # If we're here, the process has timed out
#     os.kill(process.pid, signal.SIGKILL)
#     return False

def start_bot(room_url):
    runner_settings = RunnerSettings()

    # Check passed room URL exists, we should assume that it already has a sip set up
    try:
        room: DailyRoomObject = daily_rest_helper.get_room_from_url(room_url)
    except Exception:
        raise HTTPException(
            status_code=500, detail=f"Room not found: {room_url}")

    # Give the agent a token to join the session
    token = daily_rest_helper.get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room_url}")

    # Spawn a new agent, and join the user session
    try:
        bot_settings = BotSettings(
            room_url=room.url,
            room_token=token,
            prompt=runner_settings.prompt,
            deepgram_voice=runner_settings.deepgram_voice,
            openai_model=runner_settings.openai_model,
            openai_api_key=runner_settings.openai_api_key,
        )
        bot_settings_str = bot_settings.model_dump_json(exclude_none=True)

        subprocess.Popen(
            [f"python3 -m bot -s '{bot_settings_str}'"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")


def main():
    sqs = setup_sqs()

    while True:
        message = receive_message(sqs)
        if message:
            delete_message(sqs, message['ReceiptHandle'])

            message_body = json.loads(message['Body'])
            print(f"Received message. {message_body}")

            start_bot(message_body['url'])

            # success = run_subprocess(message_body)
            # if success:
            #    print("Subprocess completed successfully.")
            # else:
            #     print("Subprocess timed out and was terminated.")

        else:
            print("No messages received. Continuing to poll...")


if __name__ == "__main__":
    # Check environment variables
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    try:
        main()
    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")

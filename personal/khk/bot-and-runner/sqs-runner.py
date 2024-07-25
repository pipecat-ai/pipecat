#
# requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
#

import boto3
import json
import subprocess
import signal
import os
import time

from pydantic import BaseModel, ValidationError
from typing import Optional

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomObject, DailyRoomProperties, DailyRoomParams

from dotenv import load_dotenv
load_dotenv(override=True)

# SQS queue URL
QUEUE_URL = os.environ.get('SQS_QUEUE_URL')

# The program to be spawned
SUBPROCESS_PROGRAM = 'your_subprocess_program.py'


def escape_bash_arg(s):
    return "'" + s.replace("'", "'\\''") + "'"

# ------------ Configuration ------------ #


MAX_SESSION_TIME = 5 * 60  # 5 minutes
REQUIRED_ENV_VARS = [
    'DAILY_API_KEY',
    'CARTESIA_API_KEY',
    'SQS_QUEUE_URL',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY']

print("DAILY_API_KEY", os.getenv("DAILY_API_KEY"))
daily_rest_helper = DailyRESTHelper(
    os.getenv("DAILY_API_KEY", ""),
    os.getenv("DAILY_API_URL", 'https://api.daily.co/v1'))

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


def run_bot(msg):
    try:
        print(f"Getting room from URL: {'https://rtvi.daily.co/' + msg['room']}")
        room: DailyRoomObject = daily_rest_helper.get_room_from_url(
            'https://rtvi.daily.co/' + msg['room'])
    except Exception:
        raise HTTPException(
            status_code=500, detail=f"Room not found: {msg['room']}")

    # Give the agent a token to join the session
    token = daily_rest_helper.get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room.url}")

    # Spawn a new agent, and join the user session
    try:
        bot_settings_str = json.dumps(msg['config'])

        print(f"Starting bot with settings: {room.url}, {token}, {bot_settings_str}")

        process = subprocess.Popen(
            [f"python3 -m bot -u {room.url} -t {token} -c {escape_bash_arg(bot_settings_str)}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)))

        start_time = time.time()
        while time.time() - start_time < (MAX_SESSION_TIME + 10):
            if process.poll() is not None:
                # process has finished
                print("BOT EXITED BEFORE TIMEOUT")
                return
            time.sleep(1)
        # process did not exit. need to kill -9 it
        print("KILLING BOT PROCESS")
        os.kill(process.pid, signal.SIGKILL)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")


def main():
    print("sqs-runner.py - started.")
    sqs = setup_sqs()

    while True:
        print("sqs-runner.py - polling")
        message = receive_message(sqs)
        if message:
            delete_message(sqs, message['ReceiptHandle'])

            message_body = json.loads(message['Body'])
            print(f"Received message. {message_body}")

            run_bot(message_body)


if __name__ == "__main__":
    # Check environment variables
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    try:
        main()
    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")

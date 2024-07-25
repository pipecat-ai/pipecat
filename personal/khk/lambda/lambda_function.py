import json
import os
import requests
import boto3
import time
from typing import Dict, Any


def get_sqs_client():
    if __name__ == "__main__":
        # for local testing, load temporary credentials. see notes.md
        try:
            with open('assume_role_output.json', 'r') as f:
                credentials = json.load(f)['Credentials']

            session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            return session.client('sqs', region_name='us-west-2')
        except FileNotFoundError:
            print("Warning: assume_role_output.json not found. Using default credentials.")
            return boto3.client('sqs')
    else:
        # When running in Lambda, use the role attached to the function
        return boto3.client('sqs')


sqs = get_sqs_client()
DAILY_API_KEY = os.environ.get('DAILY_API_KEY')
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')


def create_daily_room() -> Dict[str, Any]:
    if not DAILY_API_KEY:
        raise ValueError("DAILY_API_KEY environment variable is not set")

    url = "https://api.daily.co/v1/rooms"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DAILY_API_KEY}"
    }

    exp = int(time.time()) + 180
    payload = {
        "properties": {
            "exp": exp,
            "eject_at_room_exp": True
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def create_token(room_name: str) -> Dict[str, Any]:
    if not DAILY_API_KEY:
        raise ValueError("DAILY_API_KEY environment variable is not set")

    url = f"https://api.daily.co/v1/meeting-tokens"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DAILY_API_KEY}"
    }

    exp = int(time.time()) + 180
    payload = {
        "properties": {
            "exp": exp,
            "room_name": room_name
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def send_to_sqs(room_info: Dict[str, Any]) -> None:
    if not SQS_QUEUE_URL:
        raise ValueError("SQS_QUEUE_URL environment variable is not set")

    if sqs:
        sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(room_info)
        )
        print(f"Message sent to SQS queue: {SQS_QUEUE_URL}")
    else:
        print(f"Simulated sending message to SQS queue: {SQS_QUEUE_URL}")
        print(f"Message body: {json.dumps(room_info, indent=2)}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        path = event.get('rawPath', '')
        path = path.rstrip('/')

        if path == '/authenticate':
            print("Creating Daily room...")
            daily_room = create_daily_room()
            print(f"Daily room created: {json.dumps(daily_room, indent=2)}")

            token = create_token(daily_room['name'])
            print(f"Meeting token created: {json.dumps(token, indent=2)}")

            return {
                'statusCode': 200,
                'body': json.dumps({"room": daily_room['name'], "token": token['token']})
            }
        elif path == '/start_bot':
            body = json.loads(event['body'])

            print("Sending room info to SQS...")

            send_to_sqs(body)
            return {
                'statusCode': 200,
                'body': json.dumps({"status": "healthy"})
            }
        else:
            return {
                'statusCode': 200,
                'body': json.dumps({"success": True})
            }
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }


if __name__ == "__main__":
    # Simulate the Lambda event and context
    # event = {}
    event = {
        # "rawPath": "/authenticate"
        "rawPath": "/start_bot",
        "body": json.dumps({"room": "test-room", "token": "test"})
    }
    context = None

    result = lambda_handler(event, context)
    print(f"Lambda handler result: {json.dumps(result, indent=2)}")

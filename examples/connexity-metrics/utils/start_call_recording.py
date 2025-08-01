import os

from twilio.rest import Client

account_sid = os.environ.get("TWILIO_ACCOUNT_ID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)


def start_call_recording(call_sid):
    try:
        recording = client.calls(call_sid).recordings.create()
        print(f"Recording started with SID: {recording.sid}")
        return recording.sid
    except Exception as e:
        print(f"Error starting recording: {e}")
        return None

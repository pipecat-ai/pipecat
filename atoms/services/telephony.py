from typing import Any, Dict

import httpx
from config import settings
from loguru import logger


class TelephonyService:
    """Base telephony service."""

    async def make_outbound_call(self, to: str, from_phone: str = None) -> str:
        """Make an outbound call."""
        raise NotImplementedError


class TwilioService(TelephonyService):
    """Twilio telephony service."""

    def __init__(self):
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.phone_number = settings.twilio_phone_number

    async def make_outbound_call(self, to: str, from_phone: str = None) -> str:
        """Make an outbound Twilio call."""
        from_number = from_phone or self.phone_number

        data = {
            "Twiml": f'<Response><Connect><Stream url="wss://{settings.server_base_url}/ws/twilio" /></Connect></Response>',
            "To": to,
            "From": from_number,
            "Record": True,
            "RecordingStatusCallback": f"https://{settings.server_base_url}/webhooks/twilio/recording",
        }

        auth = httpx.BasicAuth(self.account_sid, self.auth_token)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls.json",
                auth=auth,
                data=data,
            )

            if response.status_code == 201:
                return response.json()["sid"]
            else:
                logger.error(f"Twilio call failed: {response.text}")
                raise Exception(f"Twilio call failed: {response.status_code}")


class PlivoService(TelephonyService):
    """Plivo telephony service."""

    def __init__(self):
        self.auth_id = settings.plivo_auth_id
        self.auth_token = settings.plivo_auth_token
        self.phone_number = settings.plivo_phone_number

    async def make_outbound_call(self, to: str, from_phone: str = None) -> str:
        """Make an outbound Plivo call."""
        from_number = from_phone or self.phone_number

        print("settings.server_base_url", settings.server_base_url)
        data = {
            "answer_url": f"https://{settings.server_base_url}/plivo/xml",
            "to": to,
            "from": from_number,
        }

        auth = httpx.BasicAuth(self.auth_id, self.auth_token)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.plivo.com/v1/Account/{self.auth_id}/Call",
                auth=auth,
                data=data,
            )

            if response.status_code == 201:
                return response.json()["request_uuid"]
            else:
                logger.error(f"Plivo call failed: {response.text}")
                raise Exception(f"Plivo call failed: {response.status_code}")


# Service instances
twilio_service = TwilioService()
plivo_service = PlivoService()

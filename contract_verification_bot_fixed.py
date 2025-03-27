import os
import asyncio
import json
import gspread
from datetime import datetime
from dotenv import load_dotenv
from twilio.rest import Client
from oauth2client.service_account import ServiceAccountCredentials
from typing import Dict, List
import logging

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyTransport

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setup Google Sheets
def setup_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.getenv("GOOGLE_SHEETS_CREDENTIALS"), scope)
    client = gspread.authorize(creds)
    return client.open(os.getenv("GOOGLE_SHEETS_NAME")).sheet1

# Save verification results to Google Sheets
def save_to_sheets(sheet, contract_data, recording_url, verification_status):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        current_time,
        contract_data.get("contract_id", ""),
        contract_data.get("naam", ""),
        contract_data.get("bedrijfsnaam", ""),
        contract_data.get("postcode", ""),
        contract_data.get("huisnummer", ""),
        contract_data.get("adres", ""),
        contract_data.get("telefoonnummer", ""),
        contract_data.get("email", ""),
        contract_data.get("pakket", ""),
        recording_url,
        verification_status
    ]
    sheet.append_row(row)

# Send notification to backoffice
def notify_backoffice(contract_id, status):
    # In a real implementation, this could send an email, SMS, or update a CRM
    print(f"Contract {contract_id} verification status: {status}")
    # TODO: Add actual notification logic (email, webhook, etc.)

# Create Twilio client
def get_twilio_client():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    return Client(account_sid, auth_token)

# Start an outbound call
def start_verification_call(contract_data):
    client = get_twilio_client()
    
    # Create a TwiML Bin or use a webhook URL that Twilio will call when the call is answered
    # This URL should trigger your Pipecat application
    
    call = client.calls.create(
        to=contract_data.get("telefoonnummer"),
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        url=os.getenv("TWILIO_WEBHOOK_URL"),
        record=True,
        recording_status_callback=os.getenv("RECORDING_CALLBACK_URL"),
        status_callback=os.getenv("CALL_STATUS_CALLBACK_URL"),
        # Pass contract data as parameters
        status_callback_event=["completed"],
        # When the client presses 1, forward the call
        if_machine="continue"  # Continue even if answering machine is detected
    )
    
    return call.sid

class ContractVerificationPipeline:
    """
    A pipeline for verifying contract details via a voice call.
    This is a simplified version that only demonstrates asking questions.
    """
    
    def __init__(self, contract_data: Dict[str, str]):
        """Initialize with contract data to verify."""
        self.contract_data = contract_data
        self.current_question_index = 0
        
        # Verification questions in Dutch
        self.questions = [
            f"Is uw naam {contract_data['naam']}?",
            f"Is uw bedrijfsnaam {contract_data['bedrijfsnaam']}?",
            f"Is uw adres {contract_data['adres']}?",
            f"Is uw telefoonnummer {contract_data['telefoonnummer']}?",
            f"Is uw e-mailadres {contract_data['email']}?",
            f"Bevestigt u dat u het {contract_data['pakket']} wilt afnemen?"
        ]
        
        # Intro message in Dutch
        self.intro_message = (
            f"Welkom bij de verificatie van uw contract. "
            f"Ik ga u enkele vragen stellen om uw gegevens te bevestigen. "
            f"U kunt eenvoudig antwoorden met 'ja' of 'nee'. "
            f"We beginnen met de verificatie van contract {contract_data['contract_id']}."
        )
        
        self.outro_message = (
            "Hartelijk dank voor het verifiëren van uw gegevens. "
            "Uw contract is nu bevestigd. Een bevestigingsmail wordt naar u verstuurd. "
            "Heeft u nog vragen, dan kunt u contact opnemen met onze klantenservice. "
            "Fijne dag verder!"
        )
        
        self.pipeline = None
        self.task = None
        self.transport = None
        
    async def setup(self):
        """Set up the verification pipeline."""
        # Set up services
        stt = DeepgramSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            language="nl",
            model="nova-2",
        )
        
        llm = OpenAILLMService(
            input={
                "system": (
                    "You are a Dutch-speaking assistant helping with contract verification. "
                    "Listen to what the customer says and determine if they confirmed or denied "
                    "the information. Only respond with 'confirmed' or 'denied'."
                ),
            }
        )
        
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=os.environ["ELEVENLABS_VOICE_ID"],
            model_id="eleven_multilingual_v2"
        )
        
        # Set up the Daily transport
        self.transport = DailyTransport(
            room_url=os.environ["DAILY_ROOM_URL"],
            token=os.environ.get("DAILY_TOKEN"),
            bot_name="Contract Verificatie Bot"
        )
        
        # Set up the event handlers 
        @self.transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            print(f"Participant joined: {participant.user_name} ({participant.user_id})")
            # Start the verification when a human participant joins
            if not participant.is_owner:
                await self.start_verification()

        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant):
            print(f"Participant left: {participant.user_name} ({participant.user_id})")
            if not participant.is_owner:
                print("Human participant left, ending verification")
                await self.task.cancel()
        
        # Create a new pipeline with processors
        self.pipeline = Pipeline([
            self.transport.input(),
            stt,
            llm,
            tts,
            self.transport.output()
        ])
        
        return self
    
    async def start_verification(self):
        """Start the verification process."""
        await asyncio.sleep(1)  # Wait for connection to stabilize
        
        # Play intro message
        await self.transport.queue_frame(TextFrame(self.intro_message))
        await asyncio.sleep(5)  # Give time for intro to play
        
        # Ask each question with a delay between them
        for i, question in enumerate(self.questions):
            print(f"Asking question {i+1}: {question}")
            await self.transport.queue_frame(TextFrame(question))
            await asyncio.sleep(5)  # Wait between questions
        
        # Play outro message
        await self.transport.queue_frame(TextFrame(self.outro_message))
        await asyncio.sleep(5)  # Give time for outro to play
        
        print("Verification completed")
        
    async def run(self):
        """Run the verification pipeline."""
        print("Starting verification pipeline...")
        runner = PipelineRunner()
        task = PipelineTask(self.pipeline)
        self.task = task
        
        try:
            print("Verification pipeline started, waiting for completion...")
            await runner.run(task)
            print("Verification pipeline completed")
        except asyncio.CancelledError:
            print("Verification pipeline cancelled")
        except Exception as e:
            print(f"Error in verification pipeline: {e}")
            raise
        finally:
            print("Closing pipeline")
            await self.pipeline.cleanup()

# Webhook endpoint for Salesdock
async def handle_salesdock_webhook(request_data):
    # Parse contract data from Salesdock webhook
    contract_data = request_data.get("contract", {})
    
    # Start a verification call
    call_sid = start_verification_call(contract_data)
    
    # Return info to the webhook sender
    return {
        "status": "success",
        "call_sid": call_sid,
        "message": "Verification call initiated"
    }

# Main function to run the pipeline
async def main():
    # In a real implementation, this would be triggered by a webhook
    # For demonstration, we'll use sample data
    sample_contract = {
        "contract_id": "CON12345",
        "naam": "Jan Jansen",
        "bedrijfsnaam": "Jansen IT Solutions",
        "postcode": "1234AB",
        "huisnummer": "42",
        "adres": "Hoofdstraat 42, Amsterdam",
        "telefoonnummer": "+31612345678",
        "email": "jan@jansen-it.nl",
        "pakket": "Premium Support Pakket"
    }
    
    # Initialize and run the verification pipeline
    verification = await ContractVerificationPipeline(sample_contract).setup()
    await verification.run()

if __name__ == "__main__":
    asyncio.run(main()) 
import asyncio
import os
import json
import gspread
from datetime import datetime
from dotenv import load_dotenv
from twilio.rest import Client
from oauth2client.service_account import ServiceAccountCredentials

from pipecat.frames.frames import TextFrame, AudioFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAIService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Load environment variables
load_dotenv()

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
    def __init__(self, contract_data):
        self.contract_data = contract_data
        self.verification_results = {
            "postcode_huisnummer": None,
            "bedrijfsnaam": None,
            "adres": None,
            "telefoonnummer": None,
            "email": None,
            "pakket": None,
            "overall": False
        }
        self.recording_url = None
        self.current_question = 0
        self.questions = [
            f"Zijn de postcode {contract_data.get('postcode')} en het huisnummer {contract_data.get('huisnummer')} correct?",
            f"Is de bedrijfsnaam {contract_data.get('bedrijfsnaam')} correct?",
            f"Is het adres {contract_data.get('adres')} correct?",
            f"Is uw telefoonnummer {contract_data.get('telefoonnummer')} correct?",
            f"Is het e-mailadres {contract_data.get('email')} correct?",
            f"Is het gekozen pakket {contract_data.get('pakket')}, inclusief de voorwaarden, correct?"
        ]

    async def setup(self):
        # Initialize Daily for call handling (this would be replaced by Twilio in production)
        # For development/testing we use Daily
        self.transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL"),
            token="",
            bot_name="Contract Verificatie",
            params=DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                start_video_off=True,
                start_audio_off=True
            )
        )

        # Initialize speech-to-text with Deepgram
        self.stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            language="nl"  # Dutch language
        )

        # Initialize text-to-speech with ElevenLabs
        self.tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            model_id="eleven_multilingual_v2"  # Multilingual model for Dutch support
        )

        # Initialize OpenAI for conversation understanding
        self.llm = OpenAIService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview",
            system_prompt="""
            Je bent een professionele assistent die helpt bij het verifiëren van contractgegevens.
            Luister alleen naar de directe antwoorden op de vragen en bepaal of het antwoord een bevestiging is.
            Antwoord alleen met 'true' als het antwoord een duidelijke bevestiging is (zoals 'ja', 'correct', 'klopt', etc.).
            Antwoord met 'false' als het antwoord ontkennend of twijfelachtig is.
            Reageer alleen op de verificatievraag, negeer smalltalk of andere opmerkingen.
            """
        )

        # Create the pipeline
        self.pipeline = Pipeline([
            self.stt,  # Convert speech to text
            self.llm,  # Process with OpenAI
            self.tts,  # Convert response to speech
            self.transport.output()  # Output to call
        ])

        # Create pipeline runner
        self.runner = PipelineRunner()
        self.task = PipelineTask(self.pipeline)

        # Set up event handlers
        self._setup_event_handlers()

        return self

    def _setup_event_handlers(self):
        # Handle new participant
        @self.transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            # Start the verification process
            await self.ask_next_question()

        # Handle DTMF tones (when customer presses 1)
        @self.transport.event_handler("on_dtmf_received")
        async def on_dtmf_received(transport, participant, dtmf):
            if dtmf == "1":
                # Transfer to human operator
                await self.task.queue_frame(TextFrame(
                    "U wordt nu doorverbonden met een medewerker. Een moment geduld alstublieft."
                ))
                # In a real implementation, you would use Twilio's API to transfer the call
                # Here we'll just end the task
                await self.task.cancel()

        # Handle speech input
        @self.transport.event_handler("on_participant_started_speaking")
        async def on_started_speaking(transport, participant):
            # The pipeline will automatically process the speech through STT and LLM

        # Handle text from LLM
        @self.llm.event_handler("on_text_output")
        async def on_llm_output(llm, text):
            # Process the response (true/false) from the LLM
            text = text.strip().lower()
            if self.current_question < len(self.questions):
                field = list(self.verification_results.keys())[self.current_question]
                
                if text == "true":
                    self.verification_results[field] = True
                    self.current_question += 1
                    await self.ask_next_question()
                else:
                    self.verification_results[field] = False
                    await self.handle_negative_response()

        # Handle participant leaving
        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            # Check if all questions were answered positively
            all_verified = all(result for result in self.verification_results.values() if result is not None)
            
            if all_verified:
                self.verification_results["overall"] = True
                # In a real implementation, you would save the results here
                sheet = setup_google_sheets()
                save_to_sheets(sheet, self.contract_data, self.recording_url, "Geverifieerd")
                notify_backoffice(self.contract_data.get("contract_id"), "Geverifieerd")
            
            await self.task.cancel()

    async def ask_next_question(self):
        if self.current_question < len(self.questions):
            question = self.questions[self.current_question]
            # Queue the question to be spoken
            await self.task.queue_frame(TextFrame(question))
        else:
            # All questions answered positively
            await self.task.queue_frame(TextFrame(
                "Dank u wel voor het verifiëren van uw gegevens. Uw contract is bevestigd. "
                "Een medewerker zal de afhandeling verzorgen. Nog een prettige dag."
            ))
            # In a real implementation, you would save the results here
            # We'll wait 5 seconds to let the final message play before ending
            await asyncio.sleep(5)
            await self.task.cancel()

    async def handle_negative_response(self):
        await self.task.queue_frame(TextFrame(
            "Ik begrijp dat er iets niet klopt. "
            "Druk op 1 om doorverbonden te worden met een medewerker, "
            "of zeg 'opnieuw proberen' om de vraag nogmaals te beantwoorden."
        ))
        # In a real implementation, you would implement logic to retry or transfer

    async def run(self):
        await self.runner.run(self.task)

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
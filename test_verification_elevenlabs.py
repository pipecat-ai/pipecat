import asyncio
import os
from dotenv import load_dotenv
from contract_verification_bot_elevenlabs import ContractVerificationPipeline

# Load environment variables
load_dotenv()

async def test_verification():
    """
    Test the contract verification bot with sample data.
    This allows you to test the bot without setting up the full API server.
    Uses ElevenLabs for text-to-speech.
    """
    print("Starting contract verification test with ElevenLabs TTS...")
    
    # Sample contract data
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
    
    print(f"Using sample contract: {sample_contract}")
    print("\nChecking environment variables...")
    
    required_vars = ["DAILY_ROOM_URL", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("\n❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        
        if "ELEVENLABS_VOICE_ID" in missing_vars:
            print("\nYou need an ElevenLabs Voice ID. You can get one by:")
            print("1. Sign up at https://elevenlabs.io")
            print("2. Go to your dashboard and view available voices")
            print("3. Copy the Voice ID of a multilingual voice that supports Dutch")
            print("4. Add it to your .env file as ELEVENLABS_VOICE_ID=voice_id_here")
        
        return
    
    print("\n✅ All required environment variables found.")
    print("\nSetting up verification pipeline...")
    
    try:
        # Initialize and run the verification pipeline
        verification = await ContractVerificationPipeline(sample_contract).setup()
        
        print("\nRunning verification process...")
        print("The bot will now connect to Daily.co and start the verification process.")
        print("You should connect to the Daily room to interact with the bot.")
        print(f"Daily room URL: {os.getenv('DAILY_ROOM_URL')}")
        print("\nVerification questions will be:")
        
        for i, question in enumerate(verification.questions):
            print(f"{i+1}. {question}")
        
        print("\nYou should respond with 'ja' to each question for successful verification.")
        print("Press '1' during the call to simulate forwarding to a human operator.")
        
        await verification.run()
        print("\nVerification process completed.")
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(test_verification()) 
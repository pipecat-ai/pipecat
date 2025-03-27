import asyncio
import os
from dotenv import load_dotenv
from contract_verification_bot import ContractVerificationPipeline

# Load environment variables
load_dotenv()

async def test_verification():
    """
    Test the contract verification bot with sample data.
    This allows you to test the bot without setting up the full API server.
    """
    print("Starting contract verification test...")
    
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
    print("\nSetting up verification pipeline...")
    
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
    
    try:
        await verification.run()
        print("\nVerification process completed.")
    except Exception as e:
        print(f"\nError during verification: {str(e)}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(test_verification()) 
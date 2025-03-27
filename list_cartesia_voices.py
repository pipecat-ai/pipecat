import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_cartesia_voices():
    """
    List all available voices from Cartesia API.
    This will help you select a voice ID for your contract verification bot.
    """
    api_key = os.getenv("CARTESIA_API_KEY")
    
    if not api_key:
        print("Error: CARTESIA_API_KEY not found in .env file")
        print("Please add your Cartesia API key to your .env file:")
        print("CARTESIA_API_KEY=your_api_key_here")
        return
    
    url = "https://api.cartesia.ai/api/v1/voices"
    
    headers = {
        "Accept": "application/json",
        "x-api-key": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            voices = response.json()
            
            print("\n=== Available Cartesia Voices ===\n")
            
            # Filter for Dutch voices specifically
            dutch_voices = [v for v in voices if v.get('language') == 'nl-NL']
            
            if dutch_voices:
                print("Dutch Voices (nl-NL):")
                for voice in dutch_voices:
                    print(f"ID: {voice.get('voice_id')}")
                    print(f"Name: {voice.get('name')}")
                    print(f"Gender: {voice.get('gender')}")
                    print(f"Description: {voice.get('description', 'No description')}")
                    print("-" * 40)
            
            print("\nAll Available Voices:")
            for voice in voices:
                print(f"ID: {voice.get('voice_id')}")
                print(f"Name: {voice.get('name')}")
                print(f"Language: {voice.get('language')}")
                print(f"Gender: {voice.get('gender')}")
                print(f"Description: {voice.get('description', 'No description')}")
                print("-" * 40)
                
            print("\nTo use a voice, add its ID to your .env file:")
            print("CARTESIA_VOICE_ID=voice_id_here")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error connecting to Cartesia API: {str(e)}")

if __name__ == "__main__":
    list_cartesia_voices() 
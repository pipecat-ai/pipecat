import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_daily_room():
    """
    Create a permanent Daily room for your voice agent.
    The room URL will be printed and should be added to your .env file.
    """
    api_key = os.getenv("DAILY_API_KEY")
    
    if not api_key:
        print("Error: DAILY_API_KEY not found in .env file")
        print("Please add your Daily API key to your .env file:")
        print("DAILY_API_KEY=your_api_key_here")
        print("You can find your API key in the Daily dashboard under Developers > API Keys")
        return
    
    # Room configuration
    room_name = "contract-verification"  # You can change this name
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "name": room_name,
        "properties": {
            "start_video_off": True,
            "start_audio_off": True,
            "enable_recording": "cloud"
        }
    }
    
    print(f"Creating Daily room '{room_name}'...")
    
    try:
        response = requests.post(
            "https://api.daily.co/v1/rooms",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            room_data = response.json()
            room_url = room_data["url"]
            
            print("\n✅ Room created successfully!")
            print(f"\nRoom URL: {room_url}")
            print("\nAdd this to your .env file:")
            print(f"DAILY_ROOM_URL={room_url}")
            
            # Check if .env file exists and offer to update it
            if os.path.exists(".env"):
                update = input("\nDo you want to automatically update your .env file with this URL? (y/n): ")
                if update.lower() == 'y':
                    with open(".env", "r") as file:
                        env_content = file.read()
                    
                    if "DAILY_ROOM_URL=" in env_content:
                        env_content = env_content.replace(
                            "DAILY_ROOM_URL=", 
                            f"DAILY_ROOM_URL={room_url}"
                        )
                    else:
                        env_content += f"\nDAILY_ROOM_URL={room_url}\n"
                    
                    with open(".env", "w") as file:
                        file.write(env_content)
                    
                    print("✅ .env file updated!")
        
        elif response.status_code == 401:
            print("❌ Error: Invalid API key. Please check your DAILY_API_KEY in the .env file.")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Error connecting to Daily API: {str(e)}")

if __name__ == "__main__":
    create_daily_room() 
from pydantic import BaseModel

# Define a model for API requests
class BotRequest(BaseModel):
    bot_name: str
    room_id: str

# Define a response model for the API response
class BotResponse(BaseModel):
    room_url: str
    token: str

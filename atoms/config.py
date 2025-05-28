import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    debug: bool = False

    # Twilio
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

    # Plivo
    plivo_auth_id: Optional[str] = None
    plivo_auth_token: Optional[str] = None
    plivo_phone_number: Optional[str] = None

    # AI Services
    openai_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    cartesia_api_key: Optional[str] = None

    # Server URL for webhooks
    server_base_url: Optional[str] = None

    # MongoDB
    mongodb_url: Optional[str] = None

    # Redis
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra environment variables


settings = Settings()

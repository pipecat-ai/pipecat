"""
Configuration management for Pipecat AI Backend
Optimized for local AI rig with Ollama and Cartesia
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # AI Rig Configuration (Local Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "llama3.2"  # Fast model for low latency

    # Cartesia Configuration (TTS/STT)
    CARTESIA_API_KEY: str = ""
    CARTESIA_TTS_VOICE_ID: str = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Conversational voice
    CARTESIA_TTS_MODEL: str = "sonic-3"
    CARTESIA_STT_MODEL: str = "ink-whisper"
    CARTESIA_LANGUAGE: str = "en"

    # Twilio Configuration
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None

    # Database Configuration (SQLite for simplicity, can upgrade to PostgreSQL)
    DATABASE_URL: str = "sqlite:///./backend_data.db"

    # Authentication
    SECRET_KEY: str = "change-this-to-a-secure-random-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Session Management
    SESSION_TIMEOUT_SECONDS: int = 300  # 5 minutes
    MAX_CONCURRENT_SESSIONS: int = 100

    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_MESSAGE_SIZE: int = 1024 * 1024  # 1MB

    # Audio Configuration
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1

    # Performance Tuning for Local AI Rig
    OLLAMA_NUM_THREADS: int = 6  # i5-10400 has 12 threads
    OLLAMA_NUM_GPU: int = 1  # RTX 3090

    # CORS Configuration
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]

    # Feature Flags
    ENABLE_METRICS: bool = True
    ENABLE_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Voice configurations for different use cases
VOICE_CONFIGS = {
    "conversational": {
        "voice_id": "a0e99841-438c-4a64-b679-ae501e7d6091",
        "speed": "normal",
        "emotion": ["positivity:high", "curiosity:medium"]
    },
    "professional": {
        "voice_id": "694f9389-aac1-45b6-b726-9d9369183238",
        "speed": "normal",
        "emotion": ["positivity:medium"]
    },
    "assistant": {
        "voice_id": "79a125e8-cd45-4c13-8a67-188112f4dd22",
        "speed": "fast",
        "emotion": ["positivity:high"]
    }
}


# System prompts for different scenarios
SYSTEM_PROMPTS = {
    "default": """You are a helpful AI assistant. You are having a voice conversation with a user.
Keep your responses concise and natural for voice interaction. Speak clearly and conversationally.""",

    "customer_service": """You are a professional customer service representative.
Be helpful, patient, and solve customer issues efficiently. Keep responses brief and to the point.""",

    "sales": """You are a friendly sales assistant. Help customers find what they need,
answer questions about products, and guide them through the purchasing process.""",

    "appointment": """You are an appointment scheduling assistant. Help users book,
modify, or cancel appointments. Be efficient and confirm all details clearly.""",
}

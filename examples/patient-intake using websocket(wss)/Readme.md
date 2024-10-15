# PipeCat Medical Intake System(Firebase integration(wss))

An automated medical intake system using AI-powered conversational interfaces.

## Key Features:
- Custom WebSocket server transport supporting WSS (secure WebSocket)
- Speech-to-Text (STT) using Deepgram
- Natural Language Processing with OpenAI's GPT-4
- Text-to-Speech (TTS) using Cartesia
- Firebase integration for data storage
- Voice Activity Detection (VAD) using Silero

## Core Components:
- `IntakeProcessor`: Manages conversation flow and data collection
- `Pipeline`: Orchestrates data flow between various services
- Custom `WebsocketServerTransport`: Handles secure real-time audio communication (WSS)
- Firebase integration for secure storage of patient data

## System Flow:
1. Patient connects via secure WebSocket (WSS)
2. System conducts a conversation to collect:
   - Identity verification
   - Current prescriptions
   - Allergies
   - Medical conditions
   - Reasons for the visit
3. Data is processed and stored in Firebase

## Technologies Used:
- Python 3.x
- asyncio for asynchronous programming
- Custom WebSocket server implementation supporting WSS
- Firebase Admin SDK
- OpenAI API
- Deepgram API
- Cartesia API


# Patient-intake chatbot

<img src="image.png" width="420px">

This project implements an AI-powered chatbot designed to streamline the medical intake process for Tri-County Health Services. The chatbot, named Jessica, interacts with patients to collect essential information before their doctor's visit, enhancing efficiency and improving the patient experience.

💡 Looking to build structured conversations? Check out [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for managing complex conversational states and transitions.

## Features

Identity Verification: Confirms patient identity by verifying their date of birth.
Prescription Information: Collects details about current medications and dosages.
Allergy Documentation: Records patient allergies.
Medical Conditions: Gathers information about existing medical conditions.
Reason for Visit: Asks patients about the purpose of their current doctor's visit.

## Technical Stack

Language: Python
AI Model: OpenAI's GPT-4
Text-to-Speech: Cartesia TTS Service
Audio Processing: Silero VAD (Voice Activity Detection)
Real-time Communication: Daily.co API

## Key Components

IntakeProcessor: Manages the conversation flow and information gathering process.
DailyTransport: Handles real-time audio communication.
CartesiaTTSService: Converts text responses to speech.
OpenAILLMService: Processes natural language and generates appropriate responses.
Pipeline: Orchestrates the flow of information between different components.

How It Works

The chatbot introduces itself and verifies the patient's identity.
It systematically collects information about prescriptions, allergies, medical conditions, and the reason for the visit.
The conversation is guided by a series of function calls that transition between different stages of the intake process.
All collected information is logged for later use by medical professionals.

ℹ️ The first time, things might take extra time to get started since VAD (Voice Activity Detection) model needs to be downloaded.

## Get started

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp env.example .env # and add your credentials

```

## Run the server

```bash
python server.py
```

Then, visit `http://localhost:7860/` in your browser to start a chatbot session.

## Build and test the Docker image

```
docker build -t chatbot .
docker run --env-file .env -p 7860:7860 chatbot
```
## Cartesia best practices

Since this example is using Cartesia, checkout the best practices given in Cartesia's docs. LLM prompts should be modified accordingly.
<https://docs.cartesia.ai/build-with-sonic/formatting-text-for-sonic/best-practices>

<https://docs.cartesia.ai/build-with-sonic/formatting-text-for-sonic/inserting-breaks-pauses>

<https://docs.cartesia.ai/build-with-sonic/formatting-text-for-sonic/spelling-out-input-text>
### Example
```python
messages = [
    {
        "role": "system",
        "content": '''You are a helpful AI assistant. Format all responses following these guidelines:

1. Use proper punctuation and end each response with appropriate punctuation
2. Format dates as MM/DD/YYYY
3. Insert pauses using - or <break time='1s' /> for longer pauses
4. Use ?? for emphasized questions
5. Avoid quotation marks unless citing
6. Add spaces between URLs/emails and punctuation marks
7. For domain-specific terms or proper nouns, provide pronunciation guidance in [brackets]
8. Keep responses clear and concise
9. Use appropriate voice/language pairs for multilingual content

Your goal is to demonstrate these capabilities in a succinct way. Your output will be converted to audio, so maintain natural communication flow. Respond creatively and helpfully, but keep responses brief. Start by introducing yourself.'''
    }
]
```

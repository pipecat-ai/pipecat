# Chatbot with Coval Testing & Observability

Implement a chatbot with Coval testing and observability.

## Features

### Testing Capabilities

- **Automated Testing**: Leverages Coval.dev for comprehensive testing of chatbot interactions and responses.
- **Test Case Generation**: Automatically generates test scenarios based on common user interactions.
- **Integration Testing**: Validates the interaction between different components (Speech-to-Text, LLM, Text-to-Speech).
- **Regression Testing**: Ensures new updates don't break existing functionality.

### Observability Features

- **Conversation Analytics**: Tracks and analyzes chat patterns, response times, and interaction quality.
- **Quality Assurance**: Monitors response accuracy and consistency across different conversation flows.
- **Performance Monitoring**: Measures and tracks response latency and system resource usage.
- **Error Detection**: Identifies and logs unexpected behaviors or response patterns.

# Get your basic Pipecat voice agent working

## Usage

Run the bot as is for a demo of a basic Pipecat voice agent. You can also switch to use Spanish with the commented out code samples in the bot.py file.

## Events

- Participants joining or leaving the call are handled dynamically, adjusting the chatbot's behavior accordingly.

ℹ️ The first time, things might take extra time to get started since VAD (Voice Activity Detection) model needs to be downloaded.

## Requirements

- Python 3.10+
- `python-dotenv`
- Additional libraries from the `pipecat` package.

## Create a Coval API Key

1. Go to www.coval.dev
2. Create a new account
3. Create Coval API key
   - Paste the API key into the .env file

## Setup

1. Clone the repository.
2. Install the required packages.
3. Set up environment variables for API keys:
   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`
   - `COVAL_API_KEY`
4. Run the script.

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp env.example .env # add your Coval credentials

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

# Testing: Simulate a user to test your bot with Coval

1. Go to the Coval UI at www.coval.dev and log in to your account

2. Navigate to Settings in the left sidebar menu

3. Under "Connections", click "Add New Connection"

   - Enter a name for your connection
   - Add your bot's URL (e.g. http://localhost:7860)
   - Click "Save Connection"

4. Go to "Launch Evaluation" in the left sidebar

5. Set up your test:
   - Select a test set from the dropdown menu
   - Choose your newly created connection
   - Click "Start Test" to begin evaluating your bot

# Observability: Monitor your bot with Coval

Monitoring will be automatically enabled for your bot when you run it with this example. After performing a live call or a test, you can view the results in the Coval UI.

1. Go to the Coval UI at www.coval.dev and log in to your account

2. Navigate to "Monitoring" in the left sidebar

Note: You may want to exclude simulated calls from your monitoring results. You can do this in the bot.py file.

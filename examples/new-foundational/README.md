# Pipecat Foundational Examples

This directory contains foundational examples showing how to use Pipecat to build voice and multimodal agents. Each example demonstrates specific features of the framework, building from basic to more complex concepts.

## Running the Examples

### Prerequisites

1. If you haven't already, set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install Pipecat with the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up an `.env` file with the API keys of services you'll run.

### Using the Example Runner

The examples use a web app that allows you to interact with the bots via WebRTC.

1. **Run a specific example**:

   ```bash
   python run.py <example-name>
   ```

   For example:

   ```bash
   python run.py 07-interruptible.py
   ```

2. **Open the web app** at the URL displayed in the console:

   ```
   Open your browser to: http://localhost:7860
   ```

3. **Start the example**:

Click "Connect" to start the example. Control your devices as needed.

## Troubleshooting

- **No audio or video**: Make sure your browser permissions for microphone and camera are granted
- **Connection errors**: Check that your API keys are correctly set in the `.env` file
- **Missing dependencies**: Ensure you've installed all required dependencies with `pip install -r requirements.txt`
- **Port already in use**: Change the port with `--port <number>` if the default port is unavailable

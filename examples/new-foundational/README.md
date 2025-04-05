# Pipecat Foundational Examples

This directory contains foundational examples showing how to use Pipecat to build voice and multimodal agents. Each example demonstrates specific features of the framework, building from basic to more complex concepts.

## Running the Examples

### Prerequisites

1. If you haven't already, set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Pipecat with the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up an `.env` file with the API keys of services you'll run.

### Using the Example Runner

The examples use a web app that allows you to interact with the bots via WebRTC.

1. **List available examples**:

   ```bash
   python run.py --list
   ```

2. **Run a specific example**:

   ```bash
   python run.py bots/<example-name>
   ```

   For example:

   ```bash
   python run.py bot/07-interruptible.py
   ```

3. **Open the web app** at the URL displayed in the console:

   ```
   Open your browser to: http://localhost:8000
   ```

4. **Start the example**:

Click "Connect" to start the example. Control your devices as needed.

### Command Line Options

```
usage: run.py [-h] [-p PORT] [--host HOST] [-l] [bot_file]

Pipecat Foundational Examples

positional arguments:
  bot_file              Path to bot Python file to run

options:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  Port to run the server on (default: 8000)
  --host HOST           Host to bind the server to (default: 0.0.0.0)
  -l, --list            List available bots and exit
```

## Troubleshooting

- **No audio or video**: Make sure your browser permissions for microphone and camera are granted
- **Connection errors**: Check that your API keys are correctly set in the `.env` file
- **Missing dependencies**: Ensure you've installed all required dependencies with `pip install -r requirements.txt`
- **Port already in use**: Change the port with `--port <number>` if the default port is unavailable

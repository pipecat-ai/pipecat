# AWS Strands Examples

This folder contains two Python examples demonstrating how to use Pipecat with the AWS Strands agent.

## Overview

These examples show how to delegate complex, multi-step tasks to a Strands agent, which can reason step-by-step and call tools to accomplish user requests.

These examples are intentionally simplified for demonstration, using mock API calls. They work best if you ask it:

> What's the weather where the Golden Gate Bridge is?

## Example Scripts

### `black-box.py`

A minimal example that demonstrates how to use the Strands agent with Pipecat. The agent can handle multi-step queries by calling tools, but does not explain its reasoning out loud.

### `explain-thinking.py`

An enhanced example where the Strands agent explains each step of its reasoning in clear, simple language as it works through a multi-step task.

## Quick Start

1. **Clone the repository and navigate to this example:**

   ```bash
   git clone https://github.com/pipecat-ai/pipecat.git
   cd pipecat/examples/aws-strands
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Enable AWS Bedrock models:**
   ⚠️ **Important:** AWS Strands uses Bedrock models by default. You must first activate the required models in your AWS Bedrock console before running these examples. Visit the [AWS Bedrock Model Access documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-permissions.html) to enable model access permissions.


5. **Configure environment variables:**

   Copy the provided `env.example` file to `.env` and fill in the necessary credentials:

   ```bash
   cp env.example .env
   # Then edit .env with your preferred editor
   ```

6. **Run an example:**

   ```bash
   python black-box.py
   # or
   python explain-thinking.py
   # The transport is selected via the --transport or -t command line argument. Choices are daily webrtc and twilio.defaults to    #  webrtc
   ```

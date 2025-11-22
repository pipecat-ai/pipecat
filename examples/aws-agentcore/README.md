# Amazon Bedrock AgentCore Runtime Example

This example demonstrates how to prepare a Pipecat bot for deployment to **Amazon Bedrock AgentCore Runtime** and enable it to invoke AgentCore tools.

## Overview

This example shows the set needed to:

- Deploy your Pipecat bot to Amazon Bedrock AgentCore Runtime (which hosts and runs your bot)
- Enable your bot to invoke AgentCore tools while running in the AgentCore Runtime

The key additions to a standard Pipecat bot are the AgentCore-specific configurations and tool invocation handling that allow your bot to leverage the full AgentCore ecosystem.

## Prerequisites

- Accounts with:
  - AWS
  - OpenAI
  - Deepgram
  - Cartesia
  - Daily
- Python 3.10 or higher
- `uv` package manager

## IAM Configuration

Configure your IAM user with the necessary policies for AgentCore usage. Start with these:

- `BedrockAgentCoreFullAccess`
- A new policy (maybe named `BedrockAgentCoreCLI`) configured [like this](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html#runtime-permissions-starter-toolkit)

You can also choose to specify more granular permissions; see [Amazon Bedrock AgentCore docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html) for more information.

To simplify the remaining steps in this README, it's a good idea to export some AWS-specific environment variables:

```bash
export AWS_SECRET_ACCESS_KEY=...
export AWS_ACCESS_KEY_ID=...
export AWS_REGION=...
```

## Agent Configuration

Configure your bot as an AgentCore agent.

```bash
agentcore configure -e bot.py
```

Follow the prompts to complete the configuration.

**IMPORTANT:** when asked if you want to use "Direct Code Deploy" or "Container", choose "Container". Today there is an incompatibility between Pipecat and "Direct Code Deploy".

> For the curious: "Direct Code Deploy" requires that all bot dependencies have an `aarch64_manylinux2014` wheel...which is unfortunately not true for `numba`.

## Deployment to AgentCore Runtime

Deploy your configured bot to Amazon Bedrock AgentCore Runtime for production hosting.

```bash
agentcore launch --env OPENAI_API_KEY=... --env DEEPGRAM_API_KEY=... --env CARTESIA_API_KEY=... # -a <agent_name> (if multiple agents configured)
```

You should see commands related to tailing logs printed to the console. Copy and save them for later use.

This is also the command you need to run after you've updated your bot code.

## Running on AgentCore Runtime

Run your bot on AgentCore Runtime.

```bash
agentcore invoke '{"roomUrl": "https://<your-domain>.daily.co/<room-name>"}' # -a <agent_name> (if multiple agents configured)
```

## Observation

Paste the log tailing command you received when deploying your bot to AgentCore Runtime. It should look something like:

```bash
# Replace with your actual command
aws logs tail /aws/bedrock-agentcore/runtimes/bot1-0uJkkT7QHC-DEFAULT --log-stream-name-prefix "2025/11/19/[runtime-logs]" --follow
```

## Running Locally

You can also run your bot locally, using either the SmallWebRTC or Daily transport.

First, copy `env.example` to `.env` and fill in the values.

Then, run the bot:

```bash
# SmallWebRTC
PIPECAT_LOCAL_DEV=1 uv run python bot.py

# Daily
PIPECAT_LOCAL_DEV=1 uv run python bot.py -t daily -d
```

> Ideally you should be able to use `agentcore launch --local`, but it doesn't currently appear to be working (even with [this workaround](https://github.com/aws/bedrock-agentcore-starter-toolkit/issues/156) applied), at least not for this project.

## Additional Resources

For a comprehensive guide to getting started with Amazon Bedrock AgentCore, including detailed setup instructions, see the [Amazon Bedrock AgentCore Developer Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html).

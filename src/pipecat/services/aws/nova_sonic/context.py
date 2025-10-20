#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Context management for AWS Nova Sonic LLM service.

This module provides specialized context aggregators and message handling for AWS Nova Sonic,
including conversation history management and role-specific message processing.

.. deprecated:: 0.0.91
    AWS Nova Sonic now supports `LLMContext` and `LLMContextAggregatorPair`.
    Using the new patterns should allow you to not need types from this module.

    BEFORE:
    ```
    # Setup
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Context frame type
    frame: OpenAILLMContextFrame

    # Context type
    context: AWSNovaSonicLLMContext
    # or
    context: OpenAILLMContext

    # Reading messages from context
    messages = context.messages
    ```

    AFTER:
    ```
    # Setup
    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    # Context frame type
    frame: LLMContextFrame

    # Context type
    context: LLMContext

    # Reading messages from context
    messages = context.get_messages()
    ```
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.aws.nova_sonic.context are deprecated. \n"
        "AWS Nova Sonic now supports `LLMContext` and `LLMContextAggregatorPair`. \n"
        "Using the new patterns should allow you to not need types from this module.\n\n"
        "BEFORE:\n"
        "```\n"
        "# Setup\n"
        "context = OpenAILLMContext(messages, tools)\n"
        "context_aggregator = llm.create_context_aggregator(context)\n\n"
        "# Context frame type\n"
        "frame: OpenAILLMContextFrame\n\n"
        "# Context type\n"
        "context: AWSNovaSonicLLMContext\n"
        "# or\n"
        "context: OpenAILLMContext\n\n"
        "# Reading messages from context\n"
        "messages = context.messages\n"
        "```\n\n"
        "AFTER:\n"
        "```\n"
        "# Setup\n"
        "context = LLMContext(messages, tools)\n"
        "context_aggregator = LLMContextAggregatorPair(context)\n\n"
        "# Context frame type\n"
        "frame: LLMContextFrame\n\n"
        "# Context type\n"
        "context: LLMContext\n\n"
        "# Reading messages from context\n"
        "messages = context.messages\n"
        "```",
        DeprecationWarning,
        stacklevel=2,
    )

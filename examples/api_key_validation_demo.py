#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
API Key Validation Demo

This script demonstrates the API key validation feature that was added to improve
developer experience by catching missing or empty API keys early during service
initialization.

Run this script to see how different scenarios are handled:
    python examples/api_key_validation_demo.py
"""

from pipecat.utils.api_key_validator import APIKeyError


def demo_valid_key():
    """Example 1: Service initialized with a valid API key - works correctly."""
    print("\n" + "=" * 60)
    print("Example 1: Valid API Key")
    print("=" * 60)

    try:
        from pipecat.services.openai import OpenAILLMService

        # This works - valid API key provided
        service = OpenAILLMService(api_key="sk-test-key-123", model="gpt-4.1")
        print("✅ SUCCESS: Service created with valid API key")
        print(f"   Service: {service.__class__.__name__}")
        print(f"   Model: {service.model_name}")
    except APIKeyError as e:
        print(f"❌ ERROR: {e}")


def demo_none_key_not_allowed():
    """Example 2: Service that requires API key, None provided - raises error."""
    print("\n" + "=" * 60)
    print("Example 2: None API Key (not allowed)")
    print("=" * 60)

    try:
        # Anthropic requires explicit API key
        from pipecat.services.anthropic import AnthropicLLMService

        # This will fail - None is not allowed for Anthropic
        service = AnthropicLLMService(api_key=None, model="claude-sonnet-4-5-20250929")
        print("✅ SUCCESS: Service created")
    except APIKeyError as e:
        print(f"❌ EXPECTED ERROR: {e}")
        print("   This error helps catch configuration issues early!")
    except Exception as e:
        print(f"   (Skipped - anthropic package not installed: {type(e).__name__})")


def demo_empty_key():
    """Example 3: Service initialized with empty string - raises error."""
    print("\n" + "=" * 60)
    print("Example 3: Empty String API Key")
    print("=" * 60)

    try:
        from pipecat.services.openai import OpenAILLMService

        # This will fail - empty string is not valid
        service = OpenAILLMService(api_key="", model="gpt-4.1")
        print("✅ SUCCESS: Service created")
    except APIKeyError as e:
        print(f"❌ EXPECTED ERROR: {e}")
        print("   Even blank strings are caught!")


def demo_whitespace_key():
    """Example 4: Service initialized with whitespace-only string - raises error."""
    print("\n" + "=" * 60)
    print("Example 4: Whitespace-Only API Key")
    print("=" * 60)

    try:
        from pipecat.services.openai import OpenAILLMService

        # This will fail - whitespace-only is not valid
        service = OpenAILLMService(api_key="   ", model="gpt-4.1")
        print("✅ SUCCESS: Service created")
    except APIKeyError as e:
        print(f"❌ EXPECTED ERROR: {e}")
        print("   Whitespace-only strings are also caught!")


def demo_none_key_allowed():
    """Example 5: Service that allows None (uses env var) - works correctly."""
    print("\n" + "=" * 60)
    print("Example 5: None API Key (allowed for env var fallback)")
    print("=" * 60)

    try:
        from pipecat.services.openai import OpenAILLMService

        # This works - OpenAI SDK can use OPENAI_API_KEY environment variable
        # Note: This will only work if OPENAI_API_KEY env var is set
        print("   OpenAI allows None since it can use OPENAI_API_KEY env var")
        print("   (This would work if OPENAI_API_KEY is set in environment)")
    except APIKeyError as e:
        print(f"❌ ERROR: {e}")


def main():
    """Run all demonstration examples."""
    print("\n" + "=" * 60)
    print("API Key Validation Demonstration")
    print("=" * 60)
    print("\nThis demo shows how API key validation improves developer experience")
    print("by catching configuration errors early during service initialization.")

    # Run all examples
    demo_valid_key()
    demo_none_key_not_allowed()
    demo_empty_key()
    demo_whitespace_key()
    demo_none_key_allowed()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Benefits:
1. ✅ Errors caught immediately during initialization
2. ✅ Clear, helpful error messages with environment variable hints
3. ✅ Prevents confusing runtime errors later
4. ✅ Consistent validation across all services
5. ✅ Better developer experience overall

Implementation:
- Add validation to __init__ method of each service
- Use validate_api_key() from pipecat.utils.api_key_validator
- Set allow_none=True for services that support env vars
- Set allow_none=False for services that require explicit keys
- Always provide env_var_name for helpful error messages

See API_KEY_VALIDATION_GUIDE.md for full implementation details.
    """)


if __name__ == "__main__":
    main()

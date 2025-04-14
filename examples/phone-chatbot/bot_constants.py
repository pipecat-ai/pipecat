# bot_constants.py
"""Constants used across the bot runner application."""

# Maximum session time
MAX_SESSION_TIME = 5 * 60  # 5 minutes

# Required environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "DAILY_API_KEY",
    "CARTESIA_API_KEY",
    "DEEPGRAM_API_KEY",
]

# Default example to use when handling dialin webhooks - determines which bot type to run
DEFAULT_DIALIN_EXAMPLE = "call_transfer"  # Options: call_transfer, simple_dialin

# Call transfer configuration constants
DEFAULT_CALLTRANSFER_MODE = "dialout"
DEFAULT_SPEAK_SUMMARY = True  # Speak a summary of the call to the operator
DEFAULT_STORE_SUMMARY = False  # Store summary of the call (for future implementation)
DEFAULT_TEST_IN_PREBUILT = False  # Test in prebuilt mode (bypasses need to dial in/out)

- Fixed race condition where interrupted assistant content was added to LLM context after the user's interruption message instead of before, causing the bot to repeat itself. `InterruptionFrame` is now processed synchronously through all processors to ensure proper context ordering.


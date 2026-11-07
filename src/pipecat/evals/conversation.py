#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A conversation between a user and a bot, as a conversation file holds it.

A conversation file is a JSON list of messages, in the order they happened:

- ``{"role": "user", "content": ...}``: what the user said in one turn.
- ``{"role": "assistant", "content": ...}``: everything the bot said between
  two user turns, however turn detection or a function call split it. A turn
  in which the bot said nothing is not a turn, so the Nth assistant message is
  bot turn N, the turn a metric's verdict names.
- ``{"role": "tool_call", "name": ..., "arguments": {...}, "cancelled": ...}``:
  a function the bot called, placed before the bot turn it was made in.

The file holds what each side said and the calls the bot made, which is what
the judge reads. It has no system prompt, no tool results, and no timing. The
harness saves one for every simulation run, and
:class:`~pipecat.evals.conversation_observer.EvalConversationObserver` records
one from a bot's own pipeline. Example::

    [
      {"role": "assistant", "content": "Hi, how can I help?"},
      {"role": "user", "content": "A table for four on Friday, please."},
      {"role": "tool_call", "name": "check_availability",
       "arguments": {"day": "Friday", "party_size": 4}, "cancelled": false},
      {"role": "assistant", "content": "Let me check. Friday at seven works."}
    ]
"""


class EvalConversation:
    """Builds a conversation's messages as the conversation happens.

    The user's turns and the bot's calls are added as they come. What the bot
    says gathers into its current turn, which the next user turn, or
    :meth:`end_bot_turn`, closes.
    """

    def __init__(self):
        """Initialize an empty conversation."""
        self._messages: list[dict] = []
        self._bot_said: list[str] = []

    @property
    def messages(self) -> list[dict]:
        """The messages so far, without the bot's turn in progress."""
        return list(self._messages)

    @property
    def user_spoke_last(self) -> bool:
        """Whether a user turn is the last thing said, with nothing from the bot since."""
        said = [m for m in self._messages if m["role"] != "tool_call"]
        return not self._bot_said and bool(said) and said[-1]["role"] == "user"

    def add_user_message(self, text: str) -> None:
        """Add a user turn, which ends the bot's turn; empty text only ends the bot's turn."""
        self.end_bot_turn()
        if text.strip():
            self._messages.append({"role": "user", "content": text.strip()})

    def add_assistant_message(self, text: str) -> None:
        """Add a segment of what the bot said to its current turn."""
        if text.strip():
            self._bot_said.append(text.strip())

    def add_tool_call(self, name: str, arguments: dict) -> None:
        """Add a function the bot called."""
        self._messages.append(
            {"role": "tool_call", "name": name, "arguments": dict(arguments), "cancelled": False}
        )

    def cancel_tool_call(self, name: str) -> None:
        """Mark the bot's latest call to ``name`` that is not cancelled yet as cancelled."""
        for message in reversed(self._messages):
            if (
                message["role"] == "tool_call"
                and message["name"] == name
                and not message["cancelled"]
            ):
                message["cancelled"] = True
                break

    def end_bot_turn(self) -> None:
        """End the bot's turn: what it said since the user last spoke becomes one message."""
        text = " ".join(self._bot_said)
        self._bot_said.clear()
        if text:
            self._messages.append({"role": "assistant", "content": text})

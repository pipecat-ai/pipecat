#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.evals.conversation import EvalConversation


class TestEvalConversation(unittest.TestCase):
    def test_a_bot_turn_is_what_it_said_between_user_turns_with_its_calls_before_it(self):
        conversation = EvalConversation()
        conversation.add_assistant_message("Hello!")
        conversation.add_user_message("A table at six, please.")
        conversation.add_assistant_message("Let me check.")
        conversation.add_tool_call("check_availability", {"time": "6"})
        conversation.add_assistant_message("Six is free.")
        conversation.end_bot_turn()
        self.assertEqual(
            conversation.messages,
            [
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "A table at six, please."},
                {
                    "role": "tool_call",
                    "name": "check_availability",
                    "arguments": {"time": "6"},
                    "cancelled": False,
                },
                {"role": "assistant", "content": "Let me check. Six is free."},
            ],
        )

    def test_a_turn_in_which_nothing_was_said_is_not_a_turn(self):
        conversation = EvalConversation()
        conversation.add_user_message("Hi.")
        conversation.add_tool_call("lookup", {})
        conversation.add_assistant_message("  ")
        conversation.add_user_message("Hello?")
        conversation.add_user_message("")
        conversation.end_bot_turn()
        self.assertEqual([m["role"] for m in conversation.messages], ["user", "tool_call", "user"])

    def test_a_cancel_marks_the_latest_call_by_that_name_still_standing(self):
        conversation = EvalConversation()
        conversation.add_tool_call("book", {"time": "6"})
        conversation.add_tool_call("book", {"time": "7"})

        def cancelled():
            return [m["cancelled"] for m in conversation.messages]

        conversation.cancel_tool_call("book")
        self.assertEqual(cancelled(), [False, True])
        conversation.cancel_tool_call("book")
        self.assertEqual(cancelled(), [True, True])
        conversation.cancel_tool_call("lookup")
        self.assertEqual(cancelled(), [True, True])

    def test_the_user_spoke_last_until_the_bot_says_something(self):
        conversation = EvalConversation()
        self.assertFalse(conversation.user_spoke_last)
        conversation.add_user_message("Bye.")
        self.assertTrue(conversation.user_spoke_last)
        # A call is not something said.
        conversation.add_tool_call("end", {})
        self.assertTrue(conversation.user_spoke_last)
        conversation.add_assistant_message("Goodbye!")
        self.assertFalse(conversation.user_spoke_last)


if __name__ == "__main__":
    unittest.main()

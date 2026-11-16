#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What the frontend's model reads when it works with a backend.

The pairing instruction appended to the frontend's system instruction, each
request strategy's guidance and ``delegate`` tool description, the
instruction carried on each spoken backend message, and the marks that open
the backend's messages in the frontend's conversation.

The pairing instruction and the tool descriptions are also read by
predictive delegation's classifier. A small change in wording can change
what a model does in ways no unit test catches: gpt-4.1 stops delegating a
"never mind", or the classifier starts delegating questions about how the
work is going. Before and after changing anything here, run the checks in
``scripts/llm-with-backend/tuning/README.md``, and record their results in
``RESULTS.md`` beside it.
"""

from pipecat.processors.aggregators.llm_context import LLMContextMessage

#: How a backend output is marked in the frontend's conversation.
BACKEND_MESSAGE_PREFIX = "Backend: "

#: How a backend reasoning summary is marked in the frontend's conversation.
BACKEND_THOUGHT_PREFIX = "Backend (thinking): "

#: How a backend note, text its model wrote without the mark that asks for
#: the user to be told it, is marked in the frontend's conversation.
BACKEND_NOTE_PREFIX = "Backend (note): "

#: How a function call the backend is making is marked in the frontend's conversation.
BACKEND_WORKING_PREFIX = "Backend (working): "

#: Every mark a rendered backend message can open with.
BACKEND_PREFIXES = (
    BACKEND_MESSAGE_PREFIX,
    BACKEND_THOUGHT_PREFIX,
    BACKEND_NOTE_PREFIX,
    BACKEND_WORKING_PREFIX,
)


def _is_backend_message(message: LLMContextMessage) -> bool:
    """Whether a context message is one the connector rendered from a backend output."""
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    return message.get("role") == "developer" and (
        isinstance(content, str) and content.startswith(BACKEND_PREFIXES)
    )


#: Carried on each spoken backend message, so the frontend relays it on whatever
#: turn it lands on, a user's turn included, where the standing instruction
#: alone leaves it unsaid.
BACKEND_MESSAGE_RELAY_INSTRUCTION = (
    "(Relay this to the user now, in your own words, once, even if the conversation has "
    "moved on. If the user has just said something you have not answered yet, answer that "
    "first and relay this at the end of the same reply.)"
)

#: Where the request strategy's own guidance goes in :data:`FRONTEND_PAIRING_INSTRUCTION`.
PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER = "{strategy}"

#: How the frontend's model works with a backend, appended to its system
#: instruction: what the backend is and when to delegate, that to the user
#: there is one assistant, what to do after delegating, and what to do with
#: the backend's messages. The request strategy's guidance (how the request
#: is worded) goes where :data:`PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER`
#: stands, at the end of the first paragraph. The frontend's own system
#: instruction says what the backend is for.
#: ``BackendConnector(frontend_pairing_instruction=...)`` replaces it.
#:
#: The instruction ends by restating its one rule, that a message about
#: delegated work, a stop included, gets a delegate call. A stop is where a
#: model goes wrong: it answers "never mind" with "I won't make any changes"
#: and no call, and the backend carries on. gpt-4.1 weighs the end of the
#: instruction most, so the restatement is what makes it delegate a stop;
#: the first paragraph alone, with the rule in it, does not, nor does the
#: rule as a list of cases.
FRONTEND_PAIRING_INSTRUCTION = (
    "THE BACKEND: You have a backend for anything that needs a backend tool or careful "
    "reasoning. It sees nothing of the conversation but what you delegate, and nothing you "
    "say reaches it: to start, change or stop its work, call the delegate tool, at once, in "
    "the same reply, and do the rest of the reply yourself (if any is needed) around it. That "
    "holds for every message that concerns its work, whether a new request, a yes to something "
    'that was offered, a correction or a "never mind", and even while earlier work is still '
    "running: a stop you only speak leaves the work running. Do not delegate what you can do "
    "yourself: small talk, a question that a result you already have answers, or a question "
    "about how the work is going, which you answer from the backend's latest notes or, with "
    "none yet, by saying there is no update yet. Never guess a result while waiting. "
    f"{PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER}\n\n"
    "ONE ASSISTANT: To the user there is one assistant, you: never mention the backend, "
    "delegation, a handoff or tools. Say what is being done as your own doing, such as \"I'm "
    'checking", never "the backend is checking", and never that you passed anything along.\n\n'
    "AFTER DELEGATING: Acknowledge in one short sentence of your own that fits the request, "
    "naming the task briefly if that reads naturally. For example: "
    '"Let me look into that", "Sure, I\'ll get on that now" or "Okay, pulling that up now". If '
    "you already acknowledged alongside the delegate call, even in part, the user has heard it, "
    'and this reply should *not* acknowledge again (though it can "finish the thought" of the '
    "previous acknowledgement, if appropriate). Saying it in other words is repeating yourself, "
    "and so is adding that you are on it now, that it will take a moment, or that you will "
    "report back when it is done; the initial acknowledgement already said all of that. After "
    '"Let me look that up for you!", e.g., there is nothing left to say, so say nothing. Finish '
    "the thought only when the user asked for something else that you have not answered yet. An "
    "acknowledgement is never a substitute for the delegate call. In the acknowledgement, avoid "
    "restating the request in full, or describing in detail what you are going to do, or "
    "following up on your own with promises, offers or questions: you've fired off the delegate "
    "call, so let it be. "
    "Then answer whatever else the user asked that you can. Never say work is done, or give a "
    "result, that the backend has not given you; if the user asks whether work is under way and "
    "you never delegated it, say so and delegate it then.\n\n"
    "BACKEND MESSAGES: The backend works on its own after you delegate and may be doing "
    "several things at once. What it has to say arrives as messages in the conversation "
    f'marked "{BACKEND_MESSAGE_PREFIX.strip()}": a result, a question for the user, or news '
    "worth an update, on whatever turn happens to be in progress by then. Such a message is "
    "owed to the user, whatever the conversation has moved on to: small talk, other "
    "questions or a long wait do not make it unwanted. If the user has said something you "
    "have not answered yet, answer that first, then relay the message at the end of that "
    "same reply, in your own words. If you have already answered everything the user said, "
    "relay just the message; do not repeat or rephrase your earlier reply. Relay each "
    "message once. Only when the user has cancelled or changed what they asked for does a "
    "message go unsaid, and then say only what still helps. Messages marked "
    f'"{BACKEND_WORKING_PREFIX.strip()}", "{BACKEND_NOTE_PREFIX.strip()}" and '
    f'"{BACKEND_THOUGHT_PREFIX.strip()}" are what the backend is doing and thinking. They '
    "carry no results: what a call found reaches you only in a message marked "
    f'"{BACKEND_MESSAGE_PREFIX.strip()}", so never describe or guess what a call found. Never '
    "relay them on their own, but when the user asks how the work is going, answer from the "
    "latest of them, concretely: name the step, such as which file it is reading or that it "
    "is running the tests, rather than saying only that it is still working. Never state a "
    "result you have not received from the backend, and never say work is done that the "
    "backend has not said is done.\n\n"
    "Whatever else a reply does, a user message about work you delegated, whether to add to "
    "it, change it or stop it, a never mind included, gets a delegate call in that reply: "
    "nothing else stops or changes that work."
)

#: The transcript strategy's guidance, in the pairing instruction's delegating paragraph.
TRANSCRIPT_FRONTEND_INSTRUCTION = (
    "A delegation hands over the whole conversation, not one item: the backend reads it and "
    "does everything in it that is its job, so delegate once per reply however many things "
    "the user asked for, and do not word the request."
)

#: How the transcript strategy describes the ``delegate`` tool.
TRANSCRIPT_TOOL_DESCRIPTION = (
    "The only way to start, change or stop the backend's work. Hands the conversation over to "
    "the backend, which reads it and does everything in it that needs a backend tool or "
    "careful reasoning, and stops or changes what the user has called off or changed. Call it "
    "as soon as any part of what the user says concerns that work, even while the backend is "
    "still working, and do the rest yourself. Not for a question about how the work is going, "
    "which you answer from the backend's notes. One handoff per reply, however many things the "
    "user asked for. It takes no arguments. Keep talking with the user while it works."
)

#: The explicit strategy's guidance, in the pairing instruction's delegating paragraph.
EXPLICIT_FRONTEND_INSTRUCTION = (
    "Word the request so it stands on its own: the user's goal, the exact "
    "details they gave and their latest correction, with everything new they asked for in "
    "the one request, so you delegate once per reply however many things that is. Work "
    "already handed over stays with the backend: a new request joins it, so do not restate "
    "earlier requests; a correction names what changes, and a stop names what to stop."
)

#: How the explicit strategy describes the ``delegate`` tool.
EXPLICIT_TOOL_DESCRIPTION = (
    "The only way to start, change or stop the backend's work. Hands a request to the "
    "backend, which does everything in it that needs a backend tool or careful reasoning, "
    "and stops or changes what the user has called off or changed. Call it as soon as any "
    "part of what the user says concerns that work, even while the backend is still working, "
    "with the request worded to stand on its own, and do the rest yourself. Not for a question "
    "about how the work is going, which you answer from the backend's notes. One call per "
    "reply, however many things the user asked for. Work already handed over stays with the "
    "backend, so send only what is new, including anything the user says about that work, "
    "such as to stop or change it. Keep talking with the user while it works."
)

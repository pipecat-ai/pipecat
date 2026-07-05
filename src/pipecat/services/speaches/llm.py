"""Speaches LLM Service — uses an OpenAI-compatible /v1/chat/completions endpoint."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService


def _stringify_content(content) -> str:
    """Best-effort flatten of an OpenAI message ``content`` to plain text.

    ``content`` may be a plain string, a list of content parts (each a dict with
    a ``"text"`` field for text parts), or ``None``. Non-text parts (images,
    etc.) are skipped — this helper exists only to fold a tool/assistant turn's
    text into an adjacent user/assistant turn so role alternation can hold.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def _describe_tool_calls(tool_calls) -> str:
    """Render an assistant message's ``tool_calls`` as readable inline text.

    Gemma (served by vLLM) has no ``tool``/``tool_calls`` wire role — the model
    was trained to emit calls inline as ``[func(arg=val)]`` text. When we have to
    fold an assistant-with-``tool_calls`` turn into a plain assistant turn (to
    preserve user/assistant alternation), reproduce that inline shape so the
    history still reflects that a call was made. Returns ``""`` if nothing usable.
    """
    rendered = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        rendered.append(f"{name}({args})" if args else f"{name}()")
    return f"[{', '.join(rendered)}]" if rendered else ""


def normalize_for_gemma(messages: list) -> list:
    """Rewrite a message list to be strict-alternation safe for vLLM/Gemma.

    vLLM serving Gemma enforces the Gemma chat template, which requires the
    conversation to alternate ``user`` / ``assistant`` / ``user`` / ... and to
    start (after the optional leading ``system`` message) with a ``user`` turn.
    It rejects the ENTIRE request with HTTP 400 ("Conversation roles must
    alternate user/assistant/user/assistant/...") on any violation — and that
    400 propagates as a fatal pipeline error that disconnects the live call.

    Three things routinely produce a non-alternating history in a live voice
    call, none of which Gemma tolerates:

    1. **Tool calls.** A tool call (e.g. ``web_search``/``end_call``, including
       the leaked-tool-call recovery in ``base_llm.py``) makes the assistant
       aggregator append ``assistant(tool_calls)`` then a ``tool`` (or
       ``developer``) result message — Gemma has neither a ``tool`` role nor a
       way to place two consecutive non-user turns.
    2. **Consecutive same-role turns.** Barge-ins, two quick user utterances, or
       an assistant reply that was recorded twice.
    3. **Assistant-first history.** The bot speaks first (greeting / node
       transition), so the history starts ``[system, assistant, user, ...]``.

    This normalizer makes EVERY request alternation-valid, mirroring the
    training-time normalization in ``training/train_qlora.py`` ``to_text()``
    (merge consecutive same-role turns, fold tool results inline, start with a
    user turn). It is deliberately universal — applied to tool and non-tool
    requests alike — so a single malformed turn can never 400-crash a call.

    Transformation, in order:

    - The optional leading ``system`` message is preserved as-is at the front.
    - ``assistant`` messages carrying ``tool_calls`` are flattened to a plain
      ``assistant`` turn whose content is the original text plus the inline
      ``[func(args)]`` rendering of the calls (matching how Gemma emits calls).
    - ``tool`` and ``developer`` (tool-result) messages are folded into the
      adjacent ``assistant`` turn (the model "speaking" the observed result),
      so they never appear as their own non-alternating role.
    - Remaining consecutive same-role turns are merged (content joined by a
      newline).
    - Leading ``assistant`` turns (before the first ``user`` turn) are dropped so
      the history starts with ``user``.

    The input list is never mutated; new message dicts are produced. Non-Gemma
    callers are unaffected because this is only wired into
    :class:`SpeachesLLMService`.
    """
    # 1. Peel off a single leading system message (Gemma allows one up front).
    system_msgs: list = []
    body = messages
    if body and isinstance(body[0], dict) and body[0].get("role") == "system":
        system_msgs = [dict(body[0])]
        body = body[1:]

    # 2. Map every remaining message to a plain ("user"|"assistant", text) turn.
    #    tool/developer results fold into assistant; assistant(tool_calls) flattens.
    turns: list[list] = []  # [role, content_text]
    for msg in body:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            # A stray mid-conversation system message: treat as assistant context
            # so it doesn't break alternation (Gemma has no mid-convo system role).
            text = _stringify_content(msg.get("content"))
            if text:
                turns.append(["assistant", text])
            continue
        if role in ("tool", "developer", "function"):
            # Tool result — fold into an assistant turn (Gemma has no tool role).
            text = _stringify_content(msg.get("content"))
            if text:
                turns.append(["assistant", text])
            continue
        # user / assistant
        text = _stringify_content(msg.get("content"))
        if role == "assistant" and msg.get("tool_calls"):
            inline = _describe_tool_calls(msg.get("tool_calls"))
            text = f"{text}\n{inline}".strip() if text else inline
        if not text:
            # Skip empty turns (e.g. an assistant frame that carried only a
            # now-flattened tool_call with no renderable name).
            continue
        norm_role = "assistant" if role == "assistant" else "user"
        turns.append([norm_role, text])

    # 3. Merge consecutive same-role turns.
    merged: list[list] = []
    for role, text in turns:
        if merged and merged[-1][0] == role:
            merged[-1][1] = f"{merged[-1][1]}\n{text}" if text else merged[-1][1]
        else:
            merged.append([role, text])

    # 4. Drop any leading assistant turns so the history starts with user.
    while merged and merged[0][0] != "user":
        merged.pop(0)

    out = system_msgs + [{"role": r, "content": c} for r, c in merged]
    return out


def _is_template_valid(messages: list) -> bool:
    """Return True if ``messages`` satisfies the gemma3 pythonic tool template.

    The template validates alternation over the PLAIN turns only — ``tool``
    messages and assistant turns carrying ``tool_calls`` are exempt — and the
    remaining plain turns must go user/assistant/user/... starting with user.
    Mirroring it exactly lets structured tool history pass through untouched,
    so the model sees its past calls in the trained ``[func(args)]`` rendering
    instead of a flattened text imitation it then copies into speech.
    """
    body = messages
    if body and isinstance(body[0], dict) and body[0].get("role") == "system":
        body = body[1:]
    expected = "user"
    for msg in body:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        if role == "tool":
            if not msg.get("tool_call_id"):
                return False  # wire format requires the id; normalize instead
            continue
        if role == "assistant" and msg.get("tool_calls"):
            continue
        if role not in ("user", "assistant"):
            return False  # developer / function / system mid-convo
        if role != expected:
            return False
        if not isinstance(msg.get("content"), str):
            return False  # multimodal/None plain content — normalize to text
        expected = "assistant" if expected == "user" else "user"
    return True


def normalize_preserving_tools(messages: list) -> list:
    """Alternation-fix a history WITHOUT flattening structured tool exchanges.

    The gemma3 pythonic tool template natively renders assistant ``tool_calls``
    (as ``[func(args)]``) and ``tool`` results (as ``<tool_response>`` blocks),
    and exempts both from its alternation rule — so they pass through as-is.
    Only the plain user/assistant turns are repaired: consecutive same-role
    plain turns merge, leading plain-assistant turns (before the first user)
    drop, and a same-role plain pair separated by tool traffic gains an empty
    opposite-role turn (renders as an empty model/user block — harmless).

    Flattening tool history into assistant text (the legacy
    :func:`normalize_for_gemma`) is actively harmful with the tool template:
    the model reads ``[func()] {"status": ...}`` as its own SPOKEN turns and
    starts emitting call syntax and invented result JSON in real replies.
    The input list is never mutated. May still produce a template-invalid
    shape in exotic cases — callers re-validate and fall back to the flattener.
    """
    system_msgs: list = []
    body = messages
    if body and isinstance(body[0], dict) and body[0].get("role") == "system":
        system_msgs = [dict(body[0])]
        body = body[1:]

    out: list = []
    last_plain_role: Optional[str] = None
    seen_user = False

    def _last_is_plain(role: str) -> bool:
        return (
            bool(out)
            and out[-1].get("role") == role
            and not out[-1].get("tool_calls")
            and out[-1].get("role") != "tool"
        )

    for msg in body:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            m = dict(msg)
            if not isinstance(m.get("content"), str):
                # The template raises on non-string content; None is common on
                # pure tool-call turns.
                m["content"] = _stringify_content(m.get("content"))
            out.append(m)
            continue

        if role in ("tool", "developer", "function"):
            if role == "tool" and msg.get("tool_call_id"):
                out.append(dict(msg))
            else:
                # No id to correlate — fold the result into an assistant turn
                # (legacy behavior) rather than send an invalid tool message.
                text = _stringify_content(msg.get("content"))
                if text:
                    if _last_is_plain("assistant"):
                        out[-1]["content"] = f"{out[-1]['content']}\n{text}"
                    else:
                        out.append({"role": "assistant", "content": text})
                        last_plain_role = "assistant"
            continue

        if role == "system":
            # Mid-conversation system message — Gemma has no such role; carry
            # it as assistant context.
            text = _stringify_content(msg.get("content"))
            if text:
                if _last_is_plain("assistant"):
                    out[-1]["content"] = f"{out[-1]['content']}\n{text}"
                else:
                    out.append({"role": "assistant", "content": text})
                    last_plain_role = "assistant"
            continue

        # Plain user / assistant turn.
        text = _stringify_content(msg.get("content"))
        if not text:
            continue
        norm_role = "assistant" if role == "assistant" else "user"
        if norm_role == "assistant" and not seen_user:
            continue  # leading assistant (greeting) — template needs user first
        if norm_role == "user":
            seen_user = True
        if _last_is_plain(norm_role):
            out[-1]["content"] = f"{out[-1]['content']}\n{text}"
            continue
        if last_plain_role == norm_role:
            # Same plain role again but separated by tool traffic — insert an
            # empty opposite-role turn to keep the template's alternation.
            filler = "assistant" if norm_role == "user" else "user"
            out.append({"role": filler, "content": ""})
        out.append({"role": norm_role, "content": text})
        last_plain_role = norm_role

    return system_msgs + out


def _is_already_alternating(messages: list) -> bool:
    """Return True if ``messages`` already satisfies Gemma's alternation rule.

    Used to skip the (lossy) flattening on the overwhelmingly common healthy
    request — only malformed histories pay the rewrite cost / fidelity loss.
    """
    body = messages
    if body and isinstance(body[0], dict) and body[0].get("role") == "system":
        body = body[1:]
    expected = "user"
    for msg in body:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            return False  # carries a tool_calls payload Gemma can't accept
        if role not in ("user", "assistant"):
            return False  # tool / developer / system mid-convo → not alternating
        if role != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


@dataclass
class SpeachesLLMSettings(OpenAILLMSettings):
    """Settings for Speaches LLM service."""

    pass


class SpeachesLLMService(OpenAILLMService):
    """Speaches LLM service using an OpenAI-compatible chat completions endpoint.

    Targets vLLM serving a Gemma model, which enforces strict user/assistant
    role alternation and rejects any violation with HTTP 400. Every outgoing
    request is normalized to be alternation-safe (see :func:`normalize_for_gemma`),
    and a 400 alternation error degrades gracefully (retry once, then a recovery
    line) instead of disconnecting the call.
    """

    Settings = SpeachesLLMSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:11434/v1",
        settings: SpeachesLLMSettings | None = None,
        **kwargs,
    ):
        """Initialize the Speaches LLM service.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the Speaches-compatible endpoint.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=settings,
            **kwargs,
        )

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Build request params, enforcing strict user/assistant role alternation.

        vLLM/Gemma returns HTTP 400 on non-alternating histories. This override
        is the lowest safe choke point in the request path: it runs for BOTH
        streaming completions (``get_chat_completions``) and ``run_inference``,
        since both build their request through this method, so EVERY request —
        tool call or not — leaves here alternation-valid.

        See :func:`normalize_for_gemma` for the exact transformation.
        """
        params = super().build_chat_completion_params(params_from_context)
        messages = params.get("messages")
        if isinstance(messages, list) and len(messages) > 1:
            if not _is_template_valid(messages):
                normalized = normalize_preserving_tools(messages)
                if not _is_template_valid(normalized):
                    # Exotic shape the structure-preserving pass couldn't fix —
                    # flatten everything (legacy) rather than risk a 400.
                    normalized = normalize_for_gemma(messages)
                logger.debug(
                    f"{self}: normalized history for the Gemma tool template "
                    f"({len(messages)} → {len(normalized)} msgs)"
                )
                params["messages"] = normalized
        return params

    async def get_chat_completions(self, context):
        """Stream chat completions, degrading gracefully on a 400 alternation error.

        DEFENSE IN DEPTH: even though :meth:`build_chat_completion_params`
        normalizes every request, we wrap the actual API call so that a vLLM
        ``BadRequestError`` (HTTP 400) — alternation or otherwise — can NEVER
        propagate as a fatal pipeline error that disconnects the live call.

        On a 400 we retry the call ONCE against an aggressively re-normalized
        context (built from ``context.get_messages()`` via
        :func:`normalize_for_gemma`). If that still fails, we re-raise so the
        caller can recover (``base_llm`` turns it into a non-fatal error frame),
        but the common alternation case is healed silently and the call survives.
        """
        try:
            return await super().get_chat_completions(context)
        except Exception as e:  # noqa: BLE001 — we re-raise unless we can recover
            if not _is_bad_request(e):
                raise
            logger.warning(
                f"{self}: LLM returned 400 ({_error_text(e)}); retrying once with "
                f"re-normalized (Gemma-safe) messages instead of dropping the call"
            )
            try:
                raw = list(context.get_messages())
            except Exception:  # noqa: BLE001
                raise e
            safe = normalize_for_gemma(_strip_specific_messages(raw))
            adapter = self.get_llm_adapter()
            from pipecat.services.settings import assert_given

            params_from_context = adapter.get_llm_invocation_params(
                context,
                system_instruction=assert_given(self._settings.system_instruction),
                convert_developer_to_user=not self.supports_developer_role,
            )
            params = super().build_chat_completion_params(params_from_context)
            params["messages"] = safe
            return await self._client.chat.completions.create(**params)


def _strip_specific_messages(messages: list) -> list:
    """Drop ``LLMSpecificMessage`` wrappers, keeping only plain dict messages.

    ``LLMContext.get_messages()`` can return provider-specific message wrappers
    alongside standard dict messages; the normalizer only understands dicts.
    """
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
    return out


def _is_bad_request(exc: Exception) -> bool:
    """True if ``exc`` is an OpenAI-client HTTP 400 (BadRequestError)."""
    # Match by attribute rather than import so this stays robust across openai
    # client versions and never itself raises if the class moved.
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    return type(exc).__name__ == "BadRequestError"


def _error_text(exc: Exception) -> str:
    """A short single-line description of an API error for logging."""
    msg = str(exc)
    return msg.splitlines()[0][:200] if msg else type(exc).__name__

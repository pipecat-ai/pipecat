#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Masks credentials in text bound for logs and error reports.

Error messages often quote whatever a failure carried: an SDK exception that
prints the headers it sent, a URL signed for one download, a provider body
echoing the key it rejected. `redact_secrets` masks the credential and keeps
the rest, so a message still says what went wrong.

It masks three things:

- The value of any ``name=value`` or ``name: value`` pair, quoted or not,
  whose name marks a credential (``api_key``, ``Authorization``,
  ``X-Amz-Signature``, ``apiKey``...). This covers query strings, printed
  header dicts and plain prose alike. An exception class name followed by
  its message (``KeyError: ...``) is not a credential name.
- A URL's password (``scheme://user:password@host``).
- Tokens recognizable by their format alone: provider key prefixes, AWS
  access key IDs and JWTs.
"""

import re

REDACTED = "[REDACTED]"

# A name marks a credential when one of its words is one of these, so
# `x-api-key` and `X-Amz-Security-Token` do while `keyterms` and `max_tokens`
# don't.
_CREDENTIAL_WORDS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "creds",
        "jwt",
        "key",
        "passwd",
        "password",
        "pwd",
        "secret",
        "sig",
        "signature",
        "token",
        # Google's resumable upload session ID, which authorizes the upload by itself.
        "uploadid",
    }
)

# The start of a `name=value` or `name: value` pair (or `:=`, `=>`), the name
# optionally quoted as in a printed dict. An auth scheme (`Bearer`, `Basic`...)
# ahead of the value is kept.
_PAIR_START = re.compile(
    r"""(?P<name>(?<![\w.-])(?P<nq>["']?)[A-Za-z][\w.-]*(?P=nq))"""
    r"""(?P<sep>\s*(?::=|=>|[:=])\s*)"""
    r"""(?P<scheme>(?:Bearer|Basic|Token|Api-Token|Key|Digest|AWS4-HMAC-SHA256)\s+)?"""
)

# A pair's value, quoted or running to the next delimiter. A quoted value cut
# short before its closing quote, as a truncated repr leaves it, runs to the
# next delimiter instead. A value already redacted is matched whole, so text
# redacted twice reads the same as text redacted once.
_VALUE = re.compile(
    r"""(?P<vq>["'])(?P<quoted>.*?)(?P=vq)"""
    r"""|(?P<oq>["']?)(?P<bare>(?:\[REDACTED\]|[^\s,;&"'}\])\[])+)"""
)

_URL_PASSWORD = re.compile(r"(?P<prefix>\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s/@]+(?=@)", re.I)

# Formats after gitleaks' rules for the same credentials
# (github.com/gitleaks/gitleaks, config/gitleaks.toml), loosened where a
# looser match costs nothing: a log line is masked, not scanned for findings.
_KNOWN_TOKENS = re.compile(
    r"""\b(?:"""
    r"""sk-[A-Za-z0-9_-]{20,}"""  # OpenAI, Anthropic
    r"""|(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}"""  # AWS access key IDs
    r"""|ABSK[A-Za-z0-9+/]{109,269}={0,2}"""  # Amazon Bedrock API keys, long-lived
    r"""|bedrock-api-key-[A-Za-z0-9+/=_-]+"""  # Amazon Bedrock API keys, short-lived
    r"""|AIza[0-9A-Za-z_-]{35}"""  # Google API keys
    r"""|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{82,}"""  # GitHub
    r"""|xox[abeoprs]-[A-Za-z0-9-]{10,}|xapp-\d-[A-Za-z0-9-]{10,}"""  # Slack
    # Slack webhooks, whose path is the secret
    r"""|(?:(?<=hooks\.slack\.com/services/)|(?<=hooks\.slack\.com/workflows/)"""
    r"""|(?<=hooks\.slack\.com/triggers/))[A-Za-z0-9+/]{20,}"""
    r"""|(?:hf|api_org)_[A-Za-z]{34}"""  # Hugging Face
    r"""|pplx-[A-Za-z0-9]{48}"""  # Perplexity
    r"""|SK[0-9a-fA-F]{32}"""  # Twilio API keys
    r"""|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"""  # JWTs, signed or not
    r""")"""
    # A PEM private key, through its END line or, cut short, the end of the text.
    r"""|-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----"""
    r"""(?:[\s\S]*?-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----|[\s\S]*)"""
)

# An exception or warning class name, optionally module-qualified (`KeyError`,
# `jwt.exceptions.InvalidTokenError`). It names what went wrong, so the text
# after it is the explanation, not a credential.
_EXCEPTION_NAME = re.compile(r"(?:[a-z_][a-z0-9_]*\.)*[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning)")


def _is_credential_name(name: str) -> bool:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip("\"'")).lower()
    parts = [p for p in re.split(r"[-_.]", words) if p]
    return "".join(parts) in _CREDENTIAL_WORDS or any(p in _CREDENTIAL_WORDS for p in parts)


def _is_exception_name(match: re.Match) -> bool:
    return (
        not match["nq"]
        and not match["scheme"]
        and match["sep"].strip() == ":"
        and _EXCEPTION_NAME.fullmatch(match["name"]) is not None
    )


def _masks_value(start: re.Match) -> bool:
    return _is_credential_name(start["name"]) and not _is_exception_name(start)


def _redact_pairs(text: str) -> str:
    # Only a credential's value is taken whole. Any other pair's value is read on
    # as ordinary text, so the pairs it holds (`URL: https://...?token=...`,
    # `error: Authorization: Bearer ...`) are found in place.
    out = []
    pos = 0
    while start := _PAIR_START.search(text, pos):
        value = _VALUE.match(text, start.end()) if _masks_value(start) else None
        # A credential's value that is only the next credential's name, as
        # `token:` is in `api_key: token: abc`, gives way to that credential.
        following = value and _PAIR_START.match(text, start.end())
        if not value or (following and _masks_value(following) and following.end() > value.end()):
            out.append(text[pos : start.end()])
            pos = start.end()
            continue
        quote = value["vq"] or ""
        out.append(f"{text[pos : start.end()]}{quote or value['oq']}{REDACTED}{quote}")
        pos = value.end()
    out.append(text[pos:])
    return "".join(out)


def redact_secrets(text: str) -> str:
    """Mask the credentials in `text`, leaving the rest as written.

    Args:
        text: Text that may quote a credential, such as an error message.

    Returns:
        The text with each credential replaced by ``[REDACTED]``.
    """
    # Known formats go first: a PEM key spans lines, and masking its first line
    # as a pair's value would leave the rest unrecognizable.
    text = _KNOWN_TOKENS.sub(REDACTED, text)
    text = _redact_pairs(text)
    return _URL_PASSWORD.sub(rf"\g<prefix>{REDACTED}", text)

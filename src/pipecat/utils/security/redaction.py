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
  header dicts and plain prose alike.
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

# `name=value` or `name: value` (or `:=`, `=>`), the name optionally quoted as
# in a printed dict, the value quoted or running to the next delimiter. A
# quoted value cut short before its closing quote, as a truncated repr leaves
# it, runs to the next delimiter instead. An auth scheme (`Bearer`, `Basic`...)
# ahead of the value is kept. A value already redacted is matched whole, so
# text redacted twice reads the same as text redacted once.
_PAIR = re.compile(
    r"""(?P<name>(?<![\w.-])(?P<nq>["']?)[A-Za-z][\w.-]*(?P=nq))"""
    r"""(?P<sep>\s*(?::=|=>|[:=])\s*)"""
    r"""(?P<scheme>(?:Bearer|Basic|Token|Api-Token|Key|Digest|AWS4-HMAC-SHA256)\s+)?"""
    r"""(?:(?P<vq>["'])(?P<quoted>.*?)(?P=vq)"""
    r"""|(?P<oq>["']?)(?P<bare>(?:\[REDACTED\]|[^\s,;&"'}\])\[])+))"""
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


def _is_credential_name(name: str) -> bool:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip("\"'")).lower()
    parts = [p for p in re.split(r"[-_.]", words) if p]
    return "".join(parts) in _CREDENTIAL_WORDS or any(p in _CREDENTIAL_WORDS for p in parts)


def _redact_pair(match: re.Match) -> str:
    quote = match["vq"] or ""
    if _is_credential_name(match["name"]):
        value = REDACTED
    else:
        # The value may itself hold pairs, as `URL: https://...?token=...` does.
        value = _PAIR.sub(_redact_pair, match["quoted"] or match["bare"] or "")
    opening = match["oq"] or ""
    return f"{match['name']}{match['sep']}{match['scheme'] or ''}{quote or opening}{value}{quote}"


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
    text = _PAIR.sub(_redact_pair, text)
    return _URL_PASSWORD.sub(rf"\g<prefix>{REDACTED}", text)

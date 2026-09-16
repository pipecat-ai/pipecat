#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared pronunciation parsing and normalization utilities for TTS services.

A pronunciation reaches a TTS service as a phoneme string in one alphabet
(:class:`PhonemeAlphabet`). Services each want their own markup for it, but they
all start from the same steps: tidy the notation, split it into phones, and find
which vowel each stress mark belongs to. Those steps live here, so a service's
formatter only has to render the result.

Normalization is notation-only. It unifies ways of writing the same symbol
(``ʧ`` and ``tʃ``, ``g`` and ``ɡ``, ``'`` and ``ˈ``) and never changes which sound
is written, so it is safe for any language. Arpabet conversion is the exception:
Arpabet only exists for English, so it follows English conventions.
"""

import re
import unicodedata
from enum import StrEnum


class PhonemeAlphabet(StrEnum):
    """Alphabets a pronunciation can be written in.

    Parameters:
        IPA: The International Phonetic Alphabet, e.g. ``mɛtˈfɔɹmɪn``.
        ARPABET: CMU Arpabet with stress digits on vowels, e.g.
            ``M EH0 T F AO1 R M IH0 N``. English only.
    """

    IPA = "ipa"
    ARPABET = "arpabet"


PRIMARY_STRESS = "ˈ"
SECONDARY_STRESS = "ˌ"
STRESS_MARKS = (PRIMARY_STRESS, SECONDARY_STRESS)

TIE_BAR = "\u0361"

# Ligatures and ASCII look-alikes of IPA symbols. "tʃ" and "dʒ" are always read
# as one phone, so they are written untied; other affricates keep the tie bar
# that tells them apart from a cluster.
_IPA_NOTATION = [
    ("\u035c", TIE_BAR),  # tie bar below
    ("ʧ", "tʃ"),
    ("ʤ", "dʒ"),
    (f"t{TIE_BAR}ʃ", "tʃ"),
    (f"d{TIE_BAR}ʒ", "dʒ"),
    ("ʦ", f"t{TIE_BAR}s"),
    ("ʣ", f"d{TIE_BAR}z"),
    ("g", "ɡ"),
    ("'", PRIMARY_STRESS),
    (":", "ː"),
    (".", ""),  # syllable break
]

# Sequences read as a single phone even without a tie bar: affricates written
# this way in nearly every dictionary, and the English diphthongs. Anything else
# ("ts" in "cats" against Italian "pizza") is one phone only when tied.
_IPA_MULTI = ["tʃ", "dʒ", "aɪ", "aʊ", "ɔɪ", "oʊ", "eɪ"]

_IPA_VOWELS = set("aeiouyæɑɒɔəɚɛɜɝɪʊʌɐᵻɨʉɯɤøœɶɵɘɞ")

# Characters that belong to the phone before them rather than starting a new one.
_IPA_MODIFIERS = set("ːˑ")


def normalize_ipa(ipa: str) -> str:
    """Unify the notation of an IPA string without changing what it says.

    Strips surrounding ``/…/`` or ``[…]``, composes Unicode, replaces ligatures and
    ASCII look-alikes with their IPA symbols, writes ``tʃ`` and ``dʒ`` untied and
    other affricates tied (``ʦ`` becomes ``t͡s``), drops syllable breaks, and
    collapses whitespace.

    Args:
        ipa: An IPA transcription.

    Returns:
        The same transcription in canonical notation.

    Example::

        normalize_ipa("/ˈʧɪ.kən/")  # "ˈtʃɪkən"
    """
    ipa = unicodedata.normalize("NFC", ipa.strip())
    if len(ipa) >= 2 and (ipa[0], ipa[-1]) in (("/", "/"), ("[", "]")):
        ipa = ipa[1:-1]
    for old, new in _IPA_NOTATION:
        ipa = ipa.replace(old, new)
    return " ".join(ipa.split())


def ipa_phones(ipa: str) -> list[str]:
    """Split one IPA word into phones, in written order.

    Stress marks are returned as tokens of their own, where they were written.
    Tied sequences (``t͡s``), ``tʃ``, ``dʒ`` and the English diphthongs are one
    phone each, returned without the tie bar; length marks and combining
    diacritics stay with the phone they modify.

    Args:
        ipa: An IPA transcription of a single word.

    Returns:
        The phones and stress marks of the word.

    Example::

        ipa_phones("mɛtˈfɔɹmɪn")  # ["m", "ɛ", "t", "ˈ", "f", "ɔ", "ɹ", "m", "ɪ", "n"]
    """
    ipa = normalize_ipa(ipa).replace(" ", "")
    phones: list[str] = []
    i = 0
    while i < len(ipa):
        char = ipa[i]
        if char in STRESS_MARKS:
            phones.append(char)
            i += 1
            continue
        if ipa.startswith(TIE_BAR, i + 1) and i + 2 < len(ipa):
            phone = char + ipa[i + 2]
            i += 3
        else:
            phone = next((m for m in _IPA_MULTI if ipa.startswith(m, i)), char)
            i += len(phone)
        while i < len(ipa) and (ipa[i] in _IPA_MODIFIERS or unicodedata.category(ipa[i]) == "Mn"):
            phone += ipa[i]
            i += 1
        if phones and phones[-1] not in STRESS_MARKS and _is_modifier_only(phone):
            phones[-1] += phone
        else:
            phones.append(phone)
    return phones


def is_ipa_vowel(phone: str) -> bool:
    """Whether an IPA phone (as returned by :func:`ipa_phones`) is a vowel."""
    return bool(phone) and phone[0] in _IPA_VOWELS


def stress_before_vowels(phones: list[str]) -> list[str]:
    """Move each stress mark from where it was written to just before its vowel.

    Transcriptions usually put stress at the start of the syllable
    (``mɛtˈfɔɹmɪn``); some services want it on the vowel itself
    (``mɛtfˈɔɹmɪn``). A stress mark with no vowel after it is dropped.

    Args:
        phones: Phones and stress marks, as returned by :func:`ipa_phones`.

    Returns:
        The same phones with every stress mark directly before a vowel.
    """
    result: list[str] = []
    pending: str | None = None
    for phone in phones:
        if phone in STRESS_MARKS:
            pending = phone
            continue
        if pending and is_ipa_vowel(phone):
            result.append(pending)
            pending = None
        result.append(phone)
    return result


def _is_modifier_only(phone: str) -> bool:
    return all(c in _IPA_MODIFIERS or unicodedata.category(c) == "Mn" for c in phone)


# --- Arpabet ------------------------------------------------------------------

ARPABET_VOWELS = frozenset(
    ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
)
ARPABET_CONSONANTS = frozenset(
    ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
     "T", "TH", "V", "W", "Y", "Z", "ZH"]
)  # fmt: skip

_ARPABET_TOKEN_RE = re.compile(r"^([A-Z]{1,2})([012])?$")

# General American IPA -> Arpabet. Keys are phones as :func:`ipa_phones` returns
# them, with length marks and diacritics removed.
_IPA_TO_ARPABET = {
    "ɑ": "AA", "a": "AA", "ɒ": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɐ": "AH",
    "ɔ": "AO", "aʊ": "AW", "aɪ": "AY", "ɛ": "EH", "ɚ": "ER", "ɝ": "ER", "ɜ": "ER",
    "eɪ": "EY", "e": "EY", "ɪ": "IH", "ᵻ": "IH", "ɨ": "IH", "i": "IY", "oʊ": "OW",
    "o": "OW", "ɔɪ": "OY", "ʊ": "UH", "u": "UW",
    "b": "B", "tʃ": "CH", "d": "D", "ð": "DH", "f": "F", "ɡ": "G", "h": "HH",
    "dʒ": "JH", "k": "K", "l": "L", "ɫ": "L", "m": "M", "n": "N", "ŋ": "NG", "p": "P",
    "ɹ": "R", "r": "R", "s": "S", "ʃ": "SH", "t": "T", "ɾ": "T", "θ": "TH", "v": "V",
    "w": "W", "j": "Y", "z": "Z", "ʒ": "ZH",
}  # fmt: skip


# Vowels that, followed by R, are written as the single vowel ER.
_R_COLOURING = ("AH", "ER")


def normalize_arpabet(arpabet: str) -> str | None:
    """Canonical form of an Arpabet transcription, or None if it is not valid.

    Upper-cases, separates phones with single spaces, and checks every phone:
    vowels must carry a stress digit, consonants must not. A single-vowel word
    written without a digit gets primary stress, the only reading it can have.

    Args:
        arpabet: An Arpabet transcription, e.g. ``"m eh0 t f ao1 r m ih0 n"``.

    Returns:
        The transcription in canonical form, or None if a phone is unknown or a
        stress digit is missing or misplaced.
    """
    tokens = arpabet.upper().split()
    vowels = [t for t in tokens if t.rstrip("012") in ARPABET_VOWELS]
    result = []
    for token in tokens:
        match = _ARPABET_TOKEN_RE.match(token)
        if not match:
            return None
        phone, stress = match.groups()
        if phone in ARPABET_CONSONANTS and stress is None:
            result.append(phone)
        elif phone in ARPABET_VOWELS and stress is not None:
            result.append(token)
        elif phone in ARPABET_VOWELS and len(vowels) == 1:
            result.append(f"{phone}1")
        else:
            return None
    return " ".join(result) if result else None


def ipa_to_arpabet(ipa: str) -> str | None:
    """Convert a General American IPA word to Arpabet with stress digits.

    A central vowel followed by ``ɹ`` is written ``ER``, as the CMU dictionary
    does ("aspirin" is ``AE1 S P ER0 IH0 N``), unless a stress mark between them
    makes the ``ɹ`` the start of the next syllable (``zəˈɹɛltoʊ`` keeps
    ``Z AH0 R EH1 L T OW0``).

    Args:
        ipa: An IPA transcription of a single English word.

    Returns:
        The Arpabet transcription, or None when the IPA has a sound Arpabet cannot
        write.

    Example::

        ipa_to_arpabet("mɛtˈfɔɹmɪn")  # "M EH0 T F AO1 R M IH0 N"
    """
    codes: list[str] = []
    stress = "0"
    syllable_break = False
    for phone in ipa_phones(ipa):
        if phone in STRESS_MARKS:
            stress = "1" if phone == PRIMARY_STRESS else "2"
            syllable_break = True
            continue
        base = "".join(
            c for c in phone if c not in _IPA_MODIFIERS and unicodedata.category(c) != "Mn"
        )
        if base == "ʔ":
            continue
        code = _IPA_TO_ARPABET.get(base)
        if code is None:
            return None
        if code in ARPABET_VOWELS:
            codes.append(code + stress)
            stress = "0"
            syllable_break = False
        elif code == "R" and codes and codes[-1][:2] in _R_COLOURING and not syllable_break:
            codes[-1] = "ER" + codes[-1][2]
        else:
            codes.append(code)
    return normalize_arpabet(" ".join(codes))

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.utils.text.transforms._alnum_utils import normalize
from pipecat.utils.text.transforms.acronyms import normalize_acronyms
from pipecat.utils.text.transforms.email import email_to_speech
from pipecat.utils.text.transforms.percentages import expand_percentages
from pipecat.utils.text.transforms.phone import expand_phone_numbers
from pipecat.utils.text.transforms.replacements import replace_text
from pipecat.utils.text.transforms.strip_markdown import strip_markdown
from pipecat.utils.text.transforms.units import expand_units
from pipecat.utils.text.transforms.voice_formatter import VoiceFormatter


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStripMarkdown(unittest.TestCase):
    def test_bold(self):
        result = run(strip_markdown("**Hello** world", "*"))
        self.assertEqual(result, "Hello world")

    def test_italic_asterisk(self):
        result = run(strip_markdown("*italic*", "*"))
        self.assertEqual(result, "italic")

    def test_italic_underscore(self):
        result = run(strip_markdown("_italic_", "*"))
        self.assertEqual(result, "italic")

    def test_headers(self):
        result = run(strip_markdown("# Title\n\nBody", "*"))
        self.assertNotIn("#", result)
        self.assertIn("Title", result)

    def test_inline_code(self):
        result = run(strip_markdown("`code`", "*"))
        self.assertEqual(result, "code")

    def test_alnum_preserving(self):
        text = "**Hello** and _world_"
        result = run(strip_markdown(text, "*"))
        self.assertEqual(normalize(result), normalize(text))

    def test_no_change_to_plain_text(self):
        text = "Hello world"
        result = run(strip_markdown(text, "*"))
        self.assertEqual(result, text)


class TestExpandPhoneNumbers(unittest.TestCase):
    def test_dashes(self):
        result = run(expand_phone_numbers("Call 123-456-7890", "*"))
        self.assertIn("1 2 3", result)

    def test_dots(self):
        result = run(expand_phone_numbers("123.456.7890", "*"))
        self.assertIn("1 2 3", result)

    def test_alnum_preserving(self):
        text = "123-456-7890"
        result = run(expand_phone_numbers(text, "*"))
        self.assertEqual(normalize(result), normalize(text))

    def test_no_change_to_non_phone(self):
        text = "Hello world"
        result = run(expand_phone_numbers(text, "*"))
        self.assertEqual(result, text)


class TestNormalizeAcronyms(unittest.TestCase):
    def test_nasa(self):
        result = run(normalize_acronyms("NASA launched", "*"))
        self.assertIn("N A S A", result)

    def test_http(self):
        result = run(normalize_acronyms("HTTP request", "*"))
        self.assertIn("H T T P", result)

    def test_alnum_preserving(self):
        text = "NASA and HTTP"
        result = run(normalize_acronyms(text, "*"))
        self.assertEqual(normalize(result), normalize(text))

    def test_does_not_split_camelcase(self):
        text = "iPhone"
        result = run(normalize_acronyms(text, "*"))
        self.assertEqual(result, text)


class TestReplaceText(unittest.TestCase):
    def test_exact_match(self):
        transform = replace_text([(r"\bDr\.", "Doctor")])
        result = run(transform("Dr. Smith", "*"))
        self.assertIn("Doctor", result)

    def test_multiple_replacements(self):
        transform = replace_text([(r"\bSt\.", "Street"), (r"\bvs\b", "versus")])
        result = run(transform("Main St. vs Oak St.", "*"))
        self.assertIn("Street", result)
        self.assertIn("versus", result)

    def test_no_match_unchanged(self):
        transform = replace_text([("xyz", "abc")])
        result = run(transform("Hello world", "*"))
        self.assertEqual(result, "Hello world")


class TestExpandPercentages(unittest.TestCase):
    def test_integer_percent(self):
        result = run(expand_percentages("50% off", "*"))
        self.assertIn("percent", result)
        self.assertNotIn("%", result)

    def test_decimal_percent(self):
        result = run(expand_percentages("3.5% rate", "*"))
        self.assertIn("percent", result)

    def test_alnum_changes(self):
        text = "50%"
        result = run(expand_percentages(text, "*"))
        self.assertNotEqual(normalize(result), normalize(text))


class TestExpandUnits(unittest.TestCase):
    def test_km(self):
        result = run(expand_units("5km away", "*"))
        self.assertIn("kilometers", result)

    def test_mph(self):
        result = run(expand_units("60mph speed", "*"))
        self.assertIn("miles per hour", result)

    def test_gb(self):
        result = run(expand_units("2GB file", "*"))
        self.assertIn("gigabytes", result)

    def test_no_match_unchanged(self):
        result = run(expand_units("Hello world", "*"))
        self.assertEqual(result, "Hello world")


class TestEmailToSpeech(unittest.TestCase):
    def test_simple_email(self):
        result = run(email_to_speech("user@example.com", "*"))
        self.assertIn("at", result)
        self.assertNotIn("@", result)

    def test_dot_expanded(self):
        result = run(email_to_speech("user@example.com", "*"))
        self.assertIn("dot", result)

    def test_embedded_in_sentence(self):
        result = run(email_to_speech("Email user@example.com today", "*"))
        self.assertIn("at", result)
        self.assertIn("Email", result)


class TestVoiceFormatter(unittest.TestCase):
    def test_default_strips_markdown(self):
        formatter = VoiceFormatter()
        result = run(formatter("**Hello** world", "*"))
        self.assertNotIn("**", result)
        self.assertIn("Hello", result)

    def test_disabled_option_skipped(self):
        formatter = VoiceFormatter(strip_markdown=False)
        result = run(formatter("**Hello**", "*"))
        self.assertIn("**", result)

    def test_custom_replacements_applied(self):
        formatter = VoiceFormatter(
            strip_markdown=False,
            expand_phone_numbers=False,
            normalize_acronyms=False,
            expand_currency=False,
            expand_percentages=False,
            expand_units=False,
            email_to_speech=False,
            normalize_dates=False,
            custom_replacements=[(r"\bDr\.", "Doctor")],
        )
        result = run(formatter("Dr. Smith", "*"))
        self.assertIn("Doctor", result)

    def test_all_disabled_no_change(self):
        formatter = VoiceFormatter(
            strip_markdown=False,
            expand_phone_numbers=False,
            normalize_acronyms=False,
            expand_currency=False,
            expand_percentages=False,
            expand_units=False,
            email_to_speech=False,
            normalize_dates=False,
            expand_numbers=False,
        )
        text = "Hello world"
        result = run(formatter(text, "*"))
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()

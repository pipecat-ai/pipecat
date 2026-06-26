#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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


class TestStripMarkdown(unittest.IsolatedAsyncioTestCase):
    async def test_bold(self):
        result = await strip_markdown("**Hello** world", "*")
        self.assertEqual(result, "Hello world")

    async def test_italic_asterisk(self):
        result = await strip_markdown("*italic*", "*")
        self.assertEqual(result, "italic")

    async def test_italic_underscore(self):
        result = await strip_markdown("_italic_", "*")
        self.assertEqual(result, "italic")

    async def test_headers(self):
        result = await strip_markdown("# Title\n\nBody", "*")
        self.assertNotIn("#", result)
        self.assertIn("Title", result)

    async def test_inline_code(self):
        result = await strip_markdown("`code`", "*")
        self.assertEqual(result, "code")

    async def test_alnum_preserving(self):
        text = "**Hello** and _world_"
        result = await strip_markdown(text, "*")
        self.assertEqual(normalize(result), normalize(text))

    async def test_no_change_to_plain_text(self):
        text = "Hello world"
        result = await strip_markdown(text, "*")
        self.assertEqual(result, text)


class TestExpandPhoneNumbers(unittest.IsolatedAsyncioTestCase):
    async def test_dashes(self):
        result = await expand_phone_numbers("Call 123-456-7890", "*")
        self.assertIn("1 2 3", result)

    async def test_dots(self):
        result = await expand_phone_numbers("123.456.7890", "*")
        self.assertIn("1 2 3", result)

    async def test_alnum_preserving(self):
        text = "123-456-7890"
        result = await expand_phone_numbers(text, "*")
        self.assertEqual(normalize(result), normalize(text))

    async def test_no_change_to_non_phone(self):
        text = "Hello world"
        result = await expand_phone_numbers(text, "*")
        self.assertEqual(result, text)


class TestNormalizeAcronyms(unittest.IsolatedAsyncioTestCase):
    async def test_nasa(self):
        result = await normalize_acronyms("NASA launched", "*")
        self.assertIn("N A S A", result)

    async def test_http(self):
        result = await normalize_acronyms("HTTP request", "*")
        self.assertIn("H T T P", result)

    async def test_alnum_preserving(self):
        text = "NASA and HTTP"
        result = await normalize_acronyms(text, "*")
        self.assertEqual(normalize(result), normalize(text))

    async def test_does_not_split_camelcase(self):
        text = "iPhone"
        result = await normalize_acronyms(text, "*")
        self.assertEqual(result, text)


class TestReplaceText(unittest.IsolatedAsyncioTestCase):
    async def test_exact_match(self):
        transform = replace_text([(r"\bDr\.", "Doctor")])
        result = await transform("Dr. Smith", "*")
        self.assertIn("Doctor", result)

    async def test_multiple_replacements(self):
        transform = replace_text([(r"\bSt\.", "Street"), (r"\bvs\b", "versus")])
        result = await transform("Main St. vs Oak St.", "*")
        self.assertIn("Street", result)
        self.assertIn("versus", result)

    async def test_no_match_unchanged(self):
        transform = replace_text([("xyz", "abc")])
        result = await transform("Hello world", "*")
        self.assertEqual(result, "Hello world")


class TestExpandPercentages(unittest.IsolatedAsyncioTestCase):
    async def test_integer_percent(self):
        result = await expand_percentages("50% off", "*")
        self.assertIn("percent", result)
        self.assertNotIn("%", result)

    async def test_decimal_percent(self):
        result = await expand_percentages("3.5% rate", "*")
        self.assertIn("percent", result)

    async def test_alnum_changes(self):
        text = "50%"
        result = await expand_percentages(text, "*")
        self.assertNotEqual(normalize(result), normalize(text))


class TestExpandUnits(unittest.IsolatedAsyncioTestCase):
    async def test_km(self):
        result = await expand_units("5km away", "*")
        self.assertIn("kilometers", result)

    async def test_mph(self):
        result = await expand_units("60mph speed", "*")
        self.assertIn("miles per hour", result)

    async def test_gb(self):
        result = await expand_units("2GB file", "*")
        self.assertIn("gigabytes", result)

    async def test_no_match_unchanged(self):
        result = await expand_units("Hello world", "*")
        self.assertEqual(result, "Hello world")

    async def test_ambiguous_in_preposition_not_expanded(self):
        """'in' as a preposition after a number must not be treated as inches."""
        result = await expand_units("1 in 5 people", "*")
        self.assertNotIn("inches", result)
        self.assertEqual(result, "1 in 5 people")

    async def test_ambiguous_in_ratio_not_expanded(self):
        """'9 in 10 dentists' must not expand 'in' to 'inches'."""
        result = await expand_units("9 in 10 dentists recommend it", "*")
        self.assertNotIn("inches", result)

    async def test_ambiguous_m_not_expanded_as_meters(self):
        """'m' alone after a number is highly ambiguous and must not be auto-expanded."""
        result = await expand_units("1 m people voted", "*")
        self.assertNotIn("meters", result)

    async def test_ambiguous_g_not_expanded_as_grams(self):
        """'g' standing alone in prose must not expand to 'grams'."""
        result = await expand_units("1 g of something", "*")
        self.assertNotIn("grams", result)


class TestExpandUnitsUnambiguous(unittest.IsolatedAsyncioTestCase):
    """Units that are unambiguous even with a space before them should still expand."""

    async def test_km_with_space(self):
        result = await expand_units("5 km away", "*")
        self.assertIn("kilometers", result)

    async def test_mph_with_space(self):
        result = await expand_units("60 mph speed limit", "*")
        self.assertIn("miles per hour", result)

    async def test_ghz_with_space(self):
        result = await expand_units("3 GHz processor", "*")
        self.assertIn("gigahertz", result)


class TestEmailToSpeech(unittest.IsolatedAsyncioTestCase):
    async def test_simple_email(self):
        result = await email_to_speech("user@example.com", "*")
        self.assertIn("at", result)
        self.assertNotIn("@", result)

    async def test_dot_expanded(self):
        result = await email_to_speech("user@example.com", "*")
        self.assertIn("dot", result)

    async def test_embedded_in_sentence(self):
        result = await email_to_speech("Email user@example.com today", "*")
        self.assertIn("at", result)
        self.assertIn("Email", result)


class TestVoiceFormatter(unittest.IsolatedAsyncioTestCase):
    async def test_default_strips_markdown(self):
        formatter = VoiceFormatter()
        result = await formatter("**Hello** world", "*")
        self.assertNotIn("**", result)
        self.assertIn("Hello", result)

    async def test_disabled_option_skipped(self):
        formatter = VoiceFormatter(strip_markdown=False)
        result = await formatter("**Hello**", "*")
        self.assertIn("**", result)

    async def test_custom_replacements_applied(self):
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
        result = await formatter("Dr. Smith", "*")
        self.assertIn("Doctor", result)

    async def test_all_disabled_no_change(self):
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
        result = await formatter(text, "*")
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()

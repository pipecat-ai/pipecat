#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.transforms._alnum_utils import normalize
from pipecat.utils.text.transforms.acronyms import normalize_acronyms
from pipecat.utils.text.transforms.currency import expand_currency
from pipecat.utils.text.transforms.dates import normalize_dates
from pipecat.utils.text.transforms.email import email_to_speech
from pipecat.utils.text.transforms.numbers import expand_numbers
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

    async def test_no_match_inside_longer_digit_sequence(self):
        # A 16-digit credit card number must not be treated as a phone number.
        text = "Card 4111111111111111 is invalid"
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

    async def test_expand_numbers_enabled(self):
        formatter = VoiceFormatter(expand_numbers=True, number_digit_cutoff=2025)
        self.assertIn("forty-two", await formatter("Room 42", "*"))

    async def test_all_caps_email_not_mangled_by_acronym_expander(self):
        # email_to_speech must run before normalize_acronyms so all-caps addresses
        # like USER@EXAMPLE.COM are detected before letters are spaced out.
        formatter = VoiceFormatter()
        result = await formatter("Contact USER@EXAMPLE.COM today", "*")
        self.assertIn("at", result)
        self.assertNotIn("@", result)

    async def test_uppercase_unit_not_broken_by_acronym_expander(self):
        # expand_units must run before normalize_acronyms so "100 MB" is expanded
        # to "100 megabytes" before "MB" gets letter-spaced to "M B".
        formatter = VoiceFormatter()
        result = await formatter("100 MB file", "*")
        self.assertIn("megabytes", result)
        self.assertNotIn("M B", result)


class TestExpandCurrency(unittest.IsolatedAsyncioTestCase):
    async def test_dollars_and_cents(self):
        result = await expand_currency("Your balance is $42.50", "*")
        self.assertIn("forty-two dollars", result)
        self.assertIn("fifty cents", result)
        self.assertNotIn("$", result)

    async def test_singular_unit(self):
        # "$1" -> "one dollar" (singular), not "one dollars".
        self.assertEqual(await expand_currency("$1", "*"), "one dollar")

    async def test_singular_subunit(self):
        # "£1.01" -> singular pound + singular penny.
        result = await expand_currency("£1.01", "*")
        self.assertIn("one pound", result)
        self.assertIn("one penny", result)

    async def test_zero_cents_omitted(self):
        # A ".00" fraction must not produce "and zero cents".
        result = await expand_currency("$5.00", "*")
        self.assertIn("five dollars", result)
        self.assertNotIn("cent", result)

    async def test_subunit_less_currency_drops_fraction(self):
        # Yen has no subunit, so the fractional part is dropped.
        result = await expand_currency("¥500.50", "*")
        self.assertIn("five hundred yen", result)
        self.assertNotIn("cent", result)

    async def test_thousands_separator(self):
        result = await expand_currency("$1,000", "*")
        self.assertIn("one thousand dollars", result)
        self.assertNotIn("1,000", result)

    async def test_multi_digit_without_separator(self):
        # Regression: 4+ digit amounts without commas must capture the full integer.
        # The integer group needs a \b so the regex backtracks onto the \d+ branch
        # instead of matching only the first 3 digits and leaving a stray digit.
        self.assertEqual(await expand_currency("$1000", "*"), "one thousand dollars")
        result = await expand_currency("$1000.50", "*")
        self.assertIn("one thousand dollars", result)
        self.assertIn("fifty cents", result)


class TestNormalizeDates(unittest.IsolatedAsyncioTestCase):
    async def test_iso_date(self):
        result = await normalize_dates("Meeting on 2023-05-10", "*")
        self.assertIn("May 10th", result)
        self.assertIn("twenty-three", result)  # year expanded to words
        self.assertNotIn("2023-05-10", result)

    async def test_us_date_slash(self):
        result = await normalize_dates("Meeting on 05/10/2023", "*")
        self.assertIn("May 10th", result)
        self.assertNotIn("05/10/2023", result)

    async def test_us_date_dash(self):
        result = await normalize_dates("Meeting on 05-10-2023", "*")
        self.assertIn("May 10th", result)
        self.assertNotIn("05-10-2023", result)

    async def test_invalid_date_unchanged(self):
        # An out-of-range month/day makes datetime() raise, so the text passes
        # through unchanged — checked for both the ISO and US replacement paths.
        iso = "Order 2023-13-45 shipped"
        self.assertEqual(await normalize_dates(iso, "*"), iso)
        us = "Due 13/45/2023 now"
        self.assertEqual(await normalize_dates(us, "*"), us)

    async def test_ordinal_teens_use_th(self):
        # 11/12/13 take the "th" suffix, not st/nd/rd.
        self.assertIn("11th", await normalize_dates("2023-05-11", "*"))
        self.assertIn("12th", await normalize_dates("2023-05-12", "*"))
        self.assertIn("13th", await normalize_dates("2023-05-13", "*"))

    async def test_ordinal_suffixes(self):
        self.assertIn("1st", await normalize_dates("2023-05-01", "*"))
        self.assertIn("21st", await normalize_dates("2023-05-21", "*"))
        self.assertIn("2nd", await normalize_dates("2023-05-02", "*"))
        self.assertIn("3rd", await normalize_dates("2023-05-03", "*"))


class TestExpandNumbers(unittest.IsolatedAsyncioTestCase):
    async def test_below_cutoff_expands_to_words(self):
        result = await expand_numbers(digit_cutoff=2025)("Room 42", "*")
        self.assertIn("forty-two", result)

    async def test_above_cutoff_read_digit_by_digit(self):
        result = await expand_numbers(digit_cutoff=2025)("opens in 2026", "*")
        self.assertIn("2 0 2 6", result)

    async def test_cutoff_is_inclusive(self):
        # The cutoff uses `>` not `>=`, so a number equal to it is still expanded.
        result = await expand_numbers(digit_cutoff=2025)("2025", "*")
        self.assertIn("thousand", result)
        self.assertNotIn("2 0 2 5", result)

    async def test_decimal_expands(self):
        result = await expand_numbers(digit_cutoff=None)("3.5", "*")
        self.assertIn("three point five", result)

    async def test_none_cutoff_expands_all(self):
        result = await expand_numbers(digit_cutoff=None)("9999", "*")
        self.assertNotIn("9999", result)
        self.assertIn("thousand", result)

    async def test_above_cutoff_decimal_preserves_fraction(self):
        # Integer part above cutoff → digit-by-digit; fraction must not be dropped.
        result = await expand_numbers(digit_cutoff=2025)("3000.5 units", "*")
        self.assertIn("3 0 0 0", result)
        self.assertIn("point", result)
        self.assertIn("5", result)
        self.assertNotIn("3000", result)

    async def test_above_cutoff_multi_digit_fraction(self):
        result = await expand_numbers(digit_cutoff=2025)("2500.75", "*")
        self.assertIn("2 5 0 0", result)
        self.assertIn("point", result)
        self.assertIn("7 5", result)


if __name__ == "__main__":
    unittest.main()

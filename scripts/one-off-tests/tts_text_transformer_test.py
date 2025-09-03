#!/usr/bin/env python3

# Standalone test for TTSTextTransformer without dependencies
import sys
import os
import re
import time
import statistics
from typing import List

# Copy the TTSTextTransformer class directly to avoid import issues
class TTSTextTransformer:
    """
    TTS text transformer that improves pronunciation by transforming text patterns.

    Transforms various number formats for better TTS pronunciation:
    - Times: "8:00AM" becomes "8 AM", "12:30 PM" becomes "12 30 PM"
    - Phone numbers: "4256152509" becomes "four two five, six one five, two five zero nine"
    - Addresses: "1713 Oak St" becomes "seventeen thirteen Oak St"
    - Money: "$9.99" becomes "nine ninety nine"
    - Quantities: "150 lbs" becomes "one hundred fifty lbs"

    Uses regex-based pattern matching for comprehensive number transformation.
    """

    # Regex patterns as constants for better readability
    TIME_COLON_PATTERN = r'\b[0-9]{1,2}:[0-9]{2}(?:\s*[AaPp][Mm])?\b'
    TIME_SIMPLE_PATTERN = r'\b[0-9]{1,2}[AaPp][Mm]\b'
    TIME_NO_BOUNDARY_COLON_PATTERN = r'([a-zA-Z])([0-9]{1,2}:[0-9]{2}(?:\s*[AaPp][Mm])?)'
    TIME_NO_BOUNDARY_SIMPLE_PATTERN = r'([a-zA-Z])([0-9]{1,2}[AaPp][Mm])'
    
    MONEY_WITH_DOLLARS_WORD_PATTERN = r'\$([0-9]+(?:\.[0-9]{2})?)\s+dollars?'
    MONEY_DOLLAR_SIGN_PATTERN = r'\$([0-9]+(?:\.[0-9]{2})?)'
    
    PHONE_10_DIGIT_PATTERN = r'\b[0-9]{10}\b'
    PHONE_11_DIGIT_PATTERN = r'\b[0-9]{11}\b'
    PHONE_FORMATTED_PARENS_PATTERN = r'\([0-9]{3}\)\s?[0-9]{3}[-.\s][0-9]{4}'
    PHONE_FORMATTED_DASH_PATTERN = r'\b[0-9]{3}[-.\s][0-9]{3}[-.\s][0-9]{4}\b'
    PHONE_PLUS_PATTERN = r'\+[0-9]{11}'
    PHONE_1_DASH_PATTERN = r'\b1-[0-9]{3}-[0-9]{3}-[0-9]{4}\b'
    
    STREET_ADDRESS_PATTERN = r'\b([0-9]{1,5})(\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Lane|Ln|Boulevard|Blvd|Way|Place|Pl|Court|Ct|Crescent|Cres))\b'
    ZIP_STATE_PATTERN = r'\b([0-9]{5}(?:-[0-9]{4})?)(\s+[A-Z]{2})\b'
    ZIP_AFTER_STATE_PATTERN = r'\b([A-Za-z]+,\s+)([0-9]{5})\b'
    STATE_ZIP_PATTERN = r'\b([A-Z]{2}\s+)([0-9]{5})\b'
    
    # Date patterns - long and short month forms
    DATE_LONG_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+([0-9]{1,2})\b'
    DATE_SHORT_PATTERN = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(\.?)\s+([0-9]{1,2})\b'
    DATE_THE_LONG_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+the\s+([0-9]{1,2})\b'
    DATE_THE_SHORT_PATTERN = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(\.?)\s+the\s+([0-9]{1,2})\b'
    
    # Attached date patterns (no space between month and day)
    DATE_LONG_ATTACHED_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)([0-9]{1,2})\b'
    DATE_SHORT_ATTACHED_PATTERN = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(\.?)([0-9]{1,2})\b'
    
    # Standalone ordinal patterns  
    ORDINAL_TEXT_PATTERN = r'\b([0-9]{1,2})(st|nd|rd|th)\b'
    
    UNITS_PATTERN = r'\b([0-9]+)(\s*(?:square\s+feet|sqft|sq\.?\s*ft|degrees?|°|lbs?|pounds?|kg|miles?|mi|feet|ft|inches?|in))\b'
    
    NUMBER_ADJACENT_LETTER_PATTERN = r'([a-zA-Z])([0-9]+)'
    STANDALONE_NUMBER_PATTERN = r'\b[0-9]+\b'

    def __init__(self):
        pass

    def time_transformer(self, text: str) -> str:
        """
        Transform time formats for better TTS pronunciation.

        Examples:
        - '8:00' -> '8'
        - '9:30' -> '9 30'
        - '8:00AM' -> '8AM'
        - '9:30 PM' -> '9 30 PM'
        - '9am' -> '9 am'
        - '12PM' -> '12 PM'

        Args:
            text: Input text containing time patterns

        Returns:
            Transformed text with improved time pronunciation
        """
        # First handle time patterns with colons: hour:minute with optional AM/PM
        # Group 1: hour (1-2 digits)
        # Group 2: minute (2 digits)
        # Group 3: optional AM/PM with optional space
        colon_pattern = r"\b(\d{1,2}):(\d{2})(\s?[AaPp][Mm])?\b"

        def replace_colon_time(match):
            hour = match.group(1)
            minute = match.group(2)
            am_pm = match.group(3) or ""  # AM/PM suffix if present

            # For :00 minutes, omit them (e.g., "8:00" -> "8")
            if minute == "00":
                return f"{hour}{am_pm}"
            # For other minutes, separate with space (e.g., "9:30" -> "9 30")
            return f"{hour} {minute}{am_pm}"

        # Apply colon time transformations first
        text = re.sub(colon_pattern, replace_colon_time, text)

        # Then handle simple time formats: digit + AM/PM (e.g., "9am" -> "9 am")
        # Group 1: hour (1-2 digits)
        # Group 2: AM/PM without leading space
        simple_pattern = r"\b(\d{1,2})([AaPp][Mm])\b"

        def replace_simple_time(match):
            hour = match.group(1)
            am_pm = match.group(2)
            # Add space between hour and AM/PM
            return f"{hour} {am_pm}"

        # Apply simple time transformations
        text = re.sub(simple_pattern, replace_simple_time, text)

        return text

    def transform_address_number(self, number_str: str) -> str:
        """
        Transform address-style numbers for natural pronunciation.
        
        Examples:
        - 1713 -> "seventeen thirteen"
        - 94103 -> "nine four one zero three"  
        - 2500 -> "twenty five hundred"
        - 59 -> "fifty nine"
        - 1609 -> "sixteen oh nine"
        - 818 -> "eight eighteen"
        """
        return self.number_to_address(number_str)

    def number_to_grouped_digits(self, number_str: str) -> str:
        """
        Convert large numbers to grouped digits like phone numbers.
        
        Examples:
        - 120500 -> "one two zero, five zero zero" (6 digits: 3+3)
        - 1000000 -> "one, zero zero zero, zero zero zero" (7 digits: 1+3+3)
        """
        digits = str(number_str)
        length = len(digits)
        
        if length <= 3:
            return self.number_to_digits(number_str)
        elif length == 4:
            return self.number_to_digits(number_str)
        elif length == 5:
            return self.number_to_digits(number_str) 
        elif length == 6:
            # Group as 3+3: "120500" -> "one two zero, five zero zero"
            first = digits[:3]
            second = digits[3:]
            return f"{self.number_to_digits(first)}, {self.number_to_digits(second)}"
        elif length == 7:
            # Group as 1+3+3: "1000000" -> "one, zero zero zero, zero zero zero"  
            first = digits[0]
            second = digits[1:4]
            third = digits[4:]
            return f"{self.number_to_digits(first)}, {self.number_to_digits(second)}, {self.number_to_digits(third)}"
        else:
            # For longer numbers, fall back to simple digits
            return self.number_to_digits(number_str)

    def transform_quantity_number(self, number_str: str) -> str:
        """
        Transform quantity numbers with smart logic based on context.
        
        Examples:
        - 2019 -> "two thousand nineteen" (4-digit years use standard)  
        - 94103 -> "nine four one zero three" (5 digits use digits)
        - 120500 -> "one two zero, five zero zero" (6+ digits use grouping)
        """
        num = int(number_str)
        length = len(str(num))
        
        # 4-digit numbers that look like years should use standard format
        if length == 4 and 1800 <= num <= 2100:
            return self.number_to_standard(number_str)
        # 5+ digit numbers should use grouped digits or single digits
        elif length >= 5:
            if length >= 6:
                return self.number_to_grouped_digits(number_str)
            else:
                return self.number_to_digits(number_str)
        # Other 4-digit numbers use standard format  
        elif length == 4:
            return self.number_to_standard(number_str)
        # Smaller numbers use standard format
        else:
            return self.number_to_standard(number_str)
    
    def transform_quantity_with_units(self, number_str: str) -> str:
        """
        Transform quantity numbers with units using full standard conversion.
        This is used when we detect units, so we want the full pronunciation.
        
        Examples:
        - "25000 lbs" -> "twenty five thousand lbs"
        - "2510 sqft" -> "two thousand five hundred ten sqft"  
        """
        return self.number_to_standard(number_str)

    def transform_phone_number(self, number_str: str) -> str:
        """
        Transform phone numbers into grouped digits.
        
        Examples:
        - 4256152509 -> "four two five, six one five, two five zero nine"
        - 8005551234 -> "eight zero zero, five five five, one two three four"
        """
        # Remove any formatting
        digits = re.sub(r'[^0-9]', '', number_str)
        
        if len(digits) == 10:
            # Standard US format: (425) 615-2509 -> "four two five, six one five, two five zero nine"
            area = digits[:3]
            exchange = digits[3:6] 
            number = digits[6:]
            
            area_words = ' '.join(self.digit_to_word(int(d)) for d in area)
            exchange_words = ' '.join(self.digit_to_word(int(d)) for d in exchange)
            number_words = ' '.join(self.digit_to_word(int(d)) for d in number)
            
            return f"{area_words}, {exchange_words}, {number_words}"
        elif len(digits) == 11 and digits[0] == '1':
            # US format with country code: 14256152509 -> "one, four two five, six one five, two five zero nine"
            country = digits[0]
            area = digits[1:4]
            exchange = digits[4:7]
            number = digits[7:]
            
            country_word = self.digit_to_word(int(country))
            area_words = ' '.join(self.digit_to_word(int(d)) for d in area)
            exchange_words = ' '.join(self.digit_to_word(int(d)) for d in exchange)
            number_words = ' '.join(self.digit_to_word(int(d)) for d in number)
            
            return f"{country_word}, {area_words}, {exchange_words}, {number_words}"
        else:
            # Fallback: read each digit
            return ' '.join(self.digit_to_word(int(d)) for d in digits)

    # Core number conversion methods
    def number_to_digits(self, number_str: str) -> str:
        """Convert number to digit-by-digit pronunciation: 1234 -> 'one two three four'"""
        return ' '.join(self.digit_to_word(int(d)) for d in str(number_str))
    
    def number_to_standard(self, number_str: str) -> str:
        """Convert number to standard format: 1234 -> 'one thousand two hundred thirty four'"""
        num = int(number_str)
        
        if num == 0:
            return "zero"
        
        # Handle negatives
        if num < 0:
            return f"negative {self.number_to_standard(str(-num))}"
        
        # Standard number-to-words conversion
        if num < 20:
            return self.ones[num]
        elif num < 100:
            tens, ones = divmod(num, 10)
            if ones == 0:
                return self.tens[tens]
            return f"{self.tens[tens]} {self.ones[ones]}"
        elif num < 1000:
            hundreds, remainder = divmod(num, 100)
            if remainder == 0:
                return f"{self.ones[hundreds]} hundred"
            return f"{self.ones[hundreds]} hundred {self.number_to_standard(str(remainder))}"
        elif num < 1000000:
            thousands, remainder = divmod(num, 1000)
            if remainder == 0:
                return f"{self.number_to_standard(str(thousands))} thousand"
            return f"{self.number_to_standard(str(thousands))} thousand {self.number_to_standard(str(remainder))}"
        elif num < 1000000000:
            millions, remainder = divmod(num, 1000000)
            if remainder == 0:
                return f"{self.number_to_standard(str(millions))} million"
            return f"{self.number_to_standard(str(millions))} million {self.number_to_standard(str(remainder))}"
        else:
            # For very large numbers, fall back to digit-by-digit
            return self.number_to_digits(number_str)
    
    def number_to_address(self, number_str: str) -> str:
        """Convert number to address-style pronunciation: 1713 -> 'seventeen thirteen'"""
        num = int(number_str)
        
        # Single digit
        if num < 10:
            return self.digit_to_word(num)
        
        # Two digits
        if num < 100:
            return self.two_digit_to_words(num)
        
        # Three digits - special handling for addresses
        if num < 1000:
            if num % 100 == 0:  # e.g., 800 -> "eight hundred"
                return f"{self.digit_to_word(num // 100)} hundred"
            elif str(num)[1] == '0':  # e.g., 805 -> "eight oh five"
                return f"{self.digit_to_word(num // 100)} oh {self.digit_to_word(num % 10)}"
            else:  # e.g., 818 -> "eight eighteen"
                return f"{self.digit_to_word(num // 100)} {self.two_digit_to_words(num % 100)}"
        
        # Four digits - address style (e.g., 1713 -> "seventeen thirteen")
        if num < 10000:
            first_two = num // 100
            last_two = num % 100
            if last_two == 0:
                return f"{self.two_digit_to_words(first_two)} hundred"
            elif last_two < 10:
                return f"{self.two_digit_to_words(first_two)} oh {self.digit_to_word(last_two)}"
            else:
                return f"{self.two_digit_to_words(first_two)} {self.two_digit_to_words(last_two)}"
        
        # Five digits (ZIP codes) - always spell out each digit
        if num < 100000:
            return self.number_to_digits(number_str)
        
        # Fallback for larger numbers - use standard
        return self.number_to_standard(number_str)

    def number_to_words(self, number_str: str) -> str:
        """
        Base method - defaults to standard conversion for most use cases.
        This is the foundation method that others build upon.
        
        Examples:
        - 25 -> "twenty five"
        - 1000 -> "one thousand"  
        - 2510 -> "two thousand five hundred ten"
        """
        return self.number_to_standard(number_str)
    
    def transform_money_number(self, number_str: str, preserve_dollars_word: bool = False) -> str:
        """
        Transform money amounts for natural pronunciation.
        
        Examples:
        - 9.99 -> "nine ninety nine" 
        - 999 -> "nine hundred ninety nine"
        - 2.00 -> "two"
        - 15.50 -> "fifteen fifty"
        - 0.05 -> "five cents"
        - 0.95 -> "ninety five cents"
        - 1.95 -> "one ninety five"
        """
        # Handle decimal format
        if '.' in number_str:
            dollars, cents = number_str.split('.')
            dollars_num = int(dollars)
            cents_num = int(cents)
            
            # Handle amounts less than $1 (0.xx) - just say cents
            if dollars_num == 0:
                if cents_num == 0:
                    return "zero cents"
                elif cents_num < 10:
                    return f"{self.digit_to_word(cents_num)} cents"
                else:
                    return f"{self.two_digit_to_words(cents_num)} cents"
            
            # Handle amounts $1 or more
            if cents_num == 0:
                # $2.00 -> "two dollars"
                result = self.number_to_words(dollars)
                dollar_word = "dollar" if dollars_num == 1 else "dollars"
                return f"{result} {dollar_word}" if preserve_dollars_word else result
            else:
                # All decimal amounts use "dollars and cents" format
                # $1.05 -> "one dollar and five cents"
                # $299.99 -> "two hundred ninety nine dollars and ninety nine cents"
                dollars_text = self.number_to_words(dollars)
                dollar_word = "dollar" if dollars_num == 1 else "dollars"
                
                if cents_num < 10:
                    cents_text = f"{self.digit_to_word(cents_num)} cents"
                else:
                    cents_text = f"{self.two_digit_to_words(cents_num)} cents"
                
                if preserve_dollars_word:
                    return f"{dollars_text} {dollar_word} and {cents_text}"
                else:
                    return f"{dollars_text} and {cents_text}"
        else:
            # Whole dollar amount
            result = self.number_to_words(number_str)
            return f"{result} dollars" if preserve_dollars_word else result

    def digit_to_word(self, digit: int) -> str:
        """Convert single digit to word."""
        return self.digits[digit]

    def two_digit_to_words(self, num: int) -> str:
        """Convert two-digit number to words."""
        if num < 20:
            return self.ones[num]
        tens, ones = divmod(num, 10)
        if ones == 0:
            return self.tens[tens]
        return f"{self.tens[tens]} {self.ones[ones]}"

    def number_to_ordinal(self, num: int) -> str:
        """Convert number to ordinal form (e.g., 1 -> 'first', 3 -> 'third')."""
        ordinals = {
            1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
            16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
            21: "twenty first", 22: "twenty second", 23: "twenty third", 24: "twenty fourth", 25: "twenty fifth",
            26: "twenty sixth", 27: "twenty seventh", 28: "twenty eighth", 29: "twenty ninth", 30: "thirtieth",
            31: "thirty first"
        }
        return ordinals.get(num, f"{self.transform_quantity_number(str(num))}th")

    @property
    def digits(self) -> List[str]:
        """Digit to word mapping."""
        return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    @property
    def ones(self) -> List[str]:
        """Numbers 0-19 to words."""
        return [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
            "seventeen", "eighteen", "nineteen"
        ]

    @property 
    def tens(self) -> List[str]:
        """Tens place to words."""
        return ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def transform(self, text: str) -> str:
        """
        Transform all numbers and patterns in text for better TTS pronunciation.
        
        Analyzes each number and applies the appropriate transformation:
        - Phone numbers: grouped digits
        - Money: natural money pronunciation  
        - Time: simplified time formats
        - Addresses: address-style pronunciation
        - Quantities: standard number-to-words
        
        Args:
            text: Input text to transform
            
        Returns:
            Transformed text with improved pronunciation patterns
        """
        # Apply transformations in order of priority
        result = text
        
        # 1. Time patterns (highest priority - colon patterns or AM/PM)
        def replace_time(match):
            return self.time_transformer(match.group(0))
        result = re.sub(self.TIME_COLON_PATTERN, replace_time, result)
        result = re.sub(self.TIME_SIMPLE_PATTERN, replace_time, result)
        
        # Handle times without word boundary (like "from8:00 AM" and "to12:00 PM")
        def replace_time_no_boundary(match):
            prefix = match.group(1)
            time_part = match.group(2)
            return prefix + self.time_transformer(time_part)
        result = re.sub(self.TIME_NO_BOUNDARY_COLON_PATTERN, replace_time_no_boundary, result)
        
        # Handle simple times without boundary (like remaining "8 AM" -> "eight AM")
        def replace_simple_time_no_boundary(match):
            prefix = match.group(1)
            time_part = match.group(2)
            return prefix + self.time_transformer(time_part)
        result = re.sub(self.TIME_NO_BOUNDARY_SIMPLE_PATTERN, replace_simple_time_no_boundary, result)
        
        # 2. Money patterns (dollar signs or "dollars" word)
        # Handle "$299 dollars" first (more specific pattern) - preserve "dollars" word
        def replace_money_dollars_word(match):
            return self.transform_money_number(match.group(1), preserve_dollars_word=True)
        result = re.sub(self.MONEY_WITH_DOLLARS_WORD_PATTERN, replace_money_dollars_word, result, flags=re.IGNORECASE)
        
        # Then handle regular money with $ symbol - always include "dollars"
        def replace_money_dollar(match):
            return self.transform_money_number(match.group(1), preserve_dollars_word=True)
        result = re.sub(self.MONEY_DOLLAR_SIGN_PATTERN, replace_money_dollar, result)
        
        # 3. Phone numbers (10+ digit numbers unlikely to be other things)
        # Most specific patterns first
        def replace_phone_formatted(match):
            digits = re.sub(r'[^0-9]', '', match.group(0))
            return self.transform_phone_number(digits)
        
        # Plus format: +14256152509
        result = re.sub(self.PHONE_PLUS_PATTERN, replace_phone_formatted, result)
        
        # 1-dash format: 1-555-223-4567
        result = re.sub(self.PHONE_1_DASH_PATTERN, replace_phone_formatted, result)
        
        # Formatted phone numbers: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx
        result = re.sub(self.PHONE_FORMATTED_PARENS_PATTERN, replace_phone_formatted, result)
        result = re.sub(self.PHONE_FORMATTED_DASH_PATTERN, replace_phone_formatted, result)
        
        # Plain digit patterns (after formatted patterns to avoid conflicts)
        def replace_phone_digits(match):
            return self.transform_phone_number(match.group(0))
        
        # 11-digit phone numbers  
        result = re.sub(self.PHONE_11_DIGIT_PATTERN, replace_phone_digits, result)
        
        # 10-digit phone numbers
        result = re.sub(self.PHONE_10_DIGIT_PATTERN, replace_phone_digits, result)
        
        # 4. Address patterns
        # Street addresses: "1713 Springhurst Drive", "2500 Oak Street"
        def replace_street_address(match):
            return self.transform_address_number(match.group(1)) + match.group(2)
        result = re.sub(self.STREET_ADDRESS_PATTERN, replace_street_address, result, flags=re.IGNORECASE)
        
        # ZIP codes: "94103 CA" and "Arkansas, 72701" format
        def replace_zip(match):
            return self.transform_address_number(match.group(1)) + match.group(2)
        result = re.sub(self.ZIP_STATE_PATTERN, replace_zip, result)
        
        # Handle ZIP codes after state names like "Arkansas, 72701"
        def replace_state_zip(match):
            return match.group(1) + self.transform_address_number(match.group(2))
        result = re.sub(self.ZIP_AFTER_STATE_PATTERN, replace_state_zip, result)
        
        # Handle ZIP codes after state codes like "OH 94103"
        def replace_state_code_zip(match):
            return match.group(1) + self.transform_address_number(match.group(2))
        result = re.sub(self.STATE_ZIP_PATTERN, replace_state_code_zip, result)
        
        # 5. Date patterns (month + number should be ordinal)
        def replace_date_ordinal(match):
            month = match.group(1)
            day_num = int(match.group(2))
            ordinal = self.number_to_ordinal(day_num)
            return f"{month} {ordinal}"
        
        def replace_short_date_ordinal(match):
            month = match.group(1)
            period = match.group(2)
            day_num = int(match.group(3))
            ordinal = self.number_to_ordinal(day_num)
            return f"{month}{period} {ordinal}"
        
        def replace_date_with_the(match):
            month = match.group(1)
            day_num = int(match.group(2))
            ordinal = self.number_to_ordinal(day_num)
            return f"{month} the {ordinal}"
        
        def replace_short_date_with_the(match):
            month = match.group(1)
            period = match.group(2)
            day_num = int(match.group(3))
            ordinal = self.number_to_ordinal(day_num)
            return f"{month}{period} the {ordinal}"
        
        # Handle dates with "the" first (more specific)
        result = re.sub(self.DATE_THE_LONG_PATTERN, replace_date_with_the, result, flags=re.IGNORECASE)
        result = re.sub(self.DATE_THE_SHORT_PATTERN, replace_short_date_with_the, result, flags=re.IGNORECASE)
        
        # Handle regular date patterns (with spaces)
        result = re.sub(self.DATE_LONG_PATTERN, replace_date_ordinal, result)
        result = re.sub(self.DATE_SHORT_PATTERN, replace_short_date_ordinal, result, flags=re.IGNORECASE)
        
        # Handle attached date patterns (no spaces) - add space in output
        def replace_attached_date(match):
            month = match.group(1)
            day_num = int(match.group(2))
            ordinal = self.number_to_ordinal(day_num)
            return f"{month} {ordinal}"
        
        def replace_short_attached_date(match):
            month = match.group(1)
            period = match.group(2)
            day_num = int(match.group(3))
            ordinal = self.number_to_ordinal(day_num)
            return f"{month}{period} {ordinal}"
        
        result = re.sub(self.DATE_LONG_ATTACHED_PATTERN, replace_attached_date, result)
        result = re.sub(self.DATE_SHORT_ATTACHED_PATTERN, replace_short_attached_date, result, flags=re.IGNORECASE)
        
        # Handle standalone ordinals like "3rd", "21st"
        def replace_ordinal_text(match):
            day_num = int(match.group(1))
            return self.number_to_ordinal(day_num)
        result = re.sub(self.ORDINAL_TEXT_PATTERN, replace_ordinal_text, result)
        
        # 6. Numbers with units (use full quantity transformation)
        def replace_quantity_with_unit(match):
            number_text = self.transform_quantity_with_units(match.group(1))
            unit_text = match.group(2)
            # Convert "square feet" to "sqft" for consistency
            if "square feet" in unit_text.lower():
                unit_text = unit_text.replace("square feet", "sqft").replace("Square feet", "sqft").replace("Square Feet", "sqft")
            return number_text + unit_text
        result = re.sub(self.UNITS_PATTERN, replace_quantity_with_unit, result, flags=re.IGNORECASE)
        
        # 7. Numbers adjacent to letters (like "from8" -> "from eight")
        def replace_number_adjacent_letter(match):
            prefix = match.group(1)
            number = match.group(2)
            return prefix + " " + self.transform_quantity_number(number)
        result = re.sub(self.NUMBER_ADJACENT_LETTER_PATTERN, replace_number_adjacent_letter, result)
        
        # 8. All remaining standalone numbers (default to quantity transformation)
        def replace_number(match):
            return self.transform_quantity_number(match.group(0))
        result = re.sub(self.STANDALONE_NUMBER_PATTERN, replace_number, result)
        
        return result


def run_test_group(name, test_cases, transformer, verbose=False):
    """Run a group of test cases with performance measurement"""
    print(f"\n{name}:")
    print("-" * len(name))
    
    passed = 0
    total = len(test_cases)
    execution_times = []
    
    for input_text, expected in test_cases:
        # Measure execution time
        start_time = time.perf_counter()
        result = transformer.transform(input_text)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time_ms)
        
        status = "✓" if result == expected else "✗"
        
        if verbose or result != expected:
            print(f"{status} '{input_text}'")
            print(f"   -> '{result}'")
            if result != expected:
                print(f"   Expected: '{expected}'")
        
        if result == expected:
            passed += 1
    
    # Calculate performance metrics
    p90_time = statistics.quantiles(execution_times, n=10)[8] if execution_times else 0  # P90
    avg_time = statistics.mean(execution_times) if execution_times else 0
    max_time = max(execution_times) if execution_times else 0
    
    if not verbose:
        print(f"Passed: {passed}/{total}")
    
    print(f"Performance: Avg={avg_time:.3f}ms, P90={p90_time:.3f}ms, Max={max_time:.3f}ms")
    
    return passed == total, p90_time

def test_time_patterns():
    """Test time transformation patterns"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Basic time formats without AM/PM
        ("8:00", "eight"),
        ("9:30", "nine thirty"),
        ("12:45", "twelve forty five"),
        ("1:15", "one fifteen"),
        
        # Time formats with AM/PM (no space)
        ("8:00AM", "eight AM"),
        ("9:30PM", "nine thirty PM"),
        ("12:00AM", "twelve AM"),
        ("11:45PM", "eleven forty five PM"),
        
        # Time formats with AM/PM (with space)
        ("8:00 AM", "eight AM"),
        ("9:30 PM", "nine thirty PM"),
        ("12:00 AM", "twelve AM"),
        ("11:45 PM", "eleven forty five PM"),
        
        # Mixed case AM/PM
        ("8:00am", "eight am"),
        ("9:30pm", "nine thirty pm"),
        ("8:00 Am", "eight Am"),
        ("9:30 Pm", "nine thirty Pm"),
        
        # Simple time formats (digit + AM/PM without colon)
        ("9am", "nine am"),
        ("12pm", "twelve pm"),
        ("8AM", "eight AM"),
        ("11PM", "eleven PM"),
        ("9Am", "nine Am"),
        ("12Pm", "twelve Pm"),
        
        # Text with multiple times (mixed formats)
        ("Meet at 9:00 AM and again at 2:30 PM", "Meet at nine AM and again at two thirty PM"),
        ("From 8:00 to 9:30", "From eight to nine thirty"),
        ("Available 9am to 5pm daily", "Available nine am to five pm daily"),
        ("Call between 8AM and 12:30 PM", "Call between eight AM and twelve thirty PM"),
        
        # Times in sentences
        ("I'll be there at 8:00 AM sharp", "I'll be there at eight AM sharp"),
        ("The appointment is scheduled for 2:30 PM today", "The appointment is scheduled for two thirty PM today"),

        # Some bad formatting cases
        ("From8:00 AM to12:00PM", "From eight AM to twelve PM"),
    ]
    
    return run_test_group("Time Patterns", test_cases, transformer)

def test_phone_numbers():
    """Test phone number transformations"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        ("4256152509", "four two five, six one five, two five zero nine"),
        ("8005551234", "eight zero zero, five five five, one two three four"),
        ("14256152509", "one, four two five, six one five, two five zero nine"),
        ("5551234567", "five five five, one two three, four five six seven"),
        ("555-223-4567", "five five five, two two three, four five six seven"),
        ("1-555-223-4567", "one, five five five, two two three, four five six seven"),
        ("+14256152509", "one, four two five, six one five, two five zero nine"),
        ("Call me at 4256152509", "Call me at four two five, six one five, two five zero nine"),
        ("My number is (425) 615-2509", "My number is four two five, six one five, two five zero nine"),
    ]
    
    return run_test_group("Phone Numbers", test_cases, transformer)

def test_money_patterns():
    """Test money transformations"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Basic money patterns - all decimals use "dollars and cents" format
        ("$9.99", "nine dollars and ninety nine cents"),
        ("$999", "nine hundred ninety nine dollars"),
        ("$2.00", "two dollars"),
        ("$15.50", "fifteen dollars and fifty cents"),
        ("$1.95", "one dollar and ninety five cents"),
        ("$1.05", "one dollar and five cents"),
        ("$25", "twenty five dollars"),
        ("$1000", "one thousand dollars"),
        ("$99.01", "ninety nine dollars and one cents"),
        ("$1000000", "one million dollars"),
        
        # Amounts under $1 - just say cents
        ("$0.99", "ninety nine cents"),
        ("$0.05", "five cents"),
        ("$0.01", "one cents"),
        ("$0.00", "zero cents"),
        
        # Large amounts use "dollars and cents" format
        ("$299.99", "two hundred ninety nine dollars and ninety nine cents"),
        ("$150.50", "one hundred fifty dollars and fifty cents"),
        ("$100.01", "one hundred dollars and one cents"),
        
        # Additional cents patterns
        ("$0.10", "ten cents"),
        ("$0.25", "twenty five cents"),
        
        # Dollar word preservation
        ("$299 dollars", "two hundred ninety nine dollars"),
        ("$50 dollar", "fifty dollars"),
        ("$1000 dollars", "one thousand dollars"),
        ("$25 DOLLARS", "twenty five dollars"),
        ("$2.00 dollars", "two dollars"),
        
        # Money in context
        ("It costs $9.99", "It costs nine dollars and ninety nine cents"),
        ("The price is $299 dollars", "The price is two hundred ninety nine dollars"),
        ("Only $2.00 today", "Only two dollars today"),
        ("Just $0.05", "Just five cents"),
        ("Costs $0.95", "Costs ninety five cents"),
        ("Price $1.95", "Price one dollar and ninety five cents"),
        ("It costs $0.50", "It costs fifty cents"),
        ("Only $0.05 each", "Only five cents each")
    ]
    
    return run_test_group("Money Patterns", test_cases, transformer)

def test_standalone_numbers():
    """Test standalone number transformations"""
    transformer = TTSTextTransformer()

    test_cases = [
        ("1713", "one thousand seven hundred thirteen"),
        ("94103", "nine four one zero three"),
        ("2019", "two thousand nineteen"),
        ("59", "fifty nine"),
        ("1609", "one thousand six hundred nine"),
        ("818", "eight hundred eighteen"),
        ("805", "eight hundred five"),
        ("800", "eight hundred"),
        ("5", "five"),
        ("42", "forty two"),
        ("1234", "one thousand two hundred thirty four"),
        ("1200", "one thousand two hundred"),
        ("1205", "one thousand two hundred five"),
        ("120500", "one two zero, five zero zero"),
        ("1000000", "one, zero zero zero, zero zero zero"),
    ]
    
    return run_test_group("Standalone Numbers", test_cases, transformer)

def test_address_patterns():
    """Test address transformations"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Address contexts
        ("1713 Springhurst Drive", "seventeen thirteen Springhurst Drive"),
        ("Lives at 2500 Oak Street", "Lives at twenty five hundred Oak Street"),
        ("123 Main Street", "one twenty three Main Street"),
        ("456 Oak Avenue", "four fifty six Oak Avenue"),
        ("789 Pine Drive", "seven eighty nine Pine Drive"),
        ("25135 Harvest Crescent", "two five one three five Harvest Crescent"),
        
        # ZIP codes - these should always be digitized since they are greater than 5 digits
        ("Seattle, WA 98101", "Seattle, WA nine eight one zero one"),
        ("New York, NY 10001", "New York, NY one zero zero zero one"),
        ("Los Angeles, CA 90210", "Los Angeles, CA nine zero two one zero"),
        ("OH 94103", "OH nine four one zero three"),
        ("TX 75201", "TX seven five two zero one"),
        ("Zip code is 42561.", "Zip code is four two five six one."),
    ]
    
    return run_test_group("Address Patterns", test_cases, transformer)

def test_quantity_patterns():
    """Test quantity transformations"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Quantities with units
        ("The house is 2510 sqft", "The house is two thousand five hundred ten sqft"),
        ("Temperature is 75 degrees", "Temperature is seventy five degrees"),
        ("Weighs 150 lbs", "Weighs one hundred fifty lbs"),
        ("The room is 150 sqft", "The room is one hundred fifty sqft"),
        ("Temperature: 72 degrees", "Temperature: seventy two degrees"),
        ("Weight: 180 lbs", "Weight: one hundred eighty lbs"),
        ("Distance: 5 miles", "Distance: five miles"),
        ("Height: 6 feet", "Height: six feet"),
        ("Length: 10 inches", "Length: ten inches"),
        ("My house is 25050 square feet", "My house is twenty five thousand fifty sqft"),
    ]
    
    return run_test_group("Quantity Patterns", test_cases, transformer)

def test_complex_scenarios():
    """Test complex mixed scenarios"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Mixed number types
        ("Call 4256152509 about 1713 Oak St for $299000", 
         "Call four two five, six one five, two five zero nine about seventeen thirteen Oak St for two hundred ninety nine thousand dollars"),
        
        ("Call me at 4256152509 about the house at 1713 Springhurst Drive, 94103 CA. It's 2510 sqft and costs $299000. The temperature is 72 degrees.",
         "Call me at four two five, six one five, two five zero nine about the house at seventeen thirteen Springhurst Drive, nine four one zero three CA. It's two thousand five hundred ten sqft and costs two hundred ninety nine thousand dollars. The temperature is seventy two degrees."),
        
        ("I have 3 cats, 2 dogs, and my phone number is 5551234567. My address is 123 Main St and I paid $50.99 for groceries at 2:30 PM.",
         "I have three cats, two dogs, and my phone number is five five five, one two three, four five six seven. My address is one twenty three Main St and I paid fifty dollars and ninety nine cents for groceries at two thirty PM."),
        
        ("Call 8005551234 at 9:00 AM about 1234 Oak St for $299.99",
         "Call eight zero zero, five five five, one two three four at nine AM about twelve thirty four Oak St for two hundred ninety nine dollars and ninety nine cents"),
        ("To confirm, the address is 699 North Sang Avenue, Fayetteville, Arkansas, 72701. Is that correct? If so, we'll see you at 8:00 AM.", 
         "To confirm, the address is six ninety nine North Sang Avenue, Fayetteville, Arkansas, seven two seven zero one. Is that correct? If so, we'll see you at eight AM."),
        ("Service calls are $89, which is waived with same-day repairs. Your Paschal Pro can accept cash, check, or credit card at the time of service. We have availability tomorrow, Wednesday, September 3, from 8:00 AM to 12:00 PM. Does that time work for you?",
         "Service calls are eighty nine dollars, which is waived with same-day repairs. Your Paschal Pro can accept cash, check, or credit card at the time of service. We have availability tomorrow, Wednesday, September third, from eight AM to twelve PM. Does that time work for you?"),
        ("Service calls are $89, which is waived with same-day repairs. Your Paschal Pro can accept cash, check, or credit card at the time of service. We have availability tomorrow, Wednesday, September 3, from8:00 AM to12:00 PM. Does that time work for you?",
         "Service calls are eighty nine dollars, which is waived with same-day repairs. Your Paschal Pro can accept cash, check, or credit card at the time of service. We have availability tomorrow, Wednesday, September third, from eight AM to twelve PM. Does that time work for you?"),
        # No transformation cases
        ("Hello world", "Hello world"),
        ("No numbers here", "No numbers here"),
        ("How are you?", "How are you?"),
        ("The weather is nice", "The weather is nice"),
        ("Let's meet soon", "Let's meet soon"),
        ("Have a great day", "Have a great day"),
        
        # Edge cases
        ("", ""),
        ("123", "one hundred twenty three"),
        ("12345", "one two three four five"),
    ]
    
    return run_test_group("Complex Scenarios", test_cases, transformer)

def test_pattern_priorities():
    """Test that more specific patterns take priority"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Phone number should be treated as phone, not quantity
        ("4256152509", "four two five, six one five, two five zero nine"),
        
        # Money should be treated as money, not quantity  
        ("$25", "twenty five dollars"),
        
        # Test "$X dollars" pattern preserves dollars word
        ("$299 dollars", "two hundred ninety nine dollars"),
        
        # Time patterns should work correctly
        ("Meet at 8:00 AM", "Meet at eight AM"),
        ("From 9:30 to 2:00 PM", "From nine thirty to two PM"),
    ]
    
    return run_test_group("Pattern Priorities", test_cases, transformer)

def test_date_ordinal_patterns():
    """Test date and ordinal transformations"""
    transformer = TTSTextTransformer()
    
    test_cases = [
        # Long month forms
        ("September 3", "September third"),
        ("January 1", "January first"),
        ("December 25", "December twenty fifth"),
        ("March 21", "March twenty first"),
        ("April 2", "April second"),
        
        # Short month forms
        ("Sep 3", "Sep third"),
        ("Jan 1", "Jan first"),
        ("Dec 25", "Dec twenty fifth"),
        ("Mar 21", "Mar twenty first"),
        ("Apr 2", "Apr second"),
        ("Oct. 15", "Oct. fifteenth"),
        ("Nov. 30", "Nov. thirtieth"),
        
        # Attached dates (no space between month and day)
        ("Sep3", "Sep third"),
        ("Jan1", "Jan first"),
        ("Dec25", "Dec twenty fifth"),
        ("January21", "January twenty first"),
        ("Mar.15", "Mar. fifteenth"),
        ("October31", "October thirty first"),
        
        # Dates with "the"
        ("January the 3", "January the third"),
        ("March the 15", "March the fifteenth"),
        ("Sep the 21", "Sep the twenty first"),
        ("Dec. the 25", "Dec. the twenty fifth"),
        
        # Standalone ordinals
        ("3rd", "third"),
        ("1st", "first"),
        ("2nd", "second"),
        ("21st", "twenty first"),
        ("22nd", "twenty second"),
        ("23rd", "twenty third"),
        ("31st", "thirty first"),
        ("11th", "eleventh"),
        ("12th", "twelfth"),
        ("13th", "thirteenth"),
        
        # In sentences
        ("Meet me on January 15 at noon", "Meet me on January fifteenth at noon"),
        ("The deadline is Mar 3rd", "The deadline is Mar third"),
        ("I'll be there the 21st", "I'll be there the twenty first"),
        ("Born on Feb. the 14", "Born on Feb. the fourteenth"),
    ]
    
    return run_test_group("Date/Ordinal Patterns", test_cases, transformer)

def test_transformations():
    """Run all test groups"""
    print("Testing TTS Text Transformations:")
    print("=" * 60)
    
    test_results = []
    group_names = []
    
    # Run all test groups
    test_functions = [
        ("Time Patterns", test_time_patterns),
        ("Phone Numbers", test_phone_numbers), 
        ("Money Patterns", test_money_patterns),
        ("Address Patterns", test_address_patterns),
        ("Quantity Patterns", test_quantity_patterns),
        ("Complex Scenarios", test_complex_scenarios),
        ("Pattern Priorities", test_pattern_priorities),
        ("Standalone Numbers", test_standalone_numbers),
        ("Date/Ordinal Patterns", test_date_ordinal_patterns),
    ]
    
    for group_name, test_func in test_functions:
        success, p90_time = test_func()
        test_results.append((success, p90_time))
        group_names.append(group_name)
    
    print("\n" + "=" * 60)
    
    # Check test results
    all_passed = all(result[0] for result in test_results)
    passed_count = sum(1 for result in test_results if result[0])
    total_groups = len(test_results)
    
    if all_passed:
        print(f"🎉 All test groups passed! ({passed_count}/{total_groups})")
    else:
        print(f"❌ Some test groups failed. ({passed_count}/{total_groups} passed)")
    
    # Performance check: ensure all P90s are under 1ms
    print("\nPerformance Requirements Check:")
    print("-" * 40)
    
    performance_passed = True
    for i, (group_name, (_, p90_time)) in enumerate(zip(group_names, test_results)):
        status = "✓" if p90_time < 1.0 else "✗"
        print(f"{status} {group_name}: P90 = {p90_time:.3f}ms")
        if p90_time >= 1.0:
            performance_passed = False
    
    print(f"\nPerformance requirement: {'✅ PASSED' if performance_passed else '❌ FAILED'} - All P90s < 1ms")
    
    overall_success = all_passed and performance_passed
    return overall_success


if __name__ == "__main__":
    success = test_transformations()
    sys.exit(0 if success else 1)
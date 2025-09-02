#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
from typing import List


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
        
        # Five digits (ZIP codes) - spell out each digit
        if num < 100000:
            return ' '.join(self.digit_to_word(int(d)) for d in str(num))
        
        # Fallback for larger numbers
        return self.transform_quantity_number(number_str)

    def transform_quantity_number(self, number_str: str) -> str:
        """
        Transform quantity numbers using standard number-to-words conversion.
        
        Examples:
        - 2510 -> "two thousand five hundred ten"
        - 109 -> "one hundred nine"
        - 25000 -> "twenty five thousand"
        """
        num = int(number_str)
        
        if num == 0:
            return "zero"
        
        # Handle negatives
        if num < 0:
            return f"negative {self.transform_quantity_number(str(-num))}"
        
        # Break down into groups
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
            return f"{self.ones[hundreds]} hundred {self.transform_quantity_number(str(remainder))}"
        elif num < 1000000:
            thousands, remainder = divmod(num, 1000)
            if remainder == 0:
                return f"{self.transform_quantity_number(str(thousands))} thousand"
            return f"{self.transform_quantity_number(str(thousands))} thousand {self.transform_quantity_number(str(remainder))}"
        elif num < 1000000000:
            millions, remainder = divmod(num, 1000000)
            if remainder == 0:
                return f"{self.transform_quantity_number(str(millions))} million"
            return f"{self.transform_quantity_number(str(millions))} million {self.transform_quantity_number(str(remainder))}"
        else:
            # For very large numbers, fall back to digit-by-digit
            return ' '.join(self.digit_to_word(int(d)) for d in str(num))

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
            
            # Handle amounts less than $1 (0.xx)
            if dollars_num == 0:
                if cents_num == 0:
                    return "zero"
                elif cents_num < 10:
                    return f"{self.digit_to_word(cents_num)} cents"
                else:
                    return f"{self.two_digit_to_words(cents_num)} cents"
            
            # Handle amounts $1 or more
            if cents_num == 0:
                # $2.00 -> "two" or "two dollars" if preserve_dollars_word
                result = self.transform_quantity_number(dollars)
                return f"{result} dollars" if preserve_dollars_word else result
            elif cents_num < 10:
                # $2.05 -> "two oh five"  
                return f"{self.transform_quantity_number(dollars)} oh {self.digit_to_word(cents_num)}"
            else:
                # $9.99 -> "nine ninety nine"
                return f"{self.transform_quantity_number(dollars)} {self.two_digit_to_words(cents_num)}"
        else:
            # Whole dollar amount
            result = self.transform_quantity_number(number_str)
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
        # Apply transformations in order of specificity
        result = text
        
        # 1. Phone numbers (highest priority)
        # 10-digit phone numbers
        def replace_phone_10(match):
            return self.transform_phone_number(match.group(0))
        result = re.sub(r'\b[0-9]{10}\b', replace_phone_10, result)
        
        # Formatted phone numbers: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx
        def replace_phone_formatted(match):
            digits = re.sub(r'[^0-9]', '', match.group(0))
            return self.transform_phone_number(digits)
        result = re.sub(r'\([0-9]{3}\)\s?[0-9]{3}[-.\s][0-9]{4}', replace_phone_formatted, result)
        result = re.sub(r'\b[0-9]{3}[-.\s][0-9]{3}[-.\s][0-9]{4}\b', replace_phone_formatted, result)
        
        # 2. Time patterns (high priority - before money and standalone numbers)
        def replace_time(match):
            return self.time_transformer(match.group(0))
        result = re.sub(r'\b[0-9]{1,2}:[0-9]{2}(?:\s*[AaPp][Mm])?\b', replace_time, result)
        result = re.sub(r'\b[0-9]{1,2}[AaPp][Mm]\b', replace_time, result)
        
        # 3. Money patterns
        # Handle "$299 dollars" first (more specific pattern) - preserve "dollars" word
        def replace_money_dollars_word(match):
            return self.transform_money_number(match.group(1), preserve_dollars_word=True)
        result = re.sub(r'\$([0-9]+(?:\.[0-9]{2})?)\s+dollars?', replace_money_dollars_word, result, flags=re.IGNORECASE)
        
        # Then handle regular money with $ symbol  
        def replace_money_dollar(match):
            return self.transform_money_number(match.group(1))
        result = re.sub(r'\$([0-9]+(?:\.[0-9]{2})?)', replace_money_dollar, result)
        
        # 4. Address patterns
        # Street addresses: "1713 Springhurst Drive", "2500 Oak Street"
        def replace_street_address(match):
            return self.transform_address_number(match.group(1)) + match.group(2)
        result = re.sub(r'\b([0-9]{1,5})(\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Lane|Ln|Boulevard|Blvd|Way|Place|Pl|Court|Ct))\b', replace_street_address, result, flags=re.IGNORECASE)
        
        # ZIP codes: "94103 CA"
        def replace_zip(match):
            return self.transform_address_number(match.group(1)) + match.group(2)
        result = re.sub(r'\b([0-9]{5}(?:-[0-9]{4})?)(\s+[A-Z]{2})\b', replace_zip, result)
        
        # 5. Quantities with units
        def replace_quantity_with_unit(match):
            return self.transform_quantity_number(match.group(1)) + match.group(2)
        result = re.sub(r'\b([0-9]+)(\s*(?:sqft|sq\.?\s*ft|square\s+feet|degrees?|°|lbs?|pounds?|kg|miles?|mi|feet|ft|inches?|in))\b', replace_quantity_with_unit, result, flags=re.IGNORECASE)
        
        # 6. Standalone numbers (lowest priority)
        def replace_number(match):
            return self.transform_quantity_number(match.group(0))
        result = re.sub(r'\b[0-9]+\b', replace_number, result)
        
        return result
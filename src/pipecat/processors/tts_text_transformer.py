#
# Copyright (c) 2024–2025, Netic Labs
#
#

import re
from typing import List

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
    
    # Spaced time patterns (e.g., "8 00 a m", "12 45 p m")
    SPACED_TIME_PATTERN = r'\b([0-9]{1,2})\s+([0-9]{2})\s+([aApP])\s+([mM])\b'
    
    UNITS_PATTERN = r'\b([0-9]+)(\s*(?:square\s+feet|sqft|sq\.?\s*ft|degrees?|°|lbs?|pounds?|kg|miles?|mi|feet|ft|inches?|in))\b'
    
    NUMBER_ADJACENT_LETTER_PATTERN = r'([a-zA-Z])([0-9]+)'
    COMMA_SEPARATED_NUMBER_PATTERN = r'\b[0-9]{1,3}(,[0-9]{2,3})+\b'
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

            # Convert hour to words
            hour_word = self.number_to_words(hour)

            # For :00 minutes, omit them (e.g., "8:00" -> "eight")
            if minute == "00":
                if am_pm.strip():
                    return f"{hour_word} {am_pm.strip()}"
                else:
                    return hour_word
            # For other minutes, convert to words and separate with space (e.g., "9:30" -> "nine thirty")
            minute_word = self.number_to_words(minute)
            if am_pm.strip():
                return f"{hour_word} {minute_word} {am_pm.strip()}"
            else:
                return f"{hour_word} {minute_word}"

        # Apply colon time transformations first
        text = re.sub(colon_pattern, replace_colon_time, text)

        # Then handle simple time formats: digit + AM/PM (e.g., "9am" -> "9 am")
        # Group 1: hour (1-2 digits)
        # Group 2: AM/PM without leading space
        simple_pattern = r"\b(\d{1,2})([AaPp][Mm])\b"

        def replace_simple_time(match):
            hour = match.group(1)
            am_pm = match.group(2)
            # Convert hour to words and add space between hour and AM/PM
            hour_word = self.number_to_words(hour)
            return f"{hour_word} {am_pm}"

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
        
        # First handle no-boundary patterns to avoid conflicts
        def replace_time_no_boundary(match):
            prefix = match.group(1)
            time_part = match.group(2)
            return prefix + " " + self.time_transformer(time_part)
        result = re.sub(self.TIME_NO_BOUNDARY_COLON_PATTERN, replace_time_no_boundary, result)
        
        def replace_simple_time_no_boundary(match):
            prefix = match.group(1)
            time_part = match.group(2)
            return prefix + " " + self.time_transformer(time_part)
        result = re.sub(self.TIME_NO_BOUNDARY_SIMPLE_PATTERN, replace_simple_time_no_boundary, result)
        
        # Handle spaced time patterns (e.g., "8 00 a m" -> "eight a m")
        def replace_spaced_time(match):
            hour = match.group(1)
            minute = match.group(2)
            a_or_p = match.group(3)
            m = match.group(4)
            
            hour_word = self.number_to_words(hour)
            
            # Omit "00" minutes in spaced format
            if minute == "00":
                return f"{hour_word} {a_or_p} {m}"
            else:
                minute_word = self.number_to_words(minute)
                return f"{hour_word} {minute_word} {a_or_p} {m}"
        result = re.sub(self.SPACED_TIME_PATTERN, replace_spaced_time, result)
        
        # Then handle normal boundary patterns
        result = re.sub(self.TIME_COLON_PATTERN, replace_time, result)
        result = re.sub(self.TIME_SIMPLE_PATTERN, replace_time, result)
        
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
            # Convert common unit abbreviations to full words
            unit_text = unit_text.replace("sqft", "square feet").replace("SQFT", "square feet")
            unit_text = unit_text.replace("lbs", "pounds").replace("LBS", "pounds") 
            unit_text = unit_text.replace("kg", "kilograms").replace("KG", "kilograms")
            return number_text + unit_text
        result = re.sub(self.UNITS_PATTERN, replace_quantity_with_unit, result, flags=re.IGNORECASE)
        
        # 7. Numbers adjacent to letters (like "from8" -> "from eight")
        def replace_number_adjacent_letter(match):
            prefix = match.group(1)
            number = match.group(2)
            return prefix + " " + self.transform_quantity_number(number)
        result = re.sub(self.NUMBER_ADJACENT_LETTER_PATTERN, replace_number_adjacent_letter, result)
        
        # 8. Comma-separated numbers (like "25,000")  
        def replace_comma_number(match):
            # Remove commas and convert to standard number format
            full_match = match.group(0)
            # Handle cases like "25,00" -> "25000"
            if full_match.endswith(",00"):
                clean_number = full_match.replace(",00", "000")
            else:
                clean_number = full_match.replace(',', '')
            # Use standard format for comma-separated numbers
            return self.number_to_standard(clean_number)
        result = re.sub(self.COMMA_SEPARATED_NUMBER_PATTERN, replace_comma_number, result)
        
        # 9. All remaining standalone numbers (default to quantity transformation)
        def replace_number(match):
            return self.transform_quantity_number(match.group(0))
        result = re.sub(self.STANDALONE_NUMBER_PATTERN, replace_number, result)
        
        return result

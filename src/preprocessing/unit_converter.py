"""
Deterministic Unit Converter for English-to-Ukrainian Literary Translation.

Provides rule-based conversions for Imperial measurement units to Metric equivalents,
accurate Ukrainian Slavic grammatical pluralization & declension rules, English
number-word parsing, and compound measurement handling.
"""

from __future__ import annotations
import re
from typing import Tuple, Optional, Dict, Any, Union, List


class UnitConverter:
    """
    Deterministic rule-based unit converter supporting:
    - Distance: feet, inches, yards, miles
    - Mass: pounds, ounces
    - Speed: mph
    - Temperature: Fahrenheit to Celsius
    - Ukrainian declension agreement for integers and decimal fractions
    - English word-numbers: 'eighty feet' -> '24 метри', 'fifteen miles' -> '24 кілометри'
    - Compound measurements: '6 feet 2 inches' / '6\\'2"' -> '1.88 м'
    - Policies: 'metric' (default), 'preserve', 'dual'
    """

    WORD_NUMBERS: Dict[str, int] = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1000000
    }

    # Forms: (singular_nom, plural_nom_2_4, plural_gen_5_plus, singular_gen_decimal, abbreviation)
    UKRAINIAN_UNIT_FORMS: Dict[str, Tuple[str, str, str, str, str]] = {
        'meter': ('метр', 'метри', 'метрів', 'метра', 'м'),
        'centimeter': ('сантиметр', 'сантиметри', 'сантиметрів', 'сантиметра', 'см'),
        'millimeter': ('міліметр', 'міліметри', 'міліметрів', 'міліметра', 'мм'),
        'kilometer': ('кілометр', 'кілометри', 'кілометрів', 'кілометра', 'км'),
        'kilogram': ('кілограм', 'кілограми', 'кілограмів', 'кілограма', 'кг'),
        'gram': ('грам', 'грами', 'грамів', 'грама', 'г'),
        'celsius': ('градус за Цельсієм', 'градуси за Цельсієм', 'градусів за Цельсієм', 'градуса за Цельсієм', '°C'),
        'kmh': ('кілометр за годину', 'кілометри за годину', 'кілометрів за годину', 'кілометра за годину', 'км/год'),
    }

    ENGLISH_UNIT_FORMS: Dict[str, Tuple[str, str, str, str, str]] = {
        'meter': ('meter', 'meters', 'meters', 'meters', 'm'),
        'centimeter': ('centimeter', 'centimeters', 'centimeters', 'centimeters', 'cm'),
        'millimeter': ('millimeter', 'millimeters', 'millimeters', 'millimeters', 'mm'),
        'kilometer': ('kilometer', 'kilometers', 'kilometers', 'kilometers', 'km'),
        'kilogram': ('kilogram', 'kilograms', 'kilograms', 'kilograms', 'kg'),
        'gram': ('gram', 'grams', 'grams', 'grams', 'g'),
        'celsius': ('degree Celsius', 'degrees Celsius', 'degrees Celsius', 'degrees Celsius', '°C'),
        'kmh': ('km/h', 'km/h', 'km/h', 'km/h', 'km/h'),
    }

    def __init__(self, default_policy: str = "metric"):
        self.default_policy = default_policy

    @classmethod
    def get_ukrainian_unit_form(
        cls,
        value: float,
        forms: Tuple[str, str, str, str, str],
        abbreviated: bool = False
    ) -> str:
        """
        Determines the correct Ukrainian grammatical case and number for a unit.

        Rules:
        - Decimals (e.g. 4.6, 1.8, 24.5): родовий однини (forms[3] - 'метра', 'кілометра')
        - 1, 21, 31... (ends with 1, except 11): називний однини (forms[0] - 'метр')
        - 2, 3, 4, 22, 23, 24... (ends with 2-4, except 12-14): називний множини (forms[1] - 'метри')
        - 5-20, 0, ends with 5-9, 0, 11-19: родовий множини (forms[2] - 'метрів')
        """
        if abbreviated:
            return forms[4]

        # Check for fractional numbers
        if not float(value).is_integer():
            return forms[3]

        int_val = int(abs(round(value)))
        mod100 = int_val % 100
        if 11 <= mod100 <= 19:
            return forms[2]

        mod10 = int_val % 10
        if mod10 == 1:
            return forms[0]
        elif 2 <= mod10 <= 4:
            return forms[1]
        else:
            return forms[2]

    @classmethod
    def get_ukrainian_unit(
        cls,
        num: float,
        forms: Tuple[str, str, str, str, str],
        abbreviated: bool = False
    ) -> str:
        """Alias for get_ukrainian_unit_form."""
        return cls.get_ukrainian_unit_form(num, forms, abbreviated)

    @classmethod
    def parse_number_expression(cls, expr: str) -> Optional[float]:
        """
        Parses digits, decimals, fractions (1/2, 3/4), and English word-numbers.
        Examples: '80', '15.5', '1/2', 'eighty', 'fifteen', 'twenty-four', 'one hundred fifty'.
        """
        if not expr:
            return None
        expr = expr.strip().lower()

        # Direct digit or decimal (e.g. 80, 4.6, 4,6)
        if re.match(r'^\d+(?:[.,]\d+)?$', expr):
            try:
                return float(expr.replace(',', '.'))
            except ValueError:
                return None

        # Fraction format (e.g. 1/2, 3/4)
        if '/' in expr:
            parts = expr.split('/')
            if len(parts) == 2:
                try:
                    num_val = float(parts[0].strip())
                    denom_val = float(parts[1].strip())
                    if denom_val != 0:
                        return num_val / denom_val
                except ValueError:
                    pass

        # Check for mixed numbers like '2 1/2' or '2 and a half'
        mixed_match = re.match(r'^(\d+)\s+(?:and\s+)?(\d+/\d+)$', expr)
        if mixed_match:
            whole = float(mixed_match.group(1))
            frac_parts = mixed_match.group(2).split('/')
            if len(frac_parts) == 2 and float(frac_parts[1]) != 0:
                return whole + (float(frac_parts[0]) / float(frac_parts[1]))

        # Word numbers and expressions
        # Normalize connectors and hyphens
        cleaned = expr.replace('-', ' ').replace(',', ' ')
        tokens = cleaned.split()
        if not tokens:
            return None

        # Special phrases
        if tokens == ['a'] or tokens == ['an'] or tokens == ['one']:
            return 1.0
        if tokens == ['half'] or tokens == ['a', 'half'] or tokens == ['half', 'a']:
            return 0.5
        if tokens == ['quarter'] or tokens == ['a', 'quarter']:
            return 0.25

        total = 0.0
        current = 0.0
        has_parsed_word = False
        i = 0
        while i < len(tokens):
            w = tokens[i]
            if w == 'and':
                i += 1
                continue
            if w in ('a', 'an') and i + 1 < len(tokens) and tokens[i + 1] in ('half', 'quarter', 'hundred', 'thousand'):
                i += 1
                continue
            if w == 'half':
                current += 0.5
                has_parsed_word = True
            elif w == 'quarter':
                current += 0.25
                has_parsed_word = True
            elif w in cls.WORD_NUMBERS:
                val = cls.WORD_NUMBERS[w]
                has_parsed_word = True
                if val == 100:
                    current = (current if current else 1.0) * 100.0
                elif val in (1000, 1000000):
                    total += (current if current else 1.0) * float(val)
                    current = 0.0
                else:
                    current += float(val)
            elif re.match(r'^\d+(?:[.,]\d+)?$', w):
                current += float(w.replace(',', '.'))
                has_parsed_word = True
            else:
                return None
            i += 1

        if not has_parsed_word:
            return None

        result = total + current
        return result

    @classmethod
    def _format_number(cls, value: float) -> str:
        """Formats a float cleanly (e.g. 24.0 -> '24', 4.6 -> '4.6')."""
        if float(value).is_integer():
            return str(int(round(value)))
        # Clean precision up to 2 decimal places, stripping trailing zeroes
        formatted = f"{value:.2f}".rstrip('0').rstrip('.')
        return formatted

    @classmethod
    def convert_feet_to_meters(
        cls,
        feet: float,
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts feet to meters.
        >= 10 m: rounds to nearest integer (e.g. 80 feet -> 24 метри)
        < 10 m: rounds to 1 decimal (e.g. 15 ft -> 4.6 метра, 6 feet -> 1.8 метра)
        """
        meters_raw = feet * 0.3048
        if meters_raw >= 10.0:
            rounded = round(meters_raw)
        else:
            rounded = round(meters_raw, 1)

        forms = cls.UKRAINIAN_UNIT_FORMS['meter'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['meter']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_inches_to_cm(
        cls,
        inches: float,
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts inches to centimeters.
        >= 10 cm: rounds to nearest integer (e.g. 6 inches -> 15 см / 15 сантиметрів)
        < 10 cm: rounds to 1 decimal (e.g. 2 inches -> 5.1 см)
        """
        cm_raw = inches * 2.54
        if cm_raw >= 10.0:
            rounded = round(cm_raw)
        else:
            rounded = round(cm_raw, 1)

        forms = cls.UKRAINIAN_UNIT_FORMS['centimeter'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['centimeter']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_miles_to_km(
        cls,
        miles: float,
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts miles to kilometers.
        e.g. 5 miles -> 8 кілометрів, 15 miles -> 24 кілометри, 100 miles -> 161 кілометр
        """
        km_raw = miles * 1.609344
        if km_raw >= 10.0 or abs(km_raw - round(km_raw)) < 0.15:
            rounded = round(km_raw)
        else:
            rounded = round(km_raw, 1)

        forms = cls.UKRAINIAN_UNIT_FORMS['kilometer'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['kilometer']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_yards_to_meters(
        cls,
        yards: float,
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts yards to meters.
        e.g. 10 yards -> 9.1 метра, 100 yards -> 91 метр
        """
        meters_raw = yards * 0.9144
        if meters_raw >= 100.0 or abs(meters_raw - round(meters_raw)) < 0.1:
            rounded = round(meters_raw)
        else:
            rounded = round(meters_raw, 1)

        forms = cls.UKRAINIAN_UNIT_FORMS['meter'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['meter']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_pounds_to_kg(
        cls,
        pounds: float,
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts pounds to kilograms.
        e.g. 100 lbs -> 45.4 кг / 45.4 кілограма, 150 lbs -> 68 кг
        """
        kg_raw = pounds * 0.45359237
        rounded = round(kg_raw, 1) if not float(round(kg_raw, 1)).is_integer() else round(kg_raw)

        forms = cls.UKRAINIAN_UNIT_FORMS['kilogram'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['kilogram']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_ounces_to_grams(
        cls,
        ounces: float,
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts ounces to grams.
        e.g. 1 oz -> 28 грамів, 8 oz -> 227 грамів
        """
        g_raw = ounces * 28.349523125
        if g_raw >= 10.0:
            rounded = round(g_raw)
        else:
            rounded = round(g_raw, 1)

        forms = cls.UKRAINIAN_UNIT_FORMS['gram'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['gram']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_fahrenheit_to_celsius(
        cls,
        f: float,
        abbreviated: bool = True,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts Fahrenheit to Celsius.
        e.g. 68°F -> 20°C, 32°F -> 0°C, 100°F -> 38°C
        """
        c_raw = (f - 32.0) * 5.0 / 9.0
        rounded = round(c_raw)
        forms = cls.UKRAINIAN_UNIT_FORMS['celsius'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['celsius']
        unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
        return rounded, unit_str

    @classmethod
    def convert_compound_feet_inches(
        cls,
        feet: float,
        inches: float,
        target_unit: str = "m",
        abbreviated: bool = False,
        lang: str = "uk"
    ) -> Tuple[float, str]:
        """
        Converts compound measurements (e.g. 6 feet 2 inches, 6'2") into meters or cm.
        6 feet 2 inches -> 1.88 м (1.88 метра)
        """
        total_meters = (feet * 0.3048) + (inches * 0.0254)
        if target_unit == "cm":
            cm_val = total_meters * 100.0
            rounded = round(cm_val) if cm_val >= 10 else round(cm_val, 1)
            forms = cls.UKRAINIAN_UNIT_FORMS['centimeter'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['centimeter']
            unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
            return rounded, unit_str
        else:
            rounded = round(total_meters, 2)
            forms = cls.UKRAINIAN_UNIT_FORMS['meter'] if lang == "uk" else cls.ENGLISH_UNIT_FORMS['meter']
            unit_str = cls.get_ukrainian_unit_form(rounded, forms, abbreviated)
            return rounded, unit_str

    @classmethod
    def convert_text(
        cls,
        text: str,
        target_lang: str = "uk",
        style: str = "auto",
        policy: str = "metric"
    ) -> str:
        """
        Scans text and deterministically converts imperial measurement expressions
        into their metric equivalents according to policy and target language.

        :param text: Source text containing imperial measurements
        :param target_lang: 'uk' for Ukrainian, 'en' for English metric
        :param style: 'words' (e.g. 24 метри), 'abbr' (e.g. 24 м), or 'auto'
        :param policy: 'metric' (default conversion), 'preserve' (original text), 'dual' (e.g. 80 feet (24 метри))
        """
        if not text or policy == "preserve":
            return text

        result = text

        # 1. Temperature: 72°F, 72 °F, 72 degrees Fahrenheit, 72 deg F
        temp_pattern = re.compile(
            r'\b(\d+(?:[.,]\d+)?)\s*(?:°\s*F|degrees?\s+Fahrenheit|deg\.?\s*F|deg\s+Fahrenheit)\b',
            re.IGNORECASE
        )

        def _replace_temp(m: re.Match) -> str:
            val = float(m.group(1).replace(',', '.'))
            r_val, _ = cls.convert_fahrenheit_to_celsius(val, abbreviated=True, lang=target_lang)
            conv_str = f"{cls._format_number(r_val)} °C"
            if policy == "dual":
                return f"{m.group(0)} ({conv_str})"
            return conv_str

        result = temp_pattern.sub(_replace_temp, result)

        # 2. Compound measurements: 6'2", 6' 2", 6 feet 2 inches, 6 ft 2 in
        compound_pattern = re.compile(
            r'(?:\b(\d+)\s*(?:feet|foot|ft)\s*(\d+)\s*(?:inches|inch|in\.?)\b|(?:\b(\d+)\s*\'\s*(\d+)\s*(?:\"|”|\b)))',
            re.IGNORECASE
        )

        def _replace_compound(m: re.Match) -> str:
            f_val = float(m.group(1) or m.group(3))
            i_val = float(m.group(2) or m.group(4))
            is_abbr = (style == "abbr") or (
                style == "auto" and bool(re.search(r'(\'|"|”|\bft\.?\b|\bin\.?\b)', m.group(0), re.IGNORECASE))
            )
            r_val, unit_str = cls.convert_compound_feet_inches(f_val, i_val, abbreviated=is_abbr, lang=target_lang)
            conv_str = f"{cls._format_number(r_val)} {unit_str}"
            if policy == "dual":
                return f"{m.group(0)} ({conv_str})"
            return conv_str

        result = compound_pattern.sub(_replace_compound, result)

        # 3. Compound mass: 5 lbs 4 oz, 5 pounds 4 ounces
        compound_mass_pattern = re.compile(
            r'\b(\d+)\s*(?:pounds|pound|lbs?\.?)\s*(\d+)\s*(?:ounces|ounce|oz\.?)\b',
            re.IGNORECASE
        )

        def _replace_compound_mass(m: re.Match) -> str:
            p_val = float(m.group(1))
            o_val = float(m.group(2))
            total_kg = (p_val * 0.45359237) + (o_val * 0.028349523)
            rounded = round(total_kg, 1)
            is_abbr = (style == "abbr") or (
                style == "auto" and bool(re.search(r'(\blbs?\.?\b|\boz\.?\b)', m.group(0), re.IGNORECASE))
            )
            forms = cls.UKRAINIAN_UNIT_FORMS['kilogram'] if target_lang == "uk" else cls.ENGLISH_UNIT_FORMS['kilogram']
            unit_str = cls.get_ukrainian_unit_form(rounded, forms, is_abbr)
            conv_str = f"{cls._format_number(rounded)} {unit_str}"
            if policy == "dual":
                return f"{m.group(0)} ({conv_str})"
            return conv_str

        result = compound_mass_pattern.sub(_replace_compound_mass, result)

        # 4. Spelled-out English Word Numbers with Units:
        # e.g. "eighty feet", "fifteen miles", "six inches", "twenty-four miles", "one hundred pounds"
        word_num_unit_pattern = re.compile(
            r'\b((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
            r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|a|an|half)\s*(?:-|\s+and\s+|\s+)?)+)\s+'
            r'(feet|foot|inches|inch|miles|mile|yards|yard|pounds|pound|ounces|ounce|mph)\b',
            re.IGNORECASE
        )

        def _replace_word_unit(m: re.Match) -> str:
            expr_str = m.group(1).strip()
            unit_raw = m.group(2).lower()
            val = cls.parse_number_expression(expr_str)
            if val is None:
                return m.group(0)

            is_abbr = (style == "abbr")
            conv_val, unit_name = cls._convert_value_and_unit(val, unit_raw, is_abbr, target_lang)
            if conv_val is None or unit_name is None:
                return m.group(0)

            conv_str = f"{cls._format_number(conv_val)} {unit_name}"
            if policy == "dual":
                return f"{m.group(0)} ({conv_str})"
            return conv_str

        result = word_num_unit_pattern.sub(_replace_word_unit, result)

        # 5. Standard Numeric Expressions: e.g. "80 feet", "15 ft", "6 in", "5 miles", "100 lbs"
        # Protect against false positive on standalone "in" (preposition): only match "in" if preceded by a number
        num_unit_pattern = re.compile(
            r'\b(\d+(?:[.,]\d+)?)\s*'
            r'(feet|foot|ft\.?|inches|inch|in\.?|yards?|yd\.?|miles?|mi\.?|pounds?|lbs?\.?|ounces?|oz\.?|mph)\b',
            re.IGNORECASE
        )

        def _replace_num_unit(m: re.Match) -> str:
            num_str = m.group(1).replace(',', '.')
            unit_raw = m.group(2).lower().rstrip('.')
            try:
                val = float(num_str)
            except ValueError:
                return m.group(0)

            # Determine abbreviation preference based on source or style parameter
            is_abbr = (style == "abbr") or (style == "auto" and unit_raw in ('ft', 'in', 'yd', 'mi', 'lb', 'lbs', 'oz', 'mph'))
            conv_val, unit_name = cls._convert_value_and_unit(val, unit_raw, is_abbr, target_lang)
            if conv_val is None or unit_name is None:
                return m.group(0)

            conv_str = f"{cls._format_number(conv_val)} {unit_name}"
            if policy == "dual":
                return f"{m.group(0)} ({conv_str})"
            return conv_str

        result = num_unit_pattern.sub(_replace_num_unit, result)

        return result

    @classmethod
    def _convert_value_and_unit(
        cls,
        val: float,
        unit_raw: str,
        is_abbr: bool,
        target_lang: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Internal router to calculate converted value and localized unit string."""
        if unit_raw in ('feet', 'foot', 'ft'):
            return cls.convert_feet_to_meters(val, is_abbr, target_lang)
        elif unit_raw in ('inches', 'inch', 'in'):
            return cls.convert_inches_to_cm(val, is_abbr, target_lang)
        elif unit_raw in ('miles', 'mile', 'mi'):
            return cls.convert_miles_to_km(val, is_abbr, target_lang)
        elif unit_raw in ('yards', 'yard', 'yd'):
            return cls.convert_yards_to_meters(val, is_abbr, target_lang)
        elif unit_raw in ('pounds', 'pound', 'lb', 'lbs'):
            return cls.convert_pounds_to_kg(val, is_abbr, target_lang)
        elif unit_raw in ('ounces', 'ounce', 'oz'):
            return cls.convert_ounces_to_grams(val, is_abbr, target_lang)
        elif unit_raw == 'mph':
            kmh = val * 1.609344
            rounded = round(kmh)
            forms = cls.UKRAINIAN_UNIT_FORMS['kmh'] if target_lang == "uk" else cls.ENGLISH_UNIT_FORMS['kmh']
            unit_str = cls.get_ukrainian_unit_form(rounded, forms, is_abbr)
            return rounded, unit_str
        return None, None

    def convert_sentence(
        self,
        sentence: Any,
        target_lang: str = "uk",
        policy: Optional[str] = None
    ) -> Any:
        """
        Converts units in a Sentence domain object or plain string.
        """
        active_policy = policy or self.default_policy
        if isinstance(sentence, str):
            return self.convert_text(sentence, target_lang=target_lang, policy=active_policy)

        # Handle domain model Sentence
        if hasattr(sentence, "translated_text") and sentence.translated_text:
            sentence.translated_text = self.convert_text(
                sentence.translated_text, target_lang=target_lang, policy=active_policy
            )
        elif hasattr(sentence, "original_text") and sentence.original_text:
            sentence.original_text = self.convert_text(
                sentence.original_text, target_lang=target_lang, policy=active_policy
            )

        return sentence

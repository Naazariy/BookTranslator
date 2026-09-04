"""
Unit tests for UnitConverter, QualityPipeline mixed-script cleaner, and PreprocessingPipeline.
"""

import pytest
from unittest.mock import MagicMock

from src.preprocessing.unit_converter import UnitConverter
from src.quality.pipeline import QualityPipeline, sanitize_mixed_script_words
from src.preprocessing.pipeline import PreprocessingPipeline
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk
from src.domain.models.quality import IssueSeverity


class TestUnitConverter:
    """Test suite for deterministic Imperial to Metric unit conversions."""

    def test_ukrainian_declension_rules(self):
        meter_forms = UnitConverter.UKRAINIAN_UNIT_FORMS['meter']
        
        # 1: Nominative singular
        assert UnitConverter.get_ukrainian_unit_form(1, meter_forms) == 'метр'
        assert UnitConverter.get_ukrainian_unit_form(21, meter_forms) == 'метр'
        assert UnitConverter.get_ukrainian_unit_form(101, meter_forms) == 'метр'

        # 2, 3, 4: Nominative plural
        assert UnitConverter.get_ukrainian_unit_form(2, meter_forms) == 'метри'
        assert UnitConverter.get_ukrainian_unit_form(3, meter_forms) == 'метри'
        assert UnitConverter.get_ukrainian_unit_form(4, meter_forms) == 'метри'
        assert UnitConverter.get_ukrainian_unit_form(22, meter_forms) == 'метри'
        assert UnitConverter.get_ukrainian_unit_form(24, meter_forms) == 'метри'

        # 5-20, 0, 11-19, endings with 5-9, 0: Genitive plural
        assert UnitConverter.get_ukrainian_unit_form(0, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(5, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(10, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(11, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(12, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(14, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(19, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(20, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(80, meter_forms) == 'метрів'
        assert UnitConverter.get_ukrainian_unit_form(100, meter_forms) == 'метрів'

        # Decimals: Genitive singular
        assert UnitConverter.get_ukrainian_unit_form(4.6, meter_forms) == 'метра'
        assert UnitConverter.get_ukrainian_unit_form(1.8, meter_forms) == 'метра'
        assert UnitConverter.get_ukrainian_unit_form(24.5, meter_forms) == 'метра'

        # Other units
        kg_forms = UnitConverter.UKRAINIAN_UNIT_FORMS['kilogram']
        assert UnitConverter.get_ukrainian_unit_form(1, kg_forms) == 'кілограм'
        assert UnitConverter.get_ukrainian_unit_form(4, kg_forms) == 'кілограми'
        assert UnitConverter.get_ukrainian_unit_form(20, kg_forms) == 'кілограмів'
        assert UnitConverter.get_ukrainian_unit_form(45.4, kg_forms) == 'кілограма'

    def test_feet_to_meters(self):
        # 80 feet -> 24 метри
        val, unit = UnitConverter.convert_feet_to_meters(80, abbreviated=False)
        assert val == 24
        assert unit == 'метри'
        assert UnitConverter.convert_text("80 feet") == "24 метри"

        # 15 ft -> 4.6 м / 15 feet -> 4.6 метра
        val, unit = UnitConverter.convert_feet_to_meters(15, abbreviated=False)
        assert val == 4.6
        assert unit == 'метра'
        assert UnitConverter.convert_text("15 ft") == "4.6 м"
        assert UnitConverter.convert_text("15 feet") == "4.6 метра"

        # 6 feet -> 1.8 метра
        val, unit = UnitConverter.convert_feet_to_meters(6, abbreviated=False)
        assert val == 1.8
        assert unit == 'метра'
        assert UnitConverter.convert_text("6 feet") == "1.8 метра"

    def test_inches_to_cm(self):
        # 6 inches -> 15 сантиметрів / 6 in -> 15 см
        val, unit = UnitConverter.convert_inches_to_cm(6, abbreviated=False)
        assert val == 15
        assert unit == 'сантиметрів'
        assert UnitConverter.convert_text("6 inches") == "15 сантиметрів"
        assert UnitConverter.convert_text("6 in") == "15 см"

        # 1 inch -> 2.5 сантиметра
        val, unit = UnitConverter.convert_inches_to_cm(1, abbreviated=False)
        assert val == 2.5
        assert unit == 'сантиметра'

    def test_miles_to_km(self):
        # 5 miles -> 8 кілометрів
        val, unit = UnitConverter.convert_miles_to_km(5, abbreviated=False)
        assert val == 8
        assert unit == 'кілометрів'
        assert UnitConverter.convert_text("5 miles") == "8 кілометрів"

        # 15 miles -> 24 кілометри
        val, unit = UnitConverter.convert_miles_to_km(15, abbreviated=False)
        assert val == 24
        assert unit == 'кілометри'
        assert UnitConverter.convert_text("15 miles") == "24 кілометри"

        # 100 miles -> 161 кілометр
        val, unit = UnitConverter.convert_miles_to_km(100, abbreviated=False)
        assert val == 161
        assert unit == 'кілометр'
        assert UnitConverter.convert_text("100 miles") == "161 кілометр"

    def test_yards_to_meters(self):
        # 10 yards -> 9.1 метра
        val, unit = UnitConverter.convert_yards_to_meters(10, abbreviated=False)
        assert val == 9.1
        assert unit == 'метра'
        assert UnitConverter.convert_text("10 yards") == "9.1 метра"

    def test_pounds_to_kg(self):
        # 100 lbs -> 45.4 кг / 100 pounds -> 45.4 кілограма
        val, unit = UnitConverter.convert_pounds_to_kg(100, abbreviated=False)
        assert val == 45.4
        assert unit == 'кілограма'
        assert UnitConverter.convert_text("100 lbs") == "45.4 кг"
        assert UnitConverter.convert_text("100 pounds") == "45.4 кілограма"

    def test_ounces_to_grams(self):
        # 1 oz -> 28 грамів
        val, unit = UnitConverter.convert_ounces_to_grams(1, abbreviated=False)
        assert val == 28
        assert unit == 'грамів'
        assert UnitConverter.convert_text("1 oz") == "28 г"

    def test_speed_mph(self):
        assert UnitConverter.convert_text("60 mph") == "97 км/год"
        assert UnitConverter.convert_text("100 mph") == "161 км/год"

    def test_temperature_fahrenheit(self):
        assert UnitConverter.convert_text("72 °F") == "22 °C"
        assert UnitConverter.convert_text("68°F") == "20 °C"
        assert UnitConverter.convert_text("32 degrees Fahrenheit") == "0 °C"

    def test_word_numbers_parsing(self):
        assert UnitConverter.parse_number_expression("eighty") == 80.0
        assert UnitConverter.parse_number_expression("fifteen") == 15.0
        assert UnitConverter.parse_number_expression("six") == 6.0
        assert UnitConverter.parse_number_expression("twenty-four") == 24.0
        assert UnitConverter.parse_number_expression("one hundred fifty") == 150.0

        assert UnitConverter.convert_text("eighty feet") == "24 метри"
        assert UnitConverter.convert_text("fifteen miles") == "24 кілометри"
        assert UnitConverter.convert_text("six inches") == "15 сантиметрів"
        assert UnitConverter.convert_text("one hundred pounds") == "45.4 кілограма"

    def test_compound_measurements(self):
        # 6 feet 2 inches -> 1.88 метра
        assert UnitConverter.convert_text("6 feet 2 inches") == "1.88 метра"
        assert UnitConverter.convert_text("6'2\"") == "1.88 м"
        assert UnitConverter.convert_text("5 lbs 4 oz") == "2.4 кг"

    def test_fractions_and_mixed_expressions(self):
        assert UnitConverter.parse_number_expression("1/2") == 0.5
        assert UnitConverter.parse_number_expression("3/4") == 0.75
        assert UnitConverter.parse_number_expression("2 1/2") == 2.5
        assert UnitConverter.parse_number_expression("half") == 0.5
        assert UnitConverter.parse_number_expression("a half") == 0.5
        assert UnitConverter.parse_number_expression("a quarter") == 0.25

    def test_style_variations(self):
        # Force style='abbr'
        assert UnitConverter.convert_text("80 feet", style="abbr") == "24 м"
        assert UnitConverter.convert_text("100 pounds", style="abbr") == "45.4 кг"
        # Force style='words'
        assert UnitConverter.convert_text("15 ft", style="words") == "4.6 метра"
        assert UnitConverter.convert_text("100 lbs", style="words") == "45.4 кілограма"

    def test_english_target_language(self):
        assert UnitConverter.convert_text("80 feet", target_lang="en") == "24 meters"
        assert UnitConverter.convert_text("15 ft", target_lang="en") == "4.6 m"
        assert UnitConverter.convert_text("6 inches", target_lang="en") == "15 centimeters"
        assert UnitConverter.convert_text("100 lbs", target_lang="en") == "45.4 kg"

    def test_multi_unit_paragraph(self):
        src = "The monster stood 6 feet tall, weighed 200 lbs, and could run 15 miles in 68°F weather."
        converted = UnitConverter.convert_text(src)
        assert "1.8 метра" in converted
        assert "90.7 кілограма" in converted or "91 кілограм" in converted or "90.7" in converted
        assert "24 кілометри" in converted
        assert "20 °C" in converted

    def test_convert_sentence_and_policies(self):
        conv = UnitConverter(default_policy="metric")
        
        # Test plain string
        assert conv.convert_sentence("The wall is 80 feet high.") == "The wall is 24 метри high."

        # Test Sentence domain model
        s = Sentence(original_text="The cliff is 15 feet tall.")
        conv.convert_sentence(s)
        assert s.normalized_source_text == "The cliff is 4.6 метра tall."
        assert s.original_text == "The cliff is 15 feet tall."


        # Test dual policy
        res_dual = UnitConverter.convert_text("80 feet", policy="dual")
        assert "80 feet (24 метри)" == res_dual

        # Test preserve policy
        res_pres = UnitConverter.convert_text("80 feet", policy="preserve")
        assert "80 feet" == res_pres


class TestMixedScriptSanitizer:
    """Test suite for mixed Cyrillic-Latin homoglyphs and portmanteau cleaning."""

    def test_known_portmanteaus(self):
        assert sanitize_mixed_script_words("Смачнissimo") == "Смакота"
        assert sanitize_mixed_script_words("смачнissimo") == "смакота"
        assert sanitize_mixed_script_words("Смачниссимо") == "Смакота"
        assert sanitize_mixed_script_words("Гарнissimo") == "Прегарно"
        assert sanitize_mixed_script_words("чудовissimo") == "чудово"
        assert sanitize_mixed_script_words("Прекраснissimo") == "Прекрасно"

    def test_homoglyph_replacement(self):
        # 'мiсто' with Latin 'i' (\u0069)
        mixed_misto = "м\u0069сто"
        assert sanitize_mixed_script_words(mixed_misto) == "місто"

        # 'свiт' with Latin 'i'
        mixed_svit = "св\u0069т"
        assert sanitize_mixed_script_words(mixed_svit) == "світ"

        # 'рiка' with Latin 'i' and 'a' (\u0069, \u0061)
        mixed_rika = "р\u0069к\u0061"
        assert sanitize_mixed_script_words(mixed_rika) == "ріка"

        # 'сонцe' with Latin 'e' (\u0065)
        mixed_sontse = "сонц\u0065"
        assert sanitize_mixed_script_words(mixed_sontse) == "сонце"

        # 'пoле' with Latin 'o' (\u006f)
        mixed_pole = "п\u006fле"
        assert sanitize_mixed_script_words(mixed_pole) == "поле"

    def test_foreign_suffix_stripping(self):
        assert sanitize_mixed_script_words("рабитfolk") == "рабит"
        assert sanitize_mixed_script_words("детективlike") == "детектив"


class TestQualityPipeline:
    """Test suite for QualityPipeline validation."""

    def test_validate_empty_translation(self):
        qp = QualityPipeline()
        chunk = MagicMock(spec=TranslationChunk)
        report = qp.validate(chunk, "")
        assert not report.is_passed
        assert report.translation_score == 0.0
        assert any(i.rule_name == "EmptyTranslation" for i in report.issues)

    def test_validate_mixed_script_issue(self):
        qp = QualityPipeline()
        chunk = MagicMock(spec=TranslationChunk)
        report = qp.validate(chunk, "Це не смачнissimo страва")
        assert any(i.rule_name == "MixedScriptWord" for i in report.issues)

    def test_validate_leaked_tags(self):
        qp = QualityPipeline()
        chunk = MagicMock(spec=TranslationChunk)
        report = qp.validate(chunk, "Це речення містить <tag_1> незакритий тег.")
        assert any(i.rule_name == "LeakedTechnicalTag" for i in report.issues)

    def test_validate_clean_translation(self):
        qp = QualityPipeline()
        chunk = MagicMock(spec=TranslationChunk)
        report = qp.validate(chunk, "Це неймовірно смачна страва.")
        assert report.is_passed
        assert len(report.issues) == 0

    def test_post_process_method(self):
        qp = QualityPipeline()
        assert qp.post_process("Смачнissimo вечеря") == "Смакота вечеря"
        assert qp.clean_text("м\u0069сто Київ") == "місто Київ"


class TestPreprocessingPipelineIntegration:
    """Test suite for PreprocessingPipeline unit conversion integration."""

    def test_preprocessing_converts_book_units_when_enabled(self):
        kb_mock = MagicMock()
        pipe = PreprocessingPipeline(kb_repo=kb_mock, convert_units=True)

        sent1 = Sentence(original_text="The tower was 80 feet high.")
        sent2 = Sentence(original_text="We walked 5 miles today.")
        p = Paragraph(sentences=[sent1, sent2])
        chap = Chapter(title="Chapter 1", order_index=0, paragraphs=[p])
        book = Book(title="Test Adventure", chapters=[chap], source_language="en", target_language="uk")

        pipe.process(book)

        assert sent1.normalized_source_text == "The tower was 24 метри high."
        assert sent1.original_text == "The tower was 80 feet high."
        assert sent2.normalized_source_text == "We walked 8 кілометрів today."
        assert sent2.original_text == "We walked 5 miles today."


    def test_preprocessing_skips_conversion_when_disabled(self):
        kb_mock = MagicMock()
        pipe = PreprocessingPipeline(kb_repo=kb_mock, convert_units=False)

        sent = Sentence(original_text="The tower was 80 feet high.")
        p = Paragraph(sentences=[sent])
        chap = Chapter(title="Chapter 1", order_index=0, paragraphs=[p])
        book = Book(title="Test Adventure", chapters=[chap], source_language="en", target_language="uk")

        pipe.process(book)

        assert sent.original_text == "The tower was 80 feet high."

    def test_preprocessing_pipeline_convert_units_enabled_by_default_from_settings(self):
        from src.config.settings import settings
        from src.config.app_context import ApplicationContainer as ConfigContainer
        from src.launcher.app_context import ApplicationContainer as LauncherContainer

        assert settings.convert_units is True

        for ContainerClass in (ConfigContainer, LauncherContainer):
            container = ContainerClass()
            container.kb_repository.override(MagicMock())
            pipe = container.preprocessing_pipeline()
            assert pipe.convert_units is True

    def test_cli_no_unit_conversion_flag_disables_setting(self):
        from unittest.mock import patch
        from src.config.settings import settings
        from src.launcher.cli import main

        original_val = settings.convert_units
        try:
            test_args = ["cli.py", "--file", "nonexistent.txt", "--out", "out.txt", "--no-unit-conversion"]
            with patch("sys.argv", test_args):
                with patch("pathlib.Path.exists", return_value=False):
                    with pytest.raises(SystemExit):
                        main()
            assert settings.convert_units is False
        finally:
            settings.convert_units = original_val


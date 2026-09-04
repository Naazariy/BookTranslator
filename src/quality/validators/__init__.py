"""
src/quality/validators/__init__.py

Registry of modular quality assurance validators for BookTranslator V2.
"""
from src.quality.validators.empty_translation import EmptyTranslationValidator
from src.quality.validators.completeness import CompletenessValidator
from src.quality.validators.entity_consistency import EntityConsistencyValidator
from src.quality.validators.glossary_consistency import GlossaryConsistencyValidator
from src.quality.validators.gender_agreement import GenderAgreementValidator
from src.quality.validators.numbers_and_units import NumbersAndUnitsValidator
from src.quality.validators.control_tokens import ControlTokenValidator
from src.quality.validators.dialogue_integrity import DialogueIntegrityValidator

DEFAULT_VALIDATORS = (
    EmptyTranslationValidator,
    ControlTokenValidator,
    CompletenessValidator,
    EntityConsistencyValidator,
    GlossaryConsistencyValidator,
    GenderAgreementValidator,
    NumbersAndUnitsValidator,
    DialogueIntegrityValidator,
)

__all__ = [
    "EmptyTranslationValidator",
    "CompletenessValidator",
    "EntityConsistencyValidator",
    "GlossaryConsistencyValidator",
    "GenderAgreementValidator",
    "NumbersAndUnitsValidator",
    "ControlTokenValidator",
    "DialogueIntegrityValidator",
    "DEFAULT_VALIDATORS",
]

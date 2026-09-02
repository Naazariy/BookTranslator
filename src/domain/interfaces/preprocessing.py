from typing import Protocol, List
from src.domain.models.document import Book
from src.domain.models.knowledge import RawEntity, RawTerm, CharacterEntity


class IEntityExtractor(Protocol):
    def extract_entities(self, book: Book) -> List[RawEntity]:
        ...


class ITermExtractor(Protocol):
    def extract_terms(self, book: Book) -> List[RawTerm]:
        ...


class IAliasResolver(Protocol):
    def resolve_aliases(self, entities: List[RawEntity]) -> List[CharacterEntity]:
        ...

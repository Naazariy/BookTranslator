from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from uuid import UUID, uuid4


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    ORGANIZATION = "organization"
    ITEM = "item"
    MAGIC = "magic"
    TITLE = "title"
    TERM = "term"


class Entity(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: EntityType
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    canonical_translation: Optional[str] = None
    frequency: int = 1


class GlossaryItem(BaseModel):
    source_term: str
    target_term: str
    entity_type: EntityType = EntityType.TERM
    case_sensitive: bool = True
    reviewed: bool = False
    grammatical_gender: Optional[str] = None


class RawEntity(BaseModel):
    name: str
    entity_type: EntityType
    context: Optional[str] = None


class RawTerm(BaseModel):
    term: str
    frequency: int


class CharacterEntity(Entity):
    pass

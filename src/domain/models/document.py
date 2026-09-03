from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import UUID, uuid4


class InlineTagMap(BaseModel):
    """Карта інлайн-форматування (тегів) для окремого речення.
    Плейсхолдери типу <tag_1>word</tag_1> зберігаються для підстановки у LLM.
    """
    tag_id: str
    original_html: str
    closing_html: str


class Sentence(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    original_text: str
    translated_text: Optional[str] = None
    order_index: int = 0
    tag_map: Dict[str, InlineTagMap] = Field(default_factory=dict)


class Paragraph(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sentences: List[Sentence] = Field(default_factory=list)
    is_dialogue: bool = False
    style_class: Optional[str] = None


class Chapter(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    translated_title: Optional[str] = None
    paragraphs: List[Paragraph] = Field(default_factory=list)
    order_index: int


class Book(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    author: Optional[str] = None
    chapters: List[Chapter] = Field(default_factory=list)
    source_language: str = "en"
    target_language: str = "uk"
    metadata: Dict[str, str] = Field(default_factory=dict)

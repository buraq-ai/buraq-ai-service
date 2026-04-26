from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str
    language: str = "en"  # Supported: "en", "fr", "ar"


class SourceChunk(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    page_number: int
    excerpt: str  # First 200 characters of the chunk


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    has_answer: bool
    confidence_score: float
    language_detected: str = "en"   # Language of the generated answer
    should_create_ticket: bool = False  # True when no answer found — triggers ticket creation in BURAQ-27
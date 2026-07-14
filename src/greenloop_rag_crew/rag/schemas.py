"""Typed data models for page extraction and deterministic chunking."""

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Text and metadata extracted from one PDF page."""

    source: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    section: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class DocumentChunk(BaseModel):
    """A page-bounded chunk prepared for future embedding."""

    source: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    section: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    embedding_text: str = Field(..., min_length=1)
    token_count: int = Field(..., ge=1)


class DenseSearchResult(BaseModel):
    """One standalone dense retrieval result."""

    rank: int = Field(..., ge=1)
    chunk_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    section: str = Field(..., min_length=1)
    distance: float = Field(..., ge=0)
    score: float
    text: str = Field(..., min_length=1)


class BM25SearchResult(BaseModel):
    """One lexical BM25 retrieval result."""

    rank: int = Field(..., ge=1)
    chunk_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    section: str = Field(..., min_length=1)
    bm25_score: float
    text: str = Field(..., min_length=1)


class HybridSearchResult(BaseModel):
    """One weighted reciprocal-rank-fusion retrieval result."""

    rank: int = Field(..., ge=1)
    chunk_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    section: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    fusion_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    matched_by: list[str]

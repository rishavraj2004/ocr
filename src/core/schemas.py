"""Data models and schemas for the research prototype.

All pipeline stages communicate using these strictly validated Pydantic models
to ensure reproducible serialization, type safety, and schema integrity.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"


class DocumentVariant(str, Enum):
    GROUND_TRUTH = "ground_truth"
    RAW_OCR = "raw_ocr"
    CORRECTED_OCR = "corrected_ocr"


class QuestionType(str, Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    ENTITY = "entity"
    MULTI_SENTENCE = "multi_sentence"
    CONTEXTUAL = "contextual"


class DocumentMetadata(BaseModel):
    """Metadata describing a single scanned document page."""
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(..., description="Unique identifier for the parent document (e.g., 'doc_001')")
    page_id: str = Field(..., description="Unique page ID (e.g., 'doc_001_page_001')")
    language: Language = Field(..., description="Language of the document ('en' or 'hi')")
    page_number: int = Field(default=1, description="Page index within parent document")
    doc_type: str = Field(default="report", description="Document genre: government_report, academic, news, etc.")
    layout_type: str = Field(default="single_column", description="Layout: single_column, multi_column, table_heavy")
    image_quality: str = Field(default="standard", description="Subjective quality: clean, degraded, noisy")
    image_path: str = Field(..., description="Relative or absolute path to page image")
    ground_truth_path: str = Field(..., description="Path to verified ground truth text file")


class WordOCR(BaseModel):
    """Word-level OCR token with bounding box and confidence score."""
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., description="Extracted word string")
    confidence: float = Field(..., ge=0.0, le=100.0, description="OCR engine confidence score in [0.0, 100.0]")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="Bounding box [x, y, w, h]")
    line_num: Optional[int] = Field(default=None, description="Line number within page")
    word_num: Optional[int] = Field(default=None, description="Word sequence number on line/page")


class PageOCRResult(BaseModel):
    """Complete structured output produced by an OCR engine for one page."""
    model_config = ConfigDict(extra="ignore")

    page_id: str = Field(..., description="Page ID matching DocumentMetadata")
    document_id: str = Field(..., description="Parent document ID")
    language: Language = Field(..., description="Language detected or specified")
    engine: str = Field(..., description="OCR engine name, e.g. 'tesseract', 'paddleocr'")
    engine_version: str = Field(default="unknown", description="Version of the OCR engine")
    preprocessing_config: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing operations applied")
    raw_text: str = Field(..., description="Reconstructed raw full text")
    words: List[WordOCR] = Field(default_factory=list, description="Word tokens with confidence and coordinates")
    execution_time_sec: float = Field(default=0.0, ge=0.0, description="OCR processing time in seconds")


class CorrectionSpan(BaseModel):
    """A low-confidence span flagged for selective correction."""
    model_config = ConfigDict(extra="ignore")

    span_id: str = Field(..., description="Unique span identifier (e.g. 'span_001')")
    page_id: str = Field(..., description="Page ID containing the span")
    start_word_idx: int = Field(..., ge=0, description="Start index in page word list (inclusive)")
    end_word_idx: int = Field(..., ge=0, description="End index in page word list (exclusive)")
    original_text: str = Field(..., description="Original OCR text in this span")
    avg_confidence: float = Field(..., ge=0.0, le=100.0, description="Average confidence of words in span")
    min_confidence: float = Field(..., ge=0.0, le=100.0, description="Minimum word confidence in span")
    context_before: str = Field(default="", description="Surrounding words preceding the span")
    context_after: str = Field(default="", description="Surrounding words succeeding the span")
    corrected_text: str = Field(..., description="Corrected text returned by correction model")
    changed: bool = Field(default=False, description="Whether the correction model altered the span text")
    latency_sec: float = Field(default=0.0, ge=0.0, description="Latency of correction model query")
    prompt_tokens: Optional[int] = Field(default=None, description="Input tokens used if LLM-based")
    completion_tokens: Optional[int] = Field(default=None, description="Output tokens used if LLM-based")


class PageCorrectionResult(BaseModel):
    """Result of confidence-guided selective correction for one page."""
    model_config = ConfigDict(extra="ignore")

    page_id: str = Field(..., description="Page ID matching DocumentMetadata")
    original_text: str = Field(..., description="Original raw OCR text")
    corrected_text: str = Field(..., description="Full text reconstructed after selective span correction")
    spans: List[CorrectionSpan] = Field(default_factory=list, description="All identified and processed spans")
    threshold: float = Field(..., ge=0.0, le=100.0, description="Confidence threshold applied")
    total_words: int = Field(..., ge=0, description="Total word count in raw OCR")
    corrected_words: int = Field(default=0, ge=0, description="Count of words inside flagged low-confidence spans")
    total_latency_sec: float = Field(default=0.0, ge=0.0, description="Total wall-clock correction latency")


class Question(BaseModel):
    """Evaluation question independently created from verified Ground Truth."""
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(..., description="Unique question identifier (e.g. 'q_doc001_01')")
    document_id: str = Field(..., description="Associated document ID")
    page_id: str = Field(..., description="Associated ground truth page ID")
    language: Language = Field(..., description="Language ('en' or 'hi')")
    question: str = Field(..., description="Evaluation question text")
    expected_answer: str = Field(..., description="Verified expected answer text")
    source_page: int = Field(default=1, description="Page number containing the answer")
    relevant_chunk_ids: List[str] = Field(default_factory=list, description="Ground truth chunk IDs if pre-computed")
    question_type: QuestionType = Field(default=QuestionType.FACTUAL, description="Taxonomy classification")


class TextChunk(BaseModel):
    """A single segment of text produced by chunking a document variant."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Unique chunk ID (e.g. 'c_doc001_page001_0')")
    document_id: str = Field(..., description="Parent document ID")
    page_id: str = Field(..., description="Parent page ID")
    variant: DocumentVariant = Field(..., description="Variant from which chunk was formed")
    text: str = Field(..., description="Text payload of the chunk")
    chunk_index: int = Field(..., ge=0, description="Sequential index of chunk in document/page")
    start_char: int = Field(default=0, ge=0, description="Start character offset")
    end_char: int = Field(default=0, ge=0, description="End character offset")


class RetrievalItem(BaseModel):
    """Single item retrieved from vector store."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    page_id: str = Field(..., description="Page ID")
    score: float = Field(..., description="Similarity score or distance")
    rank: int = Field(..., ge=1, description="1-based retrieval rank")
    text: str = Field(..., description="Chunk text content")


class QuestionRetrievalResult(BaseModel):
    """Retrieval evaluation metrics for a single question under one variant."""
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(..., description="Question ID")
    variant: DocumentVariant = Field(..., description="Document variant evaluated")
    retrieved_chunks: List[RetrievalItem] = Field(default_factory=list, description="Top-K retrieved chunks")
    recall_at_1: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_at_3: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_at_5: float = Field(default=0.0, ge=0.0, le=1.0)
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank for this question")
    retrieval_latency_sec: float = Field(default=0.0, ge=0.0)


class QuestionAnswerResult(BaseModel):
    """End-to-end question answering evaluation for a single question under one variant."""
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(..., description="Question ID")
    variant: DocumentVariant = Field(..., description="Document variant evaluated")
    generated_answer: str = Field(..., description="Answer produced by LLM generator")
    expected_answer: str = Field(..., description="Verified expected answer")
    exact_match: float = Field(default=0.0, ge=0.0, le=1.0, description="Exact match score (0.0 or 1.0)")
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Token-level F1 score in [0.0, 1.0]")
    generation_latency_sec: float = Field(default=0.0, ge=0.0)
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)


class PageEvaluationSummary(BaseModel):
    """OCR evaluation metrics for one page."""
    model_config = ConfigDict(extra="ignore")

    page_id: str = Field(...)
    language: Language = Field(...)
    variant: DocumentVariant = Field(...)
    cer: float = Field(..., ge=0.0, description="Character Error Rate")
    wer: float = Field(..., ge=0.0, description="Word Error Rate")


class ExperimentSummary(BaseModel):
    """Consolidated summary metrics for an entire experiment run variant."""
    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(...)
    variant: DocumentVariant = Field(...)
    language: str = Field(..., description="'en', 'hi', or 'combined'")
    num_pages: int = Field(default=0, ge=0)
    num_questions: int = Field(default=0, ge=0)
    mean_cer: Optional[float] = Field(default=None)
    mean_wer: Optional[float] = Field(default=None)
    recall_at_1: float = Field(default=0.0)
    recall_at_3: float = Field(default=0.0)
    recall_at_5: float = Field(default=0.0)
    mrr: float = Field(default=0.0)
    mean_exact_match: float = Field(default=0.0)
    mean_f1: float = Field(default=0.0)
    total_latency_sec: float = Field(default=0.0)
    total_tokens: int = Field(default=0)

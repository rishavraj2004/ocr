"""OCR evaluation metrics: Character Error Rate (CER) and Word Error Rate (WER).

Computes deterministic edit distance metrics against verified Ground Truth
with explicit, configurable Unicode NFC and tokenization rules.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple
import Levenshtein
from pydantic import BaseModel, Field

from src.core.schemas import DocumentMetadata, DocumentVariant, Language, PageEvaluationSummary
from src.core.logging import get_logger

logger = get_logger("evaluation.ocr")


def normalize_text_for_eval(
    text: str,
    nfc: bool = True,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Explicit, auditable text normalization for OCR evaluation."""
    if not text:
        return ""

    out = text

    # 1. Unicode NFC Canonical Composition (Essential for Devanagari matras/conjuncts)
    if nfc:
        out = unicodedata.normalize("NFC", out)

    # 2. Case normalization (optional, Latin script)
    if ignore_case:
        out = out.lower()

    # 3. Punctuation removal (optional)
    if ignore_punctuation:
        # Standard ASCII punctuation plus Devanagari danda and double danda
        punct_chars = string.punctuation + "।" + "॥" + "•" + "—" + "–"
        out = out.translate(str.maketrans("", "", punct_chars))

    # 4. Collapse consecutive whitespace and strip margins
    if collapse_whitespace:
        out = re.sub(r"\s+", " ", out).strip()

    return out


def calculate_cer(
    reference: str,
    hypothesis: str,
    nfc: bool = True,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
) -> float:
    """Calculate Character Error Rate (CER) = Levenshtein(ref, hyp) / len(ref)."""
    ref_norm = normalize_text_for_eval(
        reference, nfc=nfc, ignore_case=ignore_case, ignore_punctuation=ignore_punctuation
    )
    hyp_norm = normalize_text_for_eval(
        hypothesis, nfc=nfc, ignore_case=ignore_case, ignore_punctuation=ignore_punctuation
    )

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    edit_dist = Levenshtein.distance(ref_norm, hyp_norm)
    return float(edit_dist / len(ref_norm))


def calculate_wer(
    reference: str,
    hypothesis: str,
    nfc: bool = True,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
) -> float:
    """Calculate Word Error Rate (WER) using word-token Levenshtein alignment."""
    ref_norm = normalize_text_for_eval(
        reference, nfc=nfc, ignore_case=ignore_case, ignore_punctuation=ignore_punctuation
    )
    hyp_norm = normalize_text_for_eval(
        hypothesis, nfc=nfc, ignore_case=ignore_case, ignore_punctuation=ignore_punctuation
    )

    ref_words = ref_norm.split(" ") if ref_norm else []
    hyp_words = hyp_norm.split(" ") if hyp_norm else []

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Calculate Levenshtein distance over list of word tokens
    edit_dist = Levenshtein.distance(ref_words, hyp_words)
    return float(edit_dist / len(ref_words))


class OCREvaluationAggregate(BaseModel):
    """Aggregate OCR performance statistics across a document collection."""
    language: str
    variant: DocumentVariant
    num_pages: int
    mean_cer: float
    std_cer: float
    median_cer: float
    mean_wer: float
    std_wer: float
    median_wer: float
    per_page_results: List[PageEvaluationSummary] = Field(default_factory=list)

    def summary_table_str(self) -> str:
        """Format an ASCII table summarizing OCR metrics."""
        lines = [
            f"=== OCR EVALUATION SUMMARY [{self.language.upper()} - {self.variant.value.upper()}] ===",
            f"Evaluated Pages : {self.num_pages}",
            f"Mean CER        : {self.mean_cer:.4f} (std: {self.std_cer:.4f}, median: {self.median_cer:.4f})",
            f"Mean WER        : {self.mean_wer:.4f} (std: {self.std_wer:.4f}, median: {self.median_wer:.4f})",
            "-" * 60,
            f"{'Page ID':25s} | {'CER':8s} | {'WER':8s}",
            "-" * 60,
        ]
        for p in self.per_page_results:
            lines.append(f"{p.page_id:25s} | {p.cer:8.4f} | {p.wer:8.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_ocr_page(
    page_id: str,
    language: Language,
    variant: DocumentVariant,
    ground_truth: str,
    ocr_text: str,
    nfc: bool = True,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
) -> PageEvaluationSummary:
    """Compute CER and WER for a single document page against verified ground truth."""
    cer = calculate_cer(
        ground_truth,
        ocr_text,
        nfc=nfc,
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
    )
    wer = calculate_wer(
        ground_truth,
        ocr_text,
        nfc=nfc,
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
    )

    return PageEvaluationSummary(
        page_id=page_id,
        language=language,
        variant=variant,
        cer=round(cer, 4),
        wer=round(wer, 4),
    )


def compute_aggregate_ocr_metrics(
    page_summaries: List[PageEvaluationSummary],
    language: str,
    variant: DocumentVariant,
) -> OCREvaluationAggregate:
    """Aggregate per-page summaries into statistical metrics (mean, std, median)."""
    import numpy as np

    if not page_summaries:
        return OCREvaluationAggregate(
            language=language,
            variant=variant,
            num_pages=0,
            mean_cer=0.0,
            std_cer=0.0,
            median_cer=0.0,
            mean_wer=0.0,
            std_wer=0.0,
            median_wer=0.0,
            per_page_results=[],
        )

    cers = [p.cer for p in page_summaries]
    wers = [p.wer for p in page_summaries]

    return OCREvaluationAggregate(
        language=language,
        variant=variant,
        num_pages=len(page_summaries),
        mean_cer=float(np.mean(cers)),
        std_cer=float(np.std(cers)),
        median_cer=float(np.median(cers)),
        mean_wer=float(np.mean(wers)),
        std_wer=float(np.std(wers)),
        median_wer=float(np.median(wers)),
        per_page_results=page_summaries,
    )

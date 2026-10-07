"""Comprehensive dataset integrity validator.

Verifies structural consistency, schema conformance, file presence,
and referential integrity between document images, ground-truth transcripts,
metadata records, and evaluation questions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from PIL import Image
from pydantic import BaseModel, Field

from src.core.schemas import DocumentMetadata, Question, Language
from src.core.logging import get_logger

logger = get_logger("dataset.validator")


class ValidationReport(BaseModel):
    """Structured report detailing dataset consistency and statistics."""
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    total_pages: int = 0
    pages_by_language: Dict[str, int] = Field(default_factory=dict)
    total_questions: int = 0
    questions_by_language: Dict[str, int] = Field(default_factory=dict)
    questions_by_type: Dict[str, int] = Field(default_factory=dict)

    def summary_str(self) -> str:
        """Format an ASCII report for terminal and experiment log output."""
        status = "PASSED (VALID)" if self.is_valid else "FAILED (INVALID)"
        lines = [
            "=" * 60,
            f"DATASET VALIDATION REPORT: {status}",
            "=" * 60,
            f"Total Pages Validated   : {self.total_pages}",
            f"Pages by Language       : {self.pages_by_language}",
            f"Total Questions         : {self.total_questions}",
            f"Questions by Language   : {self.questions_by_language}",
            f"Questions by Type       : {self.questions_by_type}",
            "-" * 60,
            f"Errors Encountered ({len(self.errors)}):",
        ]
        if self.errors:
            for idx, err in enumerate(self.errors, 1):
                lines.append(f"  [ERROR {idx}] {err}")
        else:
            lines.append("  None")

        lines.append(f"Warnings ({len(self.warnings)}):")
        if self.warnings:
            for idx, w in enumerate(self.warnings, 1):
                lines.append(f"  [WARN {idx}] {w}")
        else:
            lines.append("  None")

        lines.append("=" * 60)
        return "\n".join(lines)


class DatasetValidator:
    """Validates physical files, schema correctness, and cross-references."""

    def __init__(
        self,
        metadata_path: str | Path,
        images_dir: str | Path = "data/images",
        ground_truth_dir: str | Path = "data/ground_truth",
        questions_path: Optional[str | Path] = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.questions_path = Path(questions_path) if questions_path else Path("data/questions/questions.json")

    def validate(self) -> ValidationReport:
        """Run full validation suite and compile report."""
        report = ValidationReport()

        # 1. Validate Metadata file existence
        if not self.metadata_path.exists():
            report.errors.append(f"Metadata file not found at: {self.metadata_path}")
            report.is_valid = False
            return report

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
        except Exception as e:
            report.errors.append(f"Failed to parse metadata JSON from {self.metadata_path}: {e}")
            report.is_valid = False
            return report

        if not isinstance(raw_metadata, list):
            report.errors.append(f"Metadata root must be a JSON array, got {type(raw_metadata).__name__}")
            report.is_valid = False
            return report

        # 2. Validate Document Pages and Physical Files
        seen_page_ids: Set[str] = set()
        page_id_to_meta: Dict[str, DocumentMetadata] = {}

        for idx, item in enumerate(raw_metadata):
            try:
                meta = DocumentMetadata.model_validate(item)
            except Exception as e:
                report.errors.append(f"Item #{idx} failed DocumentMetadata schema validation: {e}")
                report.is_valid = False
                continue

            # Duplicate page ID check
            if meta.page_id in seen_page_ids:
                report.errors.append(f"Duplicate page_id detected: '{meta.page_id}' at index {idx}")
                report.is_valid = False
            else:
                seen_page_ids.add(meta.page_id)
                page_id_to_meta[meta.page_id] = meta

            # Physical Image Check
            img_path = Path(meta.image_path)
            if not img_path.is_absolute():
                img_path = self.images_dir.parent.parent / img_path if not img_path.exists() else img_path

            if not img_path.exists():
                report.errors.append(f"Page '{meta.page_id}': Referenced image not found at '{meta.image_path}'")
                report.is_valid = False
            else:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    report.errors.append(f"Page '{meta.page_id}': Image at '{img_path}' is corrupted: {e}")
                    report.is_valid = False

            # Physical Ground Truth Text Check
            gt_path = Path(meta.ground_truth_path)
            if not gt_path.is_absolute():
                gt_path = self.ground_truth_dir.parent.parent / gt_path if not gt_path.exists() else gt_path

            if not gt_path.exists():
                report.errors.append(f"Page '{meta.page_id}': Ground truth file not found at '{meta.ground_truth_path}'")
                report.is_valid = False
            else:
                try:
                    content = gt_path.read_text(encoding="utf-8").strip()
                    if not content:
                        report.warnings.append(f"Page '{meta.page_id}': Ground truth text file is empty")
                except UnicodeDecodeError:
                    report.errors.append(f"Page '{meta.page_id}': Ground truth file is not valid UTF-8: '{gt_path}'")
                    report.is_valid = False

            # Track counts
            lang_code = meta.language.value
            report.pages_by_language[lang_code] = report.pages_by_language.get(lang_code, 0) + 1

        report.total_pages = len(page_id_to_meta)

        # 3. Validate Questions File
        if not self.questions_path.exists():
            report.warnings.append(f"Questions file not found at '{self.questions_path}'")
        else:
            try:
                with open(self.questions_path, "r", encoding="utf-8") as f:
                    raw_questions = json.load(f)
            except Exception as e:
                report.errors.append(f"Failed to parse questions JSON from {self.questions_path}: {e}")
                report.is_valid = False
                raw_questions = []

            if not isinstance(raw_questions, list):
                report.errors.append(f"Questions root must be a JSON array, got {type(raw_questions).__name__}")
                report.is_valid = False
                raw_questions = []

            seen_question_ids: Set[str] = set()

            for idx, q_item in enumerate(raw_questions):
                try:
                    q = Question.model_validate(q_item)
                except Exception as e:
                    report.errors.append(f"Question #{idx} failed Question schema validation: {e}")
                    report.is_valid = False
                    continue

                # Unique question ID check
                if q.question_id in seen_question_ids:
                    report.errors.append(f"Duplicate question_id detected: '{q.question_id}' at index {idx}")
                    report.is_valid = False
                else:
                    seen_question_ids.add(q.question_id)

                # Referential integrity check: does page_id exist?
                if q.page_id not in page_id_to_meta:
                    report.errors.append(
                        f"Question '{q.question_id}' references unknown page_id: '{q.page_id}'"
                    )
                    report.is_valid = False
                else:
                    target_meta = page_id_to_meta[q.page_id]
                    # Document ID consistency check
                    if q.document_id != target_meta.document_id:
                        report.errors.append(
                            f"Question '{q.question_id}' specifies document_id '{q.document_id}', "
                            f"but page '{q.page_id}' belongs to document '{target_meta.document_id}'"
                        )
                        report.is_valid = False

                    # Language consistency check
                    if q.language != target_meta.language:
                        report.errors.append(
                            f"Question '{q.question_id}' language '{q.language}' does not match "
                            f"page language '{target_meta.language}' for page '{q.page_id}'"
                        )
                        report.is_valid = False

                # Content sanity
                if not q.question.strip():
                    report.errors.append(f"Question '{q.question_id}' has empty question text")
                    report.is_valid = False
                if not q.expected_answer.strip():
                    report.errors.append(f"Question '{q.question_id}' has empty expected_answer")
                    report.is_valid = False

                # Breakdown stats
                lang_code = q.language.value
                report.questions_by_language[lang_code] = report.questions_by_language.get(lang_code, 0) + 1
                q_type = q.question_type.value
                report.questions_by_type[q_type] = report.questions_by_type.get(q_type, 0) + 1

            report.total_questions = len(seen_question_ids)

        if report.errors:
            report.is_valid = False

        return report

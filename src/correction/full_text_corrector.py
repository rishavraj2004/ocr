"""Full-Text OCR Corrector (Naive Baseline).

Processes entire document pages without confidence masking, representing
the standard naive approach of feeding unconstrained raw OCR text into an LLM.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from src.core.config import CorrectionConfig
from src.core.schemas import Language
from src.correction.corrector import BaseCorrector, MockCorrector

logger = logging.getLogger(__name__)

FULL_TEXT_SYSTEM_PROMPT = """You are an expert OCR post-processing assistant.
The following text was extracted by an OCR system and contains typographical, spelling, and character errors.
Correct all OCR errors to restore the original document text.
Preserve paragraph structure and do not alter words that are already correct.

Raw OCR Text:
{raw_text}

Corrected Text:"""


class FullPageCorrectionResult(BaseModel):
    """Result of unconstrained full-page OCR correction."""
    model_config = ConfigDict(extra="ignore")

    page_id: str
    language: Language
    raw_text: str
    corrected_text: str
    latency_sec: float = Field(default=0.0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    over_corrected_words: int = Field(
        default=0,
        description="Words originally matching ground truth that were erroneously altered by the corrector",
    )


class FullTextCorrector:
    """Corrects entire document pages in a single unconstrained model query."""

    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config or CorrectionConfig()
        self.model_type = self.config.model_type.lower()
        self.mock_corrector = MockCorrector()
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            try:
                import openai
                api_key = os.getenv(self.config.api_key_env)
                if not api_key:
                    raise ValueError(f"Environment variable '{self.config.api_key_env}' not set.")
                self._openai_client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                logger.error("Failed to initialize OpenAI client for full-text correction: %s", e)
                raise
        return self._openai_client

    def _compute_over_corrections(self, raw_text: str, corrected_text: str, ground_truth: str) -> int:
        """Count words that were correct in raw OCR but were corrupted by full-text correction."""
        raw_words = re.findall(r"\S+", raw_text)
        corr_words = re.findall(r"\S+", corrected_text)
        gt_words = re.findall(r"\S+", ground_truth)

        over_corrections = 0
        min_len = min(len(raw_words), len(corr_words), len(gt_words))
        for i in range(min_len):
            rw = raw_words[i]
            cw = corr_words[i]
            gw = gt_words[i]
            # If raw word was correct (matching GT), but corrected word is different from GT
            if rw == gw and cw != gw:
                over_corrections += 1

        return over_corrections

    def correct_page(
        self,
        page_id: str,
        raw_text: str,
        language: Language,
        ground_truth: Optional[str] = None,
    ) -> FullPageCorrectionResult:
        """Correct entire raw text page at once."""
        t0 = time.perf_counter()
        is_hindi = (language == Language.HINDI)

        if self.model_type == "openai":
            client = self._get_openai_client()
            prompt = FULL_TEXT_SYSTEM_PROMPT.format(raw_text=raw_text)
            response = client.chat.completions.create(
                model=self.config.model_name,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.perf_counter() - t0
            corrected_text = response.choices[0].message.content.strip()
            p_tokens = response.usage.prompt_tokens if response.usage else max(1, len(prompt) // 4)
            c_tokens = response.usage.completion_tokens if response.usage else max(1, len(corrected_text) // 4)
        else:
            # Deterministic simulation of full-text correction:
            # Applies rules across every word in the document page indiscriminately
            words = raw_text.split()
            corrected_words = []
            for w in words:
                cw = self.mock_corrector._apply_heuristic_rules(w, is_hindi=is_hindi)
                corrected_words.append(cw)
            corrected_text = " ".join(corrected_words)

            latency = time.perf_counter() - t0
            # Simulating LLM tokens for full-text processing
            prompt = FULL_TEXT_SYSTEM_PROMPT.format(raw_text=raw_text)
            p_tokens = max(1, len(prompt) // 4)
            c_tokens = max(1, len(corrected_text) // 4)

        over_corr = 0
        if ground_truth:
            over_corr = self._compute_over_corrections(raw_text, corrected_text, ground_truth)

        return FullPageCorrectionResult(
            page_id=page_id,
            language=language,
            raw_text=raw_text,
            corrected_text=corrected_text,
            latency_sec=latency,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            over_corrected_words=over_corr,
        )

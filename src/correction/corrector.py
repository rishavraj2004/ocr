"""Pluggable OCR correction backends (Mock, Rule-based, and OpenAI LLM)."""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from src.core.config import CorrectionConfig
from src.core.logging import get_logger
from src.core.schemas import CorrectionSpan, Language

logger = get_logger("correction.corrector")


class BaseCorrector(ABC):
    """Abstract interface for selective span correction models."""

    @abstractmethod
    def correct_span(
        self,
        span: CorrectionSpan,
        language: Language,
    ) -> Tuple[str, bool, float, Optional[int], Optional[int]]:
        """Correct a single low-confidence span in context.

        Returns:
            Tuple of (corrected_text, changed_flag, latency_sec, prompt_tokens, completion_tokens)
        """
        pass


class MockCorrector(BaseCorrector):
    """Deterministic rule-based corrector for testing, simulation, and offline experiments."""

    def __init__(self):
        # Known common OCR correction lookup tables for English and Hindi
        self.en_corrections = {
            "tronsition": "transition",
            "enorgy": "energy",
            "copacity": "capacity",
            "gigowatts": "gigawatts",
            "copital": "capital",
            "infrastructuro": "infrastructure",
            "photovoltak": "photovoltaic",
            "solor": "solar",
            "wostern": "western",
            "stotes": "states",
            "corridor": "corridor",
            "subsidies": "subsidies",
            "agronomk": "agronomic",
            "micro-irrigotion": "micro-irrigation",
            "formers": "farmers",
            "hectores": "hectares",
            "geostotionary": "geostationary",
            "spocecraft": "spacecraft",
            "transpondors": "transponders",
            "totol": "total",
            "torget": "target",
            "intermittoncy": "intermittency",
            "turbinos": "turbines",
            "nocolles": "nacelles",
            "corrosion-rosistant": "corrosion-resistant",
            "officio1": "official",
        }

        self.hi_corrections = {
            "सरकर": "सरकार",
            "सरचालित": "संचालित",
            "दरश": "देश",
            "८र": "८८",
            "समरहों": "समूहों",
            "रै": "है",
            "फी": "की",
            "रिए": "लिए",
            "मेशिन": "मिशन",
            "अवलोकफ": "अवलोकन",
            "समाजिक": "सामाजिक",
            "परिबरों": "परिवारों",
            "सशफत": "सशक्त",
            "वाणिज्याक": "वाणिज्यिक",
            "कृषि": "कृषि",
            "उद्यमियों": "उद्यमियों",
            "नियमित": "नियमित",
            "३र": "३५",
            "अनुदान": "अनुदान",
            "उत्सजर्न": "उत्सर्जन",
            "प्रीमियम": "प्रीमियम",
            "सर्वेक्षण": "सर्वेक्षण",
            "निपटान": "निपटान",
            "डीजीटल": "डिजिटल",
            "अभयारण्य": "अभयारण्य",
            "न्यायाधिकरण": "न्यायाधिकरण",
        }

    def _apply_heuristic_rules(self, text: str, is_hindi: bool) -> str:
        """Apply general OCR character substitution reversals."""
        out = text
        if is_hindi:
            # Revert common mock/tesseract confusions
            for bad, good in self.hi_corrections.items():
                if bad in out:
                    out = out.replace(bad, good)
            # General Hindi conjunct cleanup
            out = re.sub(r"([क-ह])र([क-ह])", r"\1्\2", out) if " " not in out else out
        else:
            for bad, good in self.en_corrections.items():
                if bad in out.lower():
                    # Preserve case
                    if out.isupper():
                        out = good.upper()
                    elif out.istitle():
                        out = good.title()
                    else:
                        out = good
            # Latin numeric-letter reversals inside words
            if re.search(r"[a-zA-Z]1[a-zA-Z]", out):
                out = re.sub(r"([a-zA-Z])1([a-zA-Z])", r"\g<1>l\g<2>", out)
            if re.search(r"[a-zA-Z]0[a-zA-Z]", out):
                out = re.sub(r"([a-zA-Z])0([a-zA-Z])", r"\g<1>o\g<2>", out)
            if "rn" in out and len(out) > 4:
                out = out.replace("rn", "m")
        return out

    def correct_span(
        self,
        span: CorrectionSpan,
        language: Language,
    ) -> Tuple[str, bool, float, Optional[int], Optional[int]]:
        """Perform deterministic simulated span correction."""
        start_t = time.perf_counter()
        is_hindi = (language == Language.HINDI)

        original = span.original_text
        corrected = self._apply_heuristic_rules(original, is_hindi)

        # Simulate small compute latency (~1-5ms)
        latency = time.perf_counter() - start_t
        changed = (corrected != original)

        # Simulated token counts based on character length
        est_tokens = max(1, len(original) // 4)
        return corrected, changed, round(latency, 4), est_tokens + 15, est_tokens


class OpenAICorrector(BaseCorrector):
    """Selective span corrector using OpenAI LLM API with strict frozen prompt."""

    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config or CorrectionConfig()
        self.api_key = os.getenv(self.config.api_key_env, None)
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                base_url = None
                model_type = self.config.model_type.lower()
                if model_type in ["gemini", "google"]:
                    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                elif model_type == "mistral":
                    base_url = "https://api.mistral.ai/v1"

                self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI/Gemini/Mistral client: {e}")

    def correct_span(
        self,
        span: CorrectionSpan,
        language: Language,
    ) -> Tuple[str, bool, float, Optional[int], Optional[int]]:
        """Query LLM to correct solely the uncertain OCR span."""
        if not self.client:
            logger.warning("OpenAI client not available. Falling back to MockCorrector.")
            return MockCorrector().correct_span(span, language)

        start_t = time.perf_counter()
        lang_name = "Hindi (Devanagari)" if language == Language.HINDI else "English"

        prompt = (
            f"You are a professional multilingual OCR proofreader and text correction specialist.\n"
            f"Language: {lang_name}\n\n"
            f"Surrounding Context:\n"
            f"... {span.context_before} [UNCERTAIN_OCR_SPAN: {span.original_text}] {span.context_after} ...\n\n"
            f"INSTRUCTION:\n"
            f"1. Correct ONLY the text inside [UNCERTAIN_OCR_SPAN: {span.original_text}].\n"
            f"2. Output ONLY the corrected replacement string. Do not repeat surrounding context or add notes.\n"
            f"3. If the span is already correct, output it unchanged."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise OCR span correction engine. You must output ONLY the replacement word/phrase without quotes, asterisks, markdown, or explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_output = response.choices[0].message.content or ""
            corrected = raw_output.strip().strip('"').strip("'").strip("`").replace("**", "").strip()
            latency = time.perf_counter() - start_t
            changed = (corrected != span.original_text)

            prompt_toks = response.usage.prompt_tokens if response.usage else None
            comp_toks = response.usage.completion_tokens if response.usage else None

            return corrected, changed, round(latency, 4), prompt_toks, comp_toks

        except Exception as e:
            logger.error(f"OpenAI API call failed for span '{span.span_id}': {e}")
            latency = time.perf_counter() - start_t
            return span.original_text, False, round(latency, 4), None, None

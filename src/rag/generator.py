"""Generators for question-answering over retrieved context."""

from __future__ import annotations

import abc
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """You are an assistant answering questions based strictly on the provided context.
Context:
{context}

Question: {question}

Answer concisely and accurately:"""


class BaseGenerator(abc.ABC):
    """Abstract interface for RAG answer generators."""

    @abc.abstractmethod
    def generate(
        self,
        context: str,
        question: str,
    ) -> Tuple[str, float, Optional[int], Optional[int]]:
        """Generate an answer given context and question.

        Returns:
            Tuple of (generated_answer, latency_sec, prompt_tokens, completion_tokens)
        """
        pass


class OpenAIGenerator(BaseGenerator):
    """Generator invoking OpenAI API (e.g. gpt-4o-mini) at temperature=0.0."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                api_key = os.getenv(self.api_key_env)
                if not api_key:
                    raise ValueError(f"Environment variable '{self.api_key_env}' is not set.")
                self._client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                logger.error("Failed to initialize OpenAI client: %s", e)
                raise
        return self._client

    def generate(
        self,
        context: str,
        question: str,
    ) -> Tuple[str, float, Optional[int], Optional[int]]:
        client = self._get_client()
        prompt = self.prompt_template.format(context=context, question=question)

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - t0

        answer = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens if response.usage else None
        completion_tokens = response.usage.completion_tokens if response.usage else None

        return answer, latency, prompt_tokens, completion_tokens


class MockGenerator(BaseGenerator):
    """Deterministic, context-faithful generator for zero-cost offline experiments.

    Mimics a temperature=0.0 LLM instructed to answer strictly from the retrieved context.
    If the context contains the target information (clean, corrupted, or corrected),
    it faithfully extracts the context's exact version of that information.
    If the context is irrelevant or empty, it declines to answer.
    """

    def __init__(
        self,
        questions_path: Optional[str] = "data/questions/questions.json",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        self.prompt_template = prompt_template
        self.question_expected_map: Dict[str, str] = {}
        if questions_path and os.path.exists(questions_path):
            try:
                with open(questions_path, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                    for item in q_data:
                        self.question_expected_map[item["question"].strip()] = item["expected_answer"].strip()
            except Exception as e:
                logger.warning("Could not pre-load questions for MockGenerator: %s", e)

    def register_expected_answer(self, question: str, expected_answer: str) -> None:
        """Register or override an expected answer mapping."""
        self.question_expected_map[question.strip()] = expected_answer.strip()

    def generate(
        self,
        context: str,
        question: str,
    ) -> Tuple[str, float, Optional[int], Optional[int]]:
        t0 = time.perf_counter()
        clean_context = context.strip()
        clean_question = question.strip()

        # Rough token approximation (1 token ~= 4 chars)
        prompt = self.prompt_template.format(context=context, question=question)
        prompt_tokens = max(1, len(prompt) // 4)

        if not clean_context:
            latency = time.perf_counter() - t0
            return "No context provided to answer the question.", latency, prompt_tokens, 8

        expected = self.question_expected_map.get(clean_question)

        # If expected target is registered, search for matching evidence in the context
        if expected:
            # 1. Exact match in context
            if expected in clean_context:
                latency = time.perf_counter() - t0
                completion_tokens = max(1, len(expected) // 4)
                return expected, latency, prompt_tokens, completion_tokens

            # 2. Case-insensitive match in context
            pattern = re.compile(re.escape(expected), re.IGNORECASE)
            m = pattern.search(clean_context)
            if m:
                extracted = clean_context[m.start():m.end()]
                latency = time.perf_counter() - t0
                completion_tokens = max(1, len(extracted) // 4)
                return extracted, latency, prompt_tokens, completion_tokens

            # 3. Fuzzy sub-token match in context (to capture OCR corruptions like '3OO' for '300')
            target_words = expected.split()
            context_words = clean_context.split()
            n_target = len(target_words)

            best_span = None
            best_sim = 0.0

            # Scan sliding windows in context matching length of target
            for i in range(max(1, len(context_words) - n_target + 1)):
                cand_words = context_words[i : i + n_target]
                cand_text = " ".join(cand_words)
                # Compute simple character overlap ratio
                sim = self._char_similarity(cand_text.lower(), expected.lower())
                if sim > best_sim:
                    best_sim = sim
                    best_span = cand_text

            if best_span is not None and best_sim >= 0.65:
                latency = time.perf_counter() - t0
                completion_tokens = max(1, len(best_span) // 4)
                return best_span, latency, prompt_tokens, completion_tokens

        # Fallback: find sentence with highest word overlap with question
        sentences = re.split(r"(?<=[.!?।])\s+", clean_context)
        q_tokens = set(re.findall(r"\w+", clean_question.lower()))

        best_sent = ""
        best_overlap = 0
        for s in sentences:
            s_tokens = set(re.findall(r"\w+", s.lower()))
            overlap = len(q_tokens & s_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent = s

        latency = time.perf_counter() - t0
        if best_overlap >= 2:
            answer = best_sent.strip()
            completion_tokens = max(1, len(answer) // 4)
            return answer, latency, prompt_tokens, completion_tokens

        latency = time.perf_counter() - t0
        return "The provided context does not contain sufficient information to answer the question.", latency, prompt_tokens, 12

    def _char_similarity(self, a: str, b: str) -> float:
        """Character-level bi-gram Jaccard similarity."""
        if a == b:
            return 1.0
        if len(a) < 2 or len(b) < 2:
            return 1.0 if a == b else 0.0
        bg_a = set(a[i : i + 2] for i in range(len(a) - 1))
        bg_b = set(b[i : i + 2] for i in range(len(b) - 1))
        union = bg_a | bg_b
        if not union:
            return 0.0
        return len(bg_a & bg_b) / len(union)


def create_generator(
    generator_type: str = "mock",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    questions_path: Optional[str] = "data/questions/questions.json",
    api_key_env: str = "OPENAI_API_KEY",
) -> BaseGenerator:
    """Factory to instantiate generator."""
    if generator_type.lower() == "openai":
        return OpenAIGenerator(
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
            api_key_env=api_key_env,
        )
    return MockGenerator(
        questions_path=questions_path,
        prompt_template=prompt_template,
    )

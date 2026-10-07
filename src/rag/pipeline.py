"""Universal RAG evaluation pipeline for Ground Truth, Raw OCR, and Corrected OCR variants."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.config import ExperimentConfig
from src.core.schemas import (
    DocumentMetadata,
    DocumentVariant,
    ExperimentSummary,
    Question,
    QuestionAnswerResult,
    QuestionRetrievalResult,
    QuestionType,
    TextChunk,
)
from src.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    compute_answer_metrics,
)
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_question,
)
from src.rag.chunker import SlidingWindowChunker
from src.rag.embeddings import BaseEmbeddingModel, create_embedding_model
from src.rag.generator import BaseGenerator, create_generator
from src.rag.retriever import DenseRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline parameterized strictly by DocumentVariant.

    Ensures identical chunking parameters, embedding models, vector indices,
    retrieval top-K, and generation prompts across all three variants.
    """

    def __init__(
        self,
        variant: DocumentVariant,
        config: ExperimentConfig,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        generator: Optional[BaseGenerator] = None,
        chunker: Optional[SlidingWindowChunker] = None,
    ):
        self.variant = variant
        self.config = config

        # 1. Chunker
        self.chunker = chunker or SlidingWindowChunker(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            unit=config.rag.chunk_unit,
        )

        # 2. Embedding Model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            use_mock = config.rag.generator_type.lower() == "mock" and config.rag.embedding_model.lower().startswith("mock")
            self.embedding_model = create_embedding_model(
                model_name=config.rag.embedding_model,
                dimension=config.rag.embedding_dim,
                use_mock=use_mock,
            )

        # 3. Retriever
        self.retriever = DenseRetriever(
            embedding_model=self.embedding_model,
            top_k=config.rag.top_k,
        )

        # 4. Generator
        self.generator = generator or create_generator(
            generator_type=config.rag.generator_type,
            model_name=config.rag.generator_model,
            temperature=config.rag.temperature,
            prompt_template=config.rag.prompt_template,
            questions_path=os.path.join(config.dataset.questions_dir, "questions.json"),
        )

        self.indexed_chunks: List[TextChunk] = []

    def load_variant_documents(self) -> Dict[str, Tuple[str, str]]:
        """Load text for this variant.

        Returns:
            Dict mapping page_id -> (document_id, text)
        """
        meta_path = self.config.dataset.metadata_path
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_list = [DocumentMetadata(**item) for item in json.load(f)]

        page_texts: Dict[str, Tuple[str, str]] = {}

        for meta in metadata_list:
            page_id = meta.page_id
            doc_id = meta.document_id

            if self.variant == DocumentVariant.GROUND_TRUTH:
                gt_path = os.path.join(self.config.dataset.ground_truth_dir, f"{page_id}.txt")
                if not os.path.exists(gt_path):
                    logger.warning("Ground truth file missing: %s", gt_path)
                    continue
                with open(gt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                page_texts[page_id] = (doc_id, text)

            elif self.variant == DocumentVariant.RAW_OCR:
                ocr_path = os.path.join(self.config.dataset.ocr_dir, f"{page_id}.json")
                if not os.path.exists(ocr_path):
                    logger.warning("Raw OCR file missing: %s", ocr_path)
                    continue
                with open(ocr_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)
                    text = ocr_data.get("raw_text") or ocr_data.get("text", "")
                page_texts[page_id] = (doc_id, text)

            elif self.variant == DocumentVariant.CORRECTED_OCR:
                corr_path = os.path.join(self.config.dataset.corrected_dir, f"{page_id}.json")
                if not os.path.exists(corr_path):
                    logger.warning("Corrected OCR file missing: %s", corr_path)
                    continue
                with open(corr_path, "r", encoding="utf-8") as f:
                    corr_data = json.load(f)
                    text = corr_data.get("corrected_text") or corr_data.get("raw_text") or corr_data.get("text", "")
                page_texts[page_id] = (doc_id, text)

        return page_texts

    def index_documents(self) -> List[TextChunk]:
        """Load, chunk, and index all documents for this variant."""
        page_texts = self.load_variant_documents()
        all_chunks: List[TextChunk] = []

        for page_id, (doc_id, text) in page_texts.items():
            chunks = self.chunker.chunk_document(
                text=text,
                document_id=doc_id,
                page_id=page_id,
                variant=self.variant,
            )
            all_chunks.extend(chunks)

        self.retriever.index_chunks(all_chunks)
        self.indexed_chunks = all_chunks
        return all_chunks

    def evaluate_question(
        self,
        question: Question,
    ) -> Tuple[QuestionRetrievalResult, QuestionAnswerResult]:
        """Evaluate a single question: retrieval + prompt assembly + answer generation + scoring."""
        # 1. Retrieval
        retrieval_res = self.retriever.evaluate_question(
            question=question,
            variant=self.variant,
            top_k=self.config.rag.top_k,
        )

        # 2. Context assembly
        context_parts = [
            f"[Passage {item.rank}]:\n{item.text}"
            for item in retrieval_res.retrieved_chunks
        ]
        context_str = "\n\n".join(context_parts)

        # 3. Answer Generation
        gen_answer, gen_latency, p_tokens, c_tokens = self.generator.generate(
            context=context_str,
            question=question.question,
        )

        # 4. Multilingual Answer Scoring
        em, f1 = compute_answer_metrics(
            prediction=gen_answer,
            ground_truth=question.expected_answer,
            language=question.language.value,
        )

        qa_res = QuestionAnswerResult(
            question_id=question.question_id,
            variant=self.variant,
            generated_answer=gen_answer,
            expected_answer=question.expected_answer,
            exact_match=em,
            f1_score=f1,
            generation_latency_sec=gen_latency,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
        )

        return retrieval_res, qa_res

    def run_evaluation(
        self,
        questions: Optional[List[Question]] = None,
    ) -> Dict[str, Any]:
        """Run complete retrieval and QA evaluation across all benchmark questions."""
        # Ensure documents are indexed
        if self.retriever.vector_store.count() == 0:
            self.index_documents()

        # Load questions if not provided
        if questions is None:
            q_path = os.path.join(self.config.dataset.questions_dir, "questions.json")
            with open(q_path, "r", encoding="utf-8") as f:
                questions = [Question(**item) for item in json.load(f)]

        logger.info(
            "Running RAG evaluation for variant '%s' on %d questions...",
            self.variant.value,
            len(questions),
        )

        retrieval_results: List[QuestionRetrievalResult] = []
        qa_results: List[QuestionAnswerResult] = []

        t0 = time.perf_counter()
        for q in questions:
            r_res, a_res = self.evaluate_question(q)
            retrieval_results.append(r_res)
            qa_results.append(a_res)
        total_eval_time = time.perf_counter() - t0

        # Aggregated overall metrics
        retrieval_agg = aggregate_retrieval_metrics(retrieval_results)
        qa_agg = aggregate_answer_metrics(qa_results)

        # Breakdown by language
        en_questions = [q for q in questions if q.language.value == "en"]
        hi_questions = [q for q in questions if q.language.value == "hi"]

        en_qa = [a for a, q in zip(qa_results, questions) if q.language.value == "en"]
        hi_qa = [a for a, q in zip(qa_results, questions) if q.language.value == "hi"]

        en_ret = [r for r, q in zip(retrieval_results, questions) if q.language.value == "en"]
        hi_ret = [r for r, q in zip(retrieval_results, questions) if q.language.value == "hi"]

        total_prompt_tokens = sum(r.prompt_tokens or 0 for r in qa_results)
        total_completion_tokens = sum(r.completion_tokens or 0 for r in qa_results)

        summary = ExperimentSummary(
            run_id=f"rag_{self.variant.value}_{int(time.time())}",
            variant=self.variant,
            language="combined",
            num_pages=len(set(c.page_id for c in self.indexed_chunks)),
            num_questions=len(questions),
            recall_at_1=retrieval_agg["mean_recall_at_1"],
            recall_at_3=retrieval_agg["mean_recall_at_3"],
            recall_at_5=retrieval_agg["mean_recall_at_5"],
            mrr=retrieval_agg["mean_mrr"],
            mean_exact_match=qa_agg["mean_exact_match"],
            mean_f1=qa_agg["mean_f1"],
            total_latency_sec=total_eval_time,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        report = {
            "variant": self.variant.value,
            "summary": summary.model_dump(),
            "overall": {
                **retrieval_agg,
                **qa_agg,
                "total_eval_latency_sec": total_eval_time,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
            },
            "by_language": {
                "en": {
                    "num_questions": len(en_questions),
                    **aggregate_retrieval_metrics(en_ret),
                    **aggregate_answer_metrics(en_qa),
                },
                "hi": {
                    "num_questions": len(hi_questions),
                    **aggregate_retrieval_metrics(hi_ret),
                    **aggregate_answer_metrics(hi_qa),
                },
            },
            "by_question_type": {
                q_type.value: {
                    "num_questions": len([q for q in questions if q.question_type == q_type]),
                    **aggregate_retrieval_metrics([r for r, q in zip(retrieval_results, questions) if q.question_type == q_type]),
                    **aggregate_answer_metrics([a for a, q in zip(qa_results, questions) if q.question_type == q_type]),
                }
                for q_type in [QuestionType.FACTUAL, QuestionType.NUMERICAL, QuestionType.ENTITY]
                if any(q.question_type == q_type for q in questions)
            },
            "retrieval_results": [r.model_dump() for r in retrieval_results],
            "qa_results": [q.model_dump() for q in qa_results],
        }

        # Save to results/rag/
        out_dir = os.path.join(self.config.output_dir, "rag")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{self.variant.value}_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Saved RAG evaluation report to %s", out_file)

        return report

"""FastAPI Application Server for Research Inspection Dashboard (Phase 12).

Serves interactive visual inspection endpoints for:
1. Executive KPIs and Triangulation metrics.
2. Document pages, images, and confidence-colored bounding boxes.
3. Selective correction audit trail and live threshold simulation.
4. 3-Variant RAG question-by-question comparative explorer.
5. Publication figures gallery and LaTeX booktabs table exports.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.core.config import AppConfig, PreprocessingConfig, load_config
from src.core.schemas import Language, WordOCR
from src.correction.confidence_detector import ConfidenceDetector
from src.correction.span_extractor import SpanExtractor
from src.correction.corrector import MockCorrector, OpenAICorrector
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.utils import load_image_as_cv2, save_cv2_image
from src.rag.pipeline import get_live_rag_service

logger = logging.getLogger("dashboard")

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


class LiveQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)
    expected_answer: Optional[str] = None
    language: str = "auto"
    page_id: Optional[str] = None


def generate_sample_questions_from_text(text: str, language: str = "en") -> List[Dict[str, str]]:
    """Derive 2-3 instant test questions from document text."""
    import re
    questions = []
    sentences = [s.strip() for s in re.split(r'[.!?।\n]+', text) if len(s.strip()) > 15]

    if language == "hi":
        for sent in sentences[:3]:
            words = sent.split()
            if len(words) >= 4:
                topic = " ".join(words[:4])
                questions.append({
                    "question": f"{topic} के संबंध में क्या जानकारी है?",
                    "expected_answer": sent[:100],
                })
        if not questions:
            questions.append({
                "question": "इस दस्तावेज़ का मुख्य सारांश क्या है?",
                "expected_answer": text[:100],
            })
    else:
        for sent in sentences[:3]:
            words = sent.split()
            nums = re.findall(r'\b\d+(?:[\.,]\d+)?\b', sent)
            if nums:
                prefix = f"{words[0]} {words[1]}" if len(words) > 1 else words[0]
                questions.append({
                    "question": f"What numerical value is reported for {prefix}?",
                    "expected_answer": nums[0],
                })
            else:
                topic = " ".join(words[:min(5, len(words))])
                questions.append({
                    "question": f"What is mentioned regarding {topic}?",
                    "expected_answer": sent[:100],
                })
        if not questions:
            questions.append({
                "question": "What is the primary topic of this document?",
                "expected_answer": text[:100],
            })
    return questions[:3]


class LiveCorrectionRequest(BaseModel):
    page_id: str
    threshold: float = Field(default=70.0, ge=0.0, le=100.0)
    context_window: int = Field(default=5, ge=1, le=15)
    model_type: str = "mock"


class LivePreprocessRequest(BaseModel):
    page_id: str
    denoise: bool = True
    denoise_h: float = 10.0
    deskew: bool = True
    threshold_method: str = "otsu"


class TextUploadRequest(BaseModel):
    title: str = Field(default="Uploaded Document")
    text: str = Field(..., min_length=5)
    language: str = Field(default="auto")
    threshold: float = Field(default=70.0, ge=0.0, le=100.0)


def handle_document_ingestion(
    title: str,
    text: Optional[str],
    image_bytes: Optional[bytes],
    filename: Optional[str],
    language: str = "auto",
    threshold: float = 70.0,
) -> Dict[str, Any]:
    """Process an uploaded document image or text file across the entire pipeline."""
    import time
    import re
    from PIL import Image, ImageDraw

    timestamp = int(time.time())
    clean_title = re.sub(r'[^a-zA-Z0-9_\u0900-\u097F-]', '_', title or "document")[:20]
    page_id = f"custom_{clean_title}_{timestamp}"
    doc_id = f"doc_{clean_title}_{timestamp}"

    # Determine language
    if language == "auto":
        probe_text = (text or "") + (filename or "")
        is_hindi = bool(re.search(r"[\u0900-\u097F]", probe_text))
        lang_str = "hi" if is_hindi else "en"
    else:
        lang_str = language
    lang_enum = Language.HINDI if lang_str == "hi" else Language.ENGLISH

    img_path = DATA_DIR / "images" / f"{page_id}.png"
    DATA_DIR.joinpath("images").mkdir(parents=True, exist_ok=True)
    DATA_DIR.joinpath("ground_truth").mkdir(parents=True, exist_ok=True)
    DATA_DIR.joinpath("ocr").mkdir(parents=True, exist_ok=True)
    DATA_DIR.joinpath("corrected").mkdir(parents=True, exist_ok=True)

    # 1. Image & Document handling
    is_pdf = (filename and filename.lower().endswith(".pdf")) or (image_bytes and image_bytes[:4] == b"%PDF")

    if is_pdf and image_bytes:
        import io
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(image_bytes))
            pdf_texts = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pdf_texts:
                text = "\n\n".join(pdf_texts).strip()
        except Exception as e:
            logger.warning("PDF text extraction failed: %s", e)

        canvas = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 40), f"PDF DOCUMENT: {title}", fill=(0, 0, 0))
        y = 80
        for line in (text or "").splitlines()[:35]:
            draw.text((40, y), line[:80], fill=(30, 30, 30))
            y += 24
        canvas.save(img_path)
    elif image_bytes:
        img_path.write_bytes(image_bytes)
    else:
        canvas = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 40), f"DOCUMENT: {title}", fill=(0, 0, 0))
        y = 80
        for line in (text or "").splitlines()[:35]:
            draw.text((40, y), line[:80], fill=(30, 30, 30))
            y += 24
        canvas.save(img_path)

    # 2. Text & OCR extraction
    gt_text = text or ""
    if not gt_text:
        from src.ocr.tesseract_ocr import TesseractOCREngine
        from src.core.config import OCRConfig
        tess = TesseractOCREngine(OCRConfig())
        if tess.is_available:
            ocr_res = tess.extract_page(img_path, page_id=page_id, document_id=doc_id, language=lang_enum)
            gt_text = ocr_res.raw_text
        else:
            mistral_key = os.getenv("MISTRAL_API_KEY")
            if mistral_key:
                try:
                    import base64
                    import openai
                    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    v_client = openai.OpenAI(api_key=mistral_key, base_url="https://api.mistral.ai/v1")
                    v_resp = v_client.chat.completions.create(
                        model="pixtral-12b-2409",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Transcribe all text from this document image exactly. Output only the extracted text."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                            ]
                        }],
                        temperature=0.0,
                    )
                    gt_text = v_resp.choices[0].message.content.strip()
                    logger.info("Transcribed %d chars from image using Pixtral Vision OCR", len(gt_text))
                except Exception as ve:
                    logger.warning("Pixtral vision transcription failed: %s", ve)

        if not gt_text:
            gt_text = f"Digitized document content for {title}."

    # Always generate realistic OCR degradation for Variant B and bounding boxes
    from src.ocr.mock_ocr import MockOCREngine
    mock_engine = MockOCREngine(error_rate=0.08)
    ocr_res = mock_engine.extract_page(
        image_input=img_path,
        page_id=page_id,
        document_id=doc_id,
        language=lang_enum,
        reference_text=gt_text,
    )
    raw_words = ocr_res.words
    raw_text = ocr_res.raw_text

    # Save Ground Truth and Raw OCR
    gt_file = DATA_DIR / "ground_truth" / f"{page_id}.txt"
    gt_file.write_text(gt_text, encoding="utf-8")

    ocr_file = DATA_DIR / "ocr" / f"{page_id}.json"
    ocr_file.write_text(ocr_res.model_dump_json(indent=2), encoding="utf-8")

    # 3. Confidence-Guided Selective Correction (Variant C)
    detector = ConfidenceDetector(threshold=threshold)
    candidates = detector.detect_low_confidence_spans(raw_words)
    extractor = SpanExtractor(context_window_words=5)
    spans = extractor.extract_contextual_spans(page_id, raw_words, candidates)

    corrector = MockCorrector()
    corrected_words = [w.text for w in raw_words]
    for s in spans:
        corr_text, changed, lat, p_toks, c_toks = corrector.correct_span(s, lang_enum)
        s.corrected_text = corr_text
        s.changed = changed
        s.latency_sec = lat
        s.prompt_tokens = p_toks
        s.completion_tokens = c_toks
        for idx in range(s.start_word_idx, min(s.end_word_idx, len(corrected_words))):
            corrected_words[idx] = corr_text

    corrected_text = " ".join(corrected_words)

    from src.core.schemas import PageCorrectionResult
    corr_res = PageCorrectionResult(
        page_id=page_id,
        original_text=raw_text,
        corrected_text=corrected_text,
        spans=spans,
        threshold=threshold,
        total_words=len(raw_words),
        corrected_words=sum(len(c.words) for c in candidates),
        total_latency_sec=sum(s.latency_sec for s in spans),
    )
    corr_file = DATA_DIR / "corrected" / f"{page_id}.json"
    corr_file.write_text(corr_res.model_dump_json(indent=2), encoding="utf-8")

    # 4. Generate document-specific sample questions and update custom_documents.json
    sample_questions = generate_sample_questions_from_text(gt_text or raw_text, lang_str)
    meta_file = DATA_DIR / "metadata" / "custom_documents.json"
    meta_list = load_json_safe(meta_file) or []
    new_meta = {
        "document_id": doc_id,
        "page_id": page_id,
        "language": lang_str,
        "page_number": 1,
        "doc_type": "user_uploaded",
        "layout_type": "single_column",
        "image_quality": "clean",
        "image_path": f"data/images/{page_id}.png",
        "ground_truth_path": f"data/ground_truth/{page_id}.txt",
        "title": title or page_id,
        "created_at": timestamp,
        "word_count": len(raw_words),
        "flagged_spans_count": len(spans),
        "sample_questions": sample_questions,
    }
    meta_list.append(new_meta)
    meta_file.write_text(json.dumps(meta_list, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. Index into FAISS vector store for live Q&A!
    service = get_live_rag_service()
    chunk_counts = service.index_custom_page(
        document_id=doc_id,
        page_id=page_id,
        text_gt=gt_text,
        text_raw=raw_text,
        text_corr=corrected_text,
    )

    return {
        "status": "success",
        "page_id": page_id,
        "document_id": doc_id,
        "title": title,
        "language": lang_str,
        "word_count": len(raw_words),
        "flagged_spans_count": len(spans),
        "flagged_words_count": sum(len(c.words) for c in candidates),
        "chunks_indexed": chunk_counts,
        "image_url": f"/images/{page_id}.png",
        "ground_truth_snippet": gt_text[:200] + ("..." if len(gt_text) > 200 else ""),
        "raw_ocr_snippet": raw_text[:200] + ("..." if len(raw_text) > 200 else ""),
        "corrected_snippet": corrected_text[:200] + ("..." if len(corrected_text) > 200 else ""),
        "sample_questions": sample_questions,
    }


def load_json_safe(path: Path) -> Optional[Any]:
    """Load JSON from path if exists, else return None."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None
    return None


def get_kpis_data() -> Dict[str, Any]:
    """Synthesize high-level benchmark KPIs from triangulation and sweeps."""
    comp_file = RESULTS_DIR / "rag" / "comparison_summary.json"
    comp_data = load_json_safe(comp_file) or {}

    sweep_file = RESULTS_DIR / "sweeps" / "threshold_sweep.json"
    sweep_data = load_json_safe(sweep_file) or {}

    eff_file = RESULTS_DIR / "sweeps" / "full_vs_selective.json"
    eff_data = load_json_safe(eff_file) or {}

    cl_file = RESULTS_DIR / "analysis" / "cross_lingual_study.json"
    cl_data = load_json_safe(cl_file) or {}

    overall_em = comp_data.get("overall", {}).get("Exact Match (EM)", {})
    overall_f1 = comp_data.get("overall", {}).get("Token F1 Score", {})
    overall_r5 = comp_data.get("overall", {}).get("Recall@5", {})
    overall_mrr = comp_data.get("overall", {}).get("MRR", {})

    en_em = comp_data.get("by_language", {}).get("en", {}).get("Exact Match (EM)", {})
    hi_em = comp_data.get("by_language", {}).get("hi", {}).get("Exact Match (EM)", {})

    return {
        "overall": {
            "exact_match": overall_em,
            "token_f1": overall_f1,
            "recall_at_5": overall_r5,
            "mrr": overall_mrr,
        },
        "by_language": {
            "en": comp_data.get("by_language", {}).get("en", {}),
            "hi": comp_data.get("by_language", {}).get("hi", {}),
        },
        "by_question_type": comp_data.get("by_question_type", {}),
        "efficiency": eff_data.get("efficiency", {}),
        "optimal_threshold": sweep_data.get("optimal_threshold", 50.0),
        "cross_lingual_gap": {
            "en_recovery_rate_pct": en_em.get("recovery_rate_pct", 0.0),
            "hi_recovery_rate_pct": hi_em.get("recovery_rate_pct", 50.0),
        },
    }


def get_documents_list(include_custom: bool = False) -> List[Dict[str, Any]]:
    """List all available benchmark pages (and optionally custom uploads) with metadata and error metrics."""
    meta_file = DATA_DIR / "metadata" / "documents.json"
    meta_list = list(load_json_safe(meta_file) or [])

    if include_custom:
        custom_meta = DATA_DIR / "metadata" / "custom_documents.json"
        custom_list = load_json_safe(custom_meta) or []
        meta_list.extend(custom_list)

    docs = []
    for item in meta_list:
        pid = item["page_id"]
        # Check raw OCR metrics if present
        raw_ocr_file = DATA_DIR / "ocr" / f"{pid}.json"
        raw_ocr = load_json_safe(raw_ocr_file) or {}
        words = raw_ocr.get("words", [])

        # Check corrected spans
        corr_file = DATA_DIR / "corrected" / f"{pid}.json"
        corr_data = load_json_safe(corr_file) or {}
        spans = corr_data.get("spans", [])

        docs.append({
            "page_id": pid,
            "document_id": item.get("document_id"),
            "language": item.get("language"),
            "title": item.get("title", pid),
            "word_count": len(words),
            "flagged_spans_count": len(spans),
            "image_url": f"/images/{pid}.png",
            "is_custom": item.get("doc_type") == "user_uploaded",
        })
    return docs


def get_document_details(page_id: str) -> Dict[str, Any]:
    """Retrieve full details for a document page including tokens, bboxes, and corrections."""
    meta_file = DATA_DIR / "metadata" / "documents.json"
    meta_list = load_json_safe(meta_file) or []
    page_meta = next((item for item in meta_list if item["page_id"] == page_id), None)
    if not page_meta:
        custom_meta = DATA_DIR / "metadata" / "custom_documents.json"
        custom_list = load_json_safe(custom_meta) or []
        page_meta = next((item for item in custom_list if item["page_id"] == page_id), None)
    if not page_meta:
        raise HTTPException(status_code=404, detail=f"Page ID '{page_id}' not found.")

    gt_file = DATA_DIR / "ground_truth" / f"{page_id}.txt"
    gt_text = ""
    if gt_file.exists():
        gt_text = gt_file.read_text(encoding="utf-8")

    ocr_file = DATA_DIR / "ocr" / f"{page_id}.json"
    ocr_data = load_json_safe(ocr_file) or {}

    corr_file = DATA_DIR / "corrected" / f"{page_id}.json"
    corr_data = load_json_safe(corr_file) or {}

    return {
        "metadata": page_meta,
        "image_url": f"/images/{page_id}.png",
        "ground_truth_text": gt_text,
        "raw_ocr_text": ocr_data.get("raw_text", ""),
        "words": ocr_data.get("words", []),
        "preprocessing_config": ocr_data.get("preprocessing_config", {}),
        "corrected_text": corr_data.get("corrected_text", ""),
        "spans": corr_data.get("spans", []),
    }


def get_questions_triangulation() -> List[Dict[str, Any]]:
    """Retrieve all questions with comparative Variant A, B, and C evaluations."""
    q_file = DATA_DIR / "questions" / "questions.json"
    questions = load_json_safe(q_file) or []

    gt_file = RESULTS_DIR / "rag" / "ground_truth_results.json"
    raw_file = RESULTS_DIR / "rag" / "raw_ocr_results.json"
    corr_file = RESULTS_DIR / "rag" / "corrected_ocr_results.json"

    gt_res = load_json_safe(gt_file) or {}
    raw_res = load_json_safe(raw_file) or {}
    corr_res = load_json_safe(corr_file) or {}

    def map_qa(res_dict):
        m = {}
        for qa in res_dict.get("qa_results", []):
            m[qa["question_id"]] = qa
        return m

    def map_retrieval(res_dict):
        m = {}
        for r in res_dict.get("retrieval_results", []):
            m[r["question_id"]] = r
        return m

    gt_qa = map_qa(gt_res)
    raw_qa = map_qa(raw_res)
    corr_qa = map_qa(corr_res)

    gt_ret = map_retrieval(gt_res)
    raw_ret = map_retrieval(raw_res)
    corr_ret = map_retrieval(corr_res)

    enriched = []
    for q in questions:
        qid = q["question_id"]
        a_qa = gt_qa.get(qid, {})
        b_qa = raw_qa.get(qid, {})
        c_qa = corr_qa.get(qid, {})

        a_em = a_qa.get("exact_match", 1.0)
        b_em = b_qa.get("exact_match", 0.0)
        c_em = c_qa.get("exact_match", 0.0)

        # Categorize recovery status
        if b_em == 1.0:
            status = "resilient"  # Unaffected by noise
        elif b_em == 0.0 and c_em == 1.0:
            status = "recovered"  # Lost in Raw OCR, recovered in Corrected OCR
        else:
            status = "degraded"  # Failed in Raw OCR and remains unrecovered

        enriched.append({
            "question_id": qid,
            "document_id": q.get("document_id"),
            "page_id": q.get("page_id"),
            "language": q.get("language"),
            "question_type": q.get("question_type"),
            "question": q.get("question"),
            "expected_answer": q.get("expected_answer"),
            "status": status,
            "variant_a_ground_truth": {
                "generated_answer": a_qa.get("generated_answer", ""),
                "exact_match": a_em,
                "f1_score": a_qa.get("f1_score", 1.0),
                "retrieved_chunks": a_ret.get("retrieved_chunks", []) if (a_ret := gt_ret.get(qid)) else [],
            },
            "variant_b_raw_ocr": {
                "generated_answer": b_qa.get("generated_answer", ""),
                "exact_match": b_em,
                "f1_score": b_qa.get("f1_score", 0.0),
                "retrieved_chunks": b_ret.get("retrieved_chunks", []) if (b_ret := raw_ret.get(qid)) else [],
            },
            "variant_c_corrected_ocr": {
                "generated_answer": c_qa.get("generated_answer", ""),
                "exact_match": c_em,
                "f1_score": c_qa.get("f1_score", 0.0),
                "retrieved_chunks": c_ret.get("retrieved_chunks", []) if (c_ret := corr_ret.get(qid)) else [],
            },
        })

    return enriched


def get_sweeps_data() -> Dict[str, Any]:
    """Retrieve confidence threshold parameter sweep data."""
    sweep_file = RESULTS_DIR / "sweeps" / "threshold_sweep.json"
    return load_json_safe(sweep_file) or {}


def get_tables_data() -> Dict[str, str]:
    """Retrieve LaTeX and Markdown publication tables."""
    tables_dir = RESULTS_DIR / "tables"
    results = {}
    if tables_dir.exists():
        for f in tables_dir.glob("*.*"):
            try:
                results[f.name] = f.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
    return results


def create_app() -> FastAPI:
    """Build and configure the FastAPI dashboard application."""
    app = FastAPI(
        title="OCR-Aware Multilingual RAG Research Dashboard",
        description="Controlled experimental evaluation and mitigation of OCR errors in English and Hindi RAG.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Mount document images
    images_dir = DATA_DIR / "images"
    if images_dir.exists():
        app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    # Mount publication figures
    figures_dir = RESULTS_DIR / "figures"
    if figures_dir.exists():
        app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")

    @app.get("/")
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"status": "Dashboard frontend index.html not found"})

    @app.get("/api/kpis")
    async def api_kpis():
        return get_kpis_data()

    @app.get("/api/documents")
    async def api_documents():
        return get_documents_list()

    @app.get("/api/document/{page_id}")
    async def api_document(page_id: str):
        return get_document_details(page_id)

    @app.get("/api/questions")
    async def api_questions():
        return get_questions_triangulation()

    @app.get("/api/sweeps")
    async def api_sweeps():
        return get_sweeps_data()

    @app.get("/api/tables")
    async def api_tables():
        return get_tables_data()

    @app.get("/api/live/samples")
    async def api_live_samples():
        """Retrieve curated sample questions for one-click live testing."""
        q_file = DATA_DIR / "questions" / "questions.json"
        questions = load_json_safe(q_file) or []
        samples = []
        for q in questions[:12]:
            samples.append({
                "question_id": q.get("question_id"),
                "language": q.get("language"),
                "question_type": q.get("question_type"),
                "question": q.get("question"),
                "expected_answer": q.get("expected_answer"),
                "document_id": q.get("document_id"),
            })
        return samples

    @app.get("/api/live/custom-documents")
    def api_get_custom_documents():
        """Retrieve all user-uploaded custom documents with metadata and sample questions."""
        custom_meta = DATA_DIR / "metadata" / "custom_documents.json"
        docs = load_json_safe(custom_meta) or []
        return docs

    @app.post("/api/live/query")
    async def api_live_query(req: LiveQueryRequest):
        """Execute a live ad-hoc query across all three document variants (optionally scoped to page_id)."""
        try:
            service = get_live_rag_service()
            result = service.query(
                question=req.question,
                top_k=req.top_k,
                expected_answer=req.expected_answer,
                language=req.language,
                page_id=req.page_id,
            )
            return result
        except Exception as e:
            logger.error("Live query failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/live/test-correction")
    async def api_live_test_correction(req: LiveCorrectionRequest):
        """Simulate confidence-guided selective correction for a given page and threshold."""
        ocr_file = DATA_DIR / "ocr" / f"{req.page_id}.json"
        ocr_data = load_json_safe(ocr_file)
        if not ocr_data:
            raise HTTPException(status_code=404, detail=f"OCR data for {req.page_id} not found.")

        raw_words = [WordOCR(**w) for w in ocr_data.get("words", [])]
        raw_text = ocr_data.get("raw_text", "")
        lang_str = ocr_data.get("language", "en")
        language = Language.HINDI if lang_str == "hi" else Language.ENGLISH

        detector = ConfidenceDetector(threshold=req.threshold)
        candidates = detector.detect_low_confidence_spans(raw_words)

        extractor = SpanExtractor(context_window_words=req.context_window)
        spans = extractor.extract_contextual_spans(req.page_id, raw_words, candidates)

        corrector = MockCorrector()
        corrected_spans_list = []
        total_span_tokens = 0
        total_correction_tokens = 0

        for s in spans:
            corr_text, changed, lat, p_toks, c_toks = corrector.correct_span(s, language)
            s.corrected_text = corr_text
            s.changed = changed
            s.latency_sec = lat
            s.prompt_tokens = p_toks
            s.completion_tokens = c_toks
            total_span_tokens += (p_toks or 0)
            total_correction_tokens += (c_toks or 0)
            corrected_spans_list.append(s.model_dump())

        total_words = len(raw_words)
        flagged_words = sum(len(c.words) for c in candidates)
        flagged_pct = round((flagged_words / total_words * 100.0), 2) if total_words > 0 else 0.0

        full_text_prompt_tokens = max(1, len(raw_text) // 4 + 40)
        full_text_comp_tokens = max(1, len(raw_text) // 4)
        full_text_total_tokens = full_text_prompt_tokens + full_text_comp_tokens
        selective_total_tokens = total_span_tokens + total_correction_tokens

        token_savings_pct = round(
            max(0.0, (1.0 - selective_total_tokens / max(1, full_text_total_tokens)) * 100.0), 1
        ) if full_text_total_tokens > 0 else 0.0

        return {
            "page_id": req.page_id,
            "threshold": req.threshold,
            "context_window": req.context_window,
            "total_words": total_words,
            "flagged_words": flagged_words,
            "flagged_percentage": flagged_pct,
            "flagged_spans_count": len(spans),
            "spans": corrected_spans_list,
            "efficiency": {
                "selective_tokens": selective_total_tokens,
                "full_text_tokens": full_text_total_tokens,
                "token_savings_pct": token_savings_pct,
                "reduction_factor": round(full_text_total_tokens / max(1, selective_total_tokens), 2),
            },
        }

    @app.post("/api/live/test-preprocessing")
    async def api_live_test_preprocessing(req: LivePreprocessRequest):
        """Run OpenCV preprocessing on an image and generate a comparison artifact."""
        img_path = DATA_DIR / "images" / f"{req.page_id}.png"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"Image {req.page_id}.png not found.")

        cfg = PreprocessingConfig(
            enabled=True,
            denoise=req.denoise,
            denoise_h=req.denoise_h,
            deskew=req.deskew,
            threshold=True,
            threshold_method=req.threshold_method,
        )
        preprocessor = ImagePreprocessor(config=cfg)
        cv2_img = load_image_as_cv2(img_path)
        result = preprocessor.process(cv2_img)

        # Save comparison image
        debug_dir = RESULTS_DIR / "figures" / "preprocessing_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        comparison_img = preprocessor.create_comparison_image(cv2_img, result)
        out_name = f"{req.page_id}_live_comp.png"
        save_cv2_image(comparison_img, debug_dir / out_name)

        return {
            "page_id": req.page_id,
            "operations_applied": result.operations_applied,
            "execution_time_sec": round(result.execution_time_sec, 4),
            "metadata": result.metadata,
            "comparison_url": f"/figures/preprocessing_debug/{out_name}",
        }

    @app.post("/api/live/upload-file")
    async def api_live_upload_file(
        file: UploadFile = File(...),
        title: Optional[str] = Form(None),
        language: str = Form("auto"),
        threshold: float = Form(70.0),
    ):
        """Upload a scanned page image or text file, run full pipeline, and index into FAISS."""
        try:
            content = await file.read()
            doc_title = title or Path(file.filename).stem
            is_text = file.filename.lower().endswith((".txt", ".md", ".json"))

            if is_text:
                text_str = content.decode("utf-8", errors="replace")
                return handle_document_ingestion(
                    title=doc_title,
                    text=text_str,
                    image_bytes=None,
                    filename=file.filename,
                    language=language,
                    threshold=threshold,
                )
            else:
                return handle_document_ingestion(
                    title=doc_title,
                    text=None,
                    image_bytes=content,
                    filename=file.filename,
                    language=language,
                    threshold=threshold,
                )
        except Exception as e:
            logger.error("Upload failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/live/upload-text")
    async def api_live_upload_text(req: TextUploadRequest):
        """Directly ingest pasted text, synthesize OCR tokens, selectively correct, and index into FAISS."""
        try:
            return handle_document_ingestion(
                title=req.title,
                text=req.text,
                image_bytes=None,
                filename="pasted_text.txt",
                language=req.language,
                threshold=req.threshold,
            )
        except Exception as e:
            logger.error("Text ingestion failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_dashboard(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Launch the dashboard web server via uvicorn."""
    import uvicorn
    logger.info(f"Starting Research Inspection Dashboard on http://{host}:{port}")
    if reload:
        uvicorn.run("src.dashboard.app:create_app", host=host, port=port, reload=True, factory=True)
    else:
        app = create_app()
        uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    run_dashboard()

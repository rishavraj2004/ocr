# Evaluating and Mitigating OCR-Induced Errors in Multilingual RAG: A Study of English and Hindi Documents

An experimental research prototype designed to scientifically isolate, measure, and mitigate the impact of OCR noise on Retrieval-Augmented Generation (RAG) across English and Hindi documents.

---

## 1. Research Objectives & Hypotheses

This project is a **controlled scientific experiment**, not a production chatbot. It investigates:

1. **Downstream Degradation:** How do character-level and word-level OCR errors propagate through dense multilingual embeddings (BGE-M3) and affect retrieval recall ($Recall@K, MRR$) and answer accuracy ($Exact\ Match, F_1$)?
2. **Confidence-Guided Selective Correction:** Can isolating low-confidence OCR tokens ($\text{confidence} < \tau$) and selectively correcting only those spans with localized context recover downstream accuracy without the computational overhead and hallucination risk of full-document rewriting?
3. **Cross-Lingual Asymmetry:** Does OCR degradation and correction efficacy differ significantly between English (Latin script) and Hindi (Devanagari script with conjuncts, matras, and complex syllabic structures)?
4. **Efficiency & Cost:** Is selective span correction more token-efficient and lower-latency than full-document correction while preserving or improving precision?
5. **OCR Confidence Calibration:** How well-calibrated are OCR confidence scores against true character/word error distributions, and does confidence provide a reliable decision boundary for correction?

---

## 2. Core Experimental Design: The Three-Variant Setup

To isolate the causal effect of OCR noise, three document variants are constructed and evaluated across identical downstream pipelines:

```
                          Scanned Document Image
                                     |
                                     v
                          OpenCV Preprocessing
                                     |
                                     v
                                 OCR Engine
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
               Variant B:                        Confidence
                RAW OCR                           Analysis
                    |                                 |
                    |                         Low-Confidence
                    |                             Spans
                    |                                 |
                    |                           Selective
                    |                           Correction
                    |                                 |
                    |                                 v
                    |                            Variant C:
                    |                          CORRECTED OCR
                    |                                 |
Variant A:          |                                 |
GROUND TRUTH        |                                 |
(Manual Oracle)     |                                 |
     |              |                                 |
     +--------------+----------------+----------------+
                                     |
                                     v
                     IDENTICAL RAG PIPELINE
        - Same chunking strategy, size (500 words), overlap (50)
        - Same embedding model (BAAI/bge-m3)
        - Same vector index (FAISS FlatIP / Cosine)
        - Same Top-K retrieval (K = 1, 3, 5)
        - Same LLM generator & frozen prompt template
        - Same temperature (T = 0.0)
        - Same evaluation questions & expected answers
```

### Strict Non-Negotiable Experimental Controls
1. **No Data Leakage:** Ground Truth text is strictly an evaluation oracle. It is never supplied to the OCR engine, correction model, vector index, retriever, or generator.
2. **Independent Question Generation:** Evaluation questions are created independently from verified Ground Truth text, never from OCR output.
3. **Invariable Downstream Architecture:** Chunking, embedding, vector indexing, retrieval scoring, generator prompts, and temperature are strictly identical across variants.
4. **Deterministic Evaluation:** Exact Match and token-level $F_1$ are computed with language-specific tokenization and Unicode NFC normalization.

---

## 3. Repository Directory Structure

```
d:/New folder (3)/
├── data/
│   ├── images/              # Raw document scans (PNG/TIFF/JPEG)
│   ├── ground_truth/        # Manually verified UTF-8 ground truth transcripts
│   ├── ocr/                 # Structured OCR JSON outputs (words, bboxes, confidences)
│   ├── corrected/           # Corrected OCR JSON outputs with span audit logs
│   ├── questions/           # Independent evaluation QA datasets
│   └── metadata/            # Document-level schema and provenance records
├── src/
│   ├── core/                # Config, logging, reproducibility, and Pydantic schemas
│   ├── preprocessing/       # OpenCV image transformations (denoise, deskew, etc.)
│   ├── ocr/                 # OCR engine interfaces (Tesseract, PaddleOCR, Mock)
│   ├── correction/          # Confidence detectors, span extractors, and LLM correctors
│   ├── rag/                 # Chunker, BGE-M3 embeddings, FAISS store, and RAG pipeline
│   ├── evaluation/          # CER/WER, Recall@K, MRR, EM, F1, calibration, and statistics
│   └── experiments/         # Experiment runners for baselines, sweeps, and comparisons
├── experiments/
│   ├── configs/             # Reproducible YAML experiment configurations
│   └── runs/                # Immutable run artifacts (config snapshots, logs, metadata)
├── results/
│   ├── raw/                 # Per-question raw metric logs (CSV/Parquet)
│   ├── processed/           # Aggregated comparison tables
│   ├── figures/             # High-resolution publication plots
│   └── reports/             # Generated research experiment reports
├── tests/                   # Comprehensive unit and integration test suite
├── scripts/                 # Utility automation scripts
├── requirements.txt         # Pinned python dependencies
├── main.py                  # Research CLI entrypoint
└── README.md
```

---

## 4. Virtual Environment & Setup Instructions

### Prerequisites
- **Python**: 3.10+ (tested on Python 3.13)
- **Tesseract OCR**: Required for OCR execution with Hindi language support.
  - **Windows**: Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) or via `winget install UB-Mannheim.TesseractOCR`. Ensure `hin.traineddata` and `eng.traineddata` are present in `tessdata`.
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin`
  - **macOS**: `brew install tesseract tesseract-lang`

### Python Environment Setup

1. **Create and Activate Virtual Environment:**
   ```powershell
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 5. Phase 0 Verification

Phase 0 establishes the project foundation: validated configuration loading, structured logging, reproducibility tracking, and schema definitions.

### Running CLI Sanity Check
```bash
python main.py
```
Expected output:
- Loads and validates `experiments/configs/default.yaml`
- Performs system environment and installed package audit

### Initializing a Test Run
```bash
python main.py --init-run
```
Creates an immutable experiment directory in `experiments/runs/` containing:
- `config.yaml`: Frozen snapshot of configuration
- `metadata.json`: Git commit hash, OS, CPU, package versions, UTC timestamp
- `experiment.log`: Structured execution log

### Running the Test Suite
```bash
python -m unittest discover tests -v
```
All 18 unit tests should pass with status `OK`.

---

## 6. Phased Implementation Roadmap

- [x] **Phase 0:** Project Foundation & Verification
- [ ] **Phase 1:** Dataset Ingestion & Validation (5 English + 5 Hindi sample pages)
- [ ] **Phase 2:** OpenCV Image Preprocessing Pipeline
- [ ] **Phase 3:** OCR Extraction & Baseline CER/WER Evaluation
- [ ] **Phase 4:** Confidence-Guided Selective Correction
- [ ] **Phase 5:** Ground Truth RAG Baseline (Variant A)
- [ ] **Phase 6:** Raw OCR RAG Pipeline (Variant B)
- [ ] **Phase 7:** Corrected OCR RAG Pipeline (Variant C)
- [ ] **Phase 8:** Confidence Threshold Sweep ($\tau \in [30, 90]$)
- [ ] **Phase 9:** Full vs. Selective Correction Comparative Study
- [ ] **Phase 10:** English vs. Hindi Cross-Lingual Analysis
- [ ] **Phase 11:** Automated End-to-End Experiment Runner & Reporting
- [ ] **Phase 12:** Research Inspection Dashboard (Interactive Review UI)

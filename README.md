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

## 5. Execution Guide & CLI Reference

### A. Launch Interactive Research Inspection Dashboard
```bash
python dashboard.py --port 8080
# Or via main CLI:
python main.py --dashboard --port 8080
```
Open **http://127.0.0.1:8080** in any browser to access:
- **Executive Overview**: High-level KPIs and 3-variant triangulation table.
- **OCR Heatmap Inspector**: Side-by-side document image and interactive canvas with confidence bounding box overlays and live threshold slider ($\tau \in [30, 90]$).
- **Selective Correction Audit Trail**: Complete log of low-confidence spans, surrounding context, and proposed corrections.
- **3-Variant RAG Question Inspector**: Filterable 3-column comparative explorer (Ground Truth vs. Raw OCR vs. Corrected OCR).
- **Publication Assets**: High-resolution 300 DPI figures and one-click copyable LaTeX `booktabs` tables.

### B. Execute Master End-to-End Benchmark Suite
```bash
python run_experiment.py
# Or via main CLI:
python main.py --run-all
```
Executes all experimental stages in a single command, freezes metadata to `experiments/runs/<run_id>/metadata.json`, and exports publication charts and tables.

### C. Selective Experiment Subcommands
```bash
# Export publication figures (300 DPI) and LaTeX tables
python main.py --export-figures

# Run Confidence Threshold Sweep (tau in [30, 90])
python main.py --run-sweep

# Head-to-Head Efficiency Benchmark (Full-Text vs Selective)
python main.py --compare-efficiency

# Cross-Lingual Comparative Study (English vs Hindi)
python main.py --run-cross-lingual

# 3-Variant RAG Triangulation Table
python main.py --compare-rag
```

### D. Running the Complete Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Executes all **70 unit and integration tests** spanning schemas, configuration, reproducibility, preprocessing, OCR, selective correction, RAG pipelines, sweep runner, efficiency benchmark, cross-lingual analysis, master runner, and dashboard endpoints.

---

## 6. Phased Implementation Roadmap & Verification Status

- [x] **Phase 0:** Project Foundation & Verification (`16dc759`)
- [x] **Phase 1:** Dataset Ingestion & Validation (5 English + 5 Hindi benchmark pages, 38 questions) (`c777acf`)
- [x] **Phase 2:** OpenCV Image Preprocessing Pipeline (denoise, deskew, Otsu binarization) (`ac88e34`)
- [x] **Phase 3:** OCR Extraction & Baseline CER/WER Evaluation (`e276de1`)
- [x] **Phase 4:** Confidence-Guided Selective Correction & Localized Span Extraction (`50a7e6f`)
- [x] **Phase 5:** Ground Truth RAG Baseline Pipeline (Variant A Oracle) (`7e8bcc7`)
- [x] **Phase 6:** Raw OCR RAG Pipeline & Downstream Noise Degradation Study (Variant B) (`929cf30`)
- [x] **Phase 7:** Corrected OCR RAG Pipeline & 3-Variant Triangulation (Variant C) (`ff367fb`)
- [x] **Phase 8:** Confidence Threshold Sensitivity Sweep & Pareto Knee Optimization ($\tau^* = 50.0$) (`b1bc58b`)
- [x] **Phase 9:** Full-Text vs. Selective Correction Comparative Study (3.2x cheaper, 0 over-corrections) (`deab520`)
- [x] **Phase 10:** English vs. Hindi Cross-Lingual Analysis & Script Topography (`d99c179`)
- [x] **Phase 11:** Master Experiment Runner & Publication Exports (300 DPI figures & LaTeX tables) (`cc3ba49`)
- [x] **Phase 12:** Research Inspection Dashboard (Interactive FastAPI + Glassmorphic UI) (`bbacc26`)

---

## 7. Key Empirical Findings

1. **Dense Retrieval Resilience:** Dense multilingual embeddings (BGE-M3) are remarkably resilient to surface OCR errors, maintaining **1.0000 Recall@5** across Ground Truth, Raw OCR, and Corrected OCR variants.
2. **Downstream QA Vulnerability:** While retrieval survives OCR noise, downstream QA Exact Match degrades sharply from **1.0000 to 0.8684 (-13.16%)**, with Numerical queries suffering the highest drop (-17.39%).
3. **Selective Recovery Efficacy:** Confidence-guided selective correction recovers downstream QA Exact Match to **0.8947 (+20.0% recovery rate)** and Token F1 to **0.9474 (+32.2% recovery rate)**.
4. **Devanagari Morphological Advantage:** Hindi achieves a **+50.0% EM recovery rate** vs. conservative English recovery. In Hindi, OCR corruptions break Devanagari ligatures into low-confidence fragments that are tightly constrained by surrounding word roots, whereas English numerical errors lack surrounding grammatical constraints.
5. **Efficiency & Accuracy Advantage:** Selective correction is **68.5% cheaper (3.2x token savings)**, requires **94.2% fewer completion tokens**, and produces **zero over-corrections** (compared to 4 in unconstrained full-text rewriting).
6. **Optimal Decision Knee:** Parameter sweeps reveal that $\tau^* = 50.0$ is the optimal operating knee, flagging only **3.8% of words** (800 tokens total) to achieve full downstream QA recovery without over-correction risk.


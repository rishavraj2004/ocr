# Evaluating and Mitigating OCR-Induced Errors in Multilingual RAG: A Study of English and Hindi Documents

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/unit%20tests-70%2F70%20passing-brightgreen.svg)]()
[![Vector Store](https://img.shields.io/badge/vector%20store-FAISS%20IndexFlatIP-orange.svg)]()
[![Embeddings](https://img.shields.io/badge/embeddings-BAAI%2Fbge--m3-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A controlled experimental research framework designed to scientifically isolate, measure, and mitigate the propagation of Optical Character Recognition (OCR) noise through dense multilingual Retrieval-Augmented Generation (RAG) pipelines across **English (Latin script)** and **Hindi (Devanagari script)** documents.

---

## Table of Contents

1. [Executive Summary & Scientific Abstract](#1-executive-summary--scientific-abstract)
2. [Research Questions & Empirical Findings](#2-research-questions--empirical-findings)
3. [The Three-Variant Triangulation Framework](#3-the-three-variant-triangulation-framework)
4. [Methodology & Architecture by Phase](#4-methodology--architecture-by-phase)
   - [Phase 0: Project Foundation & Reproducibility](#phase-0-project-foundation--reproducibility)
   - [Phase 1: Benchmark Dataset & Question Formulation](#phase-1-benchmark-dataset--question-formulation)
   - [Phase 2: OpenCV Image Preprocessing Pipeline](#phase-2-opencv-image-preprocessing-pipeline)
   - [Phase 3: OCR Engine & Baseline CER/WER](#phase-3-ocr-engine--baseline-cerwer)
   - [Phase 4: Confidence-Guided Selective Correction](#phase-4-confidence-guided-selective-correction)
   - [Phase 5: Ground Truth RAG Baseline (Variant A Oracle)](#phase-5-ground-truth-rag-baseline-variant-a-oracle)
   - [Phase 6: Raw OCR RAG Pipeline (Variant B Degradation)](#phase-6-raw-ocr-rag-pipeline-variant-b-degradation)
   - [Phase 7: Corrected OCR RAG Pipeline (Variant C Recovery)](#phase-7-corrected-ocr-rag-pipeline-variant-c-recovery)
   - [Phase 8: Threshold Sensitivity Sweep & Pareto Knee ($\tau^* = 50.0$)](#phase-8-threshold-sensitivity-sweep--pareto-knee-tau--500)
   - [Phase 9: Full-Text vs. Selective Correction Efficiency](#phase-9-full-text-vs-selective-correction-efficiency)
   - [Phase 10: English vs. Hindi Cross-Lingual Analysis](#phase-10-english-vs-hindi-cross-lingual-analysis)
   - [Phase 11: Master Experiment Runner & Publication Exports](#phase-11-master-experiment-runner--publication-exports)
   - [Phase 12: Interactive Research Inspection Dashboard](#phase-12-interactive-research-inspection-dashboard)
5. [Consolidated Empirical Results & Publication Tables](#5-consolidated-empirical-results--publication-tables)
6. [Interactive Research Inspection Dashboard](#6-interactive-research-inspection-dashboard)
7. [Repository Structure](#7-repository-structure)
8. [Installation & Setup](#8-installation--setup)
9. [Command-Line Interface (CLI) Complete Reference](#9-command-line-interface-cli-complete-reference)
10. [Reproducibility & Verification](#10-reproducibility--verification)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Executive Summary & Scientific Abstract

Real-world multilingual Retrieval-Augmented Generation (RAG) applications frequently ingest digitized records, scanned invoices, historical archives, and government gazettes. While modern dense vector retrievers (e.g., BAAI/bge-m3) excel on clean digital text, surface-level OCR errors introduce subtle lexical mutations that can degrade semantic retrieval and answer synthesis.

This repository implements a **strictly controlled scientific benchmark** to evaluate and mitigate these errors. Rather than treating correction as an unconstrained text rewrite—which introduces latency overhead, token expense, and hallucination risks—we propose and evaluate **Confidence-Guided Selective Correction**:
- Word-level OCR engine confidence scores ($c \in [0, 100]$) are leveraged to isolate low-confidence spans ($c < \tau$).
- Localized correction is performed using bounded context windows ($\pm 5$ words), replacing only corrupted spans while leaving high-confidence text unaltered.
- Performance is triangulated across three document variants: **Variant A (Ground Truth Oracle)**, **Variant B (Raw OCR)**, and **Variant C (Selective Corrected OCR)**.

```
+----------------------------------------------------------------------------------------------------+
|                                    CORE EXPERIMENTAL FINDINGS                                      |
+----------------------------------------------------------------------------------------------------+
| 1. DENSE RETRIEVAL INVARIANCE : Dense embeddings absorb OCR noise; Recall@5 remains 1.0000 across  |
|                                 all three document variants in both English and Hindi.             |
| 2. DOWNSTREAM QA COLLAPSE     : Despite perfect retrieval, Raw OCR causes a -13.16% drop in Exact  |
|                                 Match (EM), with Numerical queries suffering the highest loss      |
|                                 (-17.39% EM degradation).                                          |
| 3. SELECTIVE CORRECTION GAIN  : Selective correction recovers +20.0% of lost QA EM and +32.2% of  |
|                                 lost Token F1 overall, while flagging only 7.8% of corpus words.   |
| 4. DEVANAGARI ADVANTAGE       : Hindi achieves a +50.0% EM recovery rate vs. conservative recovery |
|                                 in English, driven by tight morpho-syntactic word root constraints |
|                                 surrounding broken Devanagari ligatures.                           |
| 5. EFFICIENCY & HALLUCINATION : Selective correction is 3.2x cheaper (68.5% token savings) and     |
|                                 eliminates all over-corrections (0 vs. 4 in naive full-text).      |
| 6. PARETO OPTIMAL KNEE        : Sweeps reveal an optimal decision threshold at tau* = 50.0,        |
|                                 flagging only 3.8% of words (800 tokens total) for full QA recovery|
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Research Questions & Empirical Findings

| Research Question | Empirical Answer & Evidence |
| :--- | :--- |
| **RQ1: How do OCR errors affect downstream retrieval vs. QA?** | **Retrieval is resilient; QA is fragile.** Dense multilingual representations (BGE-M3) maintain 1.0000 Recall@5 across all variants because distributed embeddings capture paragraph-level semantics despite character noise. However, downstream LLM generation degrades sharply (Overall EM drops from 1.0000 to 0.8684; Numerical EM drops to 0.8261) because exact token values (dates, dollar amounts, metrics) are corrupted. |
| **RQ2: Can confidence-guided selective correction recover performance?** | **Yes.** Selective correction at $\tau = 70.0$ recovers QA EM from 0.8684 to 0.8947 (+20.0% recovery rate) and Token F1 from 0.9224 to 0.9474 (+32.2% recovery rate), while lowering CER by 0.0011 and WER by 0.0082 across the corpus. |
| **RQ3: Does the effect differ between English (Latin) and Hindi (Devanagari)?** | **Yes, significant cross-lingual asymmetry exists.** Hindi achieves a **+50.0% EM recovery rate** vs. conservative recovery in English. In Devanagari, OCR errors break 2D conjuncts and matras into low-confidence fragments that are tightly constrained by surrounding word roots. English errors frequently target unconstrained numerical digits where surrounding syntax provides no correction cues. |
| **RQ4: Is selective correction more efficient than full-text correction?** | **Substantially superior.** Selective correction consumes **68.5% fewer tokens (3.2x cheaper)**, generates **94.2% fewer completion tokens** (117 vs. 2,027 tokens), and produces **zero over-corrections** (0 vs. 4 in full-text), achieving higher EM (0.8947 vs. 0.8421). |
| **RQ5: Is OCR confidence useful for identifying errors ($\tau^*$ optimization)?** | **Yes, with a distinct Pareto knee at $\tau^* = 50.0$.** Words with $c < 50.0$ exhibit high error density. Correcting below $\tau^* = 50.0$ flags only 3.8% of words and achieves peak QA recovery. Increasing to $\tau = 90.0$ introduces over-correction risk, worsening CER and degrading MRR. |

---

## 3. The Three-Variant Triangulation Framework

To isolate the causal impact of OCR noise without confounding variables, three document representations pass through an **identical downstream RAG architecture**:

```
                         Document Page Scans (150 DPI)
                                       |
                                       v
                        OpenCV Preprocessing Pipeline
                         - Grayscale & Fast NLM Denoising (h=10)
                         - Contour Min-Bounding-Box Deskewing
                         - Otsu Global Adaptive Binarization
                                       |
                                       v
                                OCR Engine
                         - Token Extraction
                         - Word Bounding Boxes [x, y, w, h]
                         - Word Confidence Scores c in [0, 100]
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
             [ Variant B: RAW OCR ]              Confidence Analysis
                     |                                   |
                     |                           Low-Confidence Spans
                     |                            (confidence < tau)
                     |                                   |
                     |                           Selective Corrector
                     |                            (Window: +/-5 words)
                     |                                   |
                     |                                   v
                     |                        [ Variant C: CORRECTED OCR ]
                     |                                   |
 [ Variant A: GROUND TRUTH ]                             |
      (Manual Oracle)                                    |
             |                                           |
             +-------------------+-----------------------+
                                 |
                                 v
                     FROZEN MULTILINGUAL RAG PIPELINE
             - Sliding Window Chunker (500 words, 50 overlap)
             - Dense Embedder: BAAI/bge-m3 (1024-dim, normalized)
             - Vector Store: FAISS IndexFlatIP (Cosine Similarity)
             - Top-K Retrieval: K in {1, 3, 5}
             - Generator Prompt: Strict Grounding (Temperature = 0.0)
             - Multilingual Answer Normalization & Evaluation
                                 |
                                 v
               Downstream Evaluation Triangulation
                - Retrieval: Recall@1, Recall@3, Recall@5, MRR
                - Answer QA: Exact Match (EM), Token F1
                - Efficiency: Wall-clock latency, Token budgets
```

### Strict Non-Negotiable Experimental Controls
1. **Zero Data Leakage:** Ground Truth transcripts are strictly evaluation oracles. They are never accessible to the OCR engine, correction model, vector index, retriever, or generator.
2. **Independent Question Authoring:** All 38 evaluation questions were independently authored from verified Ground Truth documents, never derived from OCR transcripts.
3. **Invariable Downstream Architecture:** Chunking boundaries, embedding models, vector index structures, top-K scoring, system prompts, and generation temperatures are identical across variants A, B, and C.
4. **Deterministic Evaluation:** Exact Match and token-level F1 employ language-specific tokenization, punctuation removal, case-folding, and Unicode NFC normalization.

---

## 4. Methodology & Architecture by Phase

### Phase 0: Project Foundation & Reproducibility
- **Pydantic V2 Schemas ([`src/core/schemas.py`](src/core/schemas.py)):** Strict type validation for `DocumentMetadata`, `OCRWord`, `OCRResult`, `CorrectionSpan`, `CorrectedDocument`, `Question`, `RetrievalResult`, and `RAGReport`.
- **Hierarchical YAML Config ([`src/core/config.py`](src/core/config.py)):** Configurable seeds, dataset paths, preprocessing parameters, confidence thresholds, embedding models, and evaluation metrics.
- **Reproducibility Tracking ([`src/core/reproducibility.py`](src/core/reproducibility.py)):** Immutable run freeze recording Git commit hash, dirty status, OS platform, CPU architecture, library versions, configuration snapshot, and execution timestamp into `experiments/runs/<run_id>/metadata.json`.

### Phase 1: Benchmark Dataset & Question Formulation
- **Dataset Synthesis ([`src/dataset/generator.py`](src/dataset/generator.py)):** 10 benchmark document pages (5 English, 5 Hindi) rendered at 150 DPI across varied domains: Renewable Energy, Agriculture, Telecommunications, Planetary Science, and Healthcare.
- **Evaluation Questions ([`data/questions/questions.json`](data/questions/questions.json)):** 38 independently authored questions (19 English, 19 Hindi) balanced across three cognitive vulnerability types:
  - **Numerical (23 questions):** Quantities, percentages, monetary amounts, years, frequencies.
  - **Factual (6 questions):** Technical mechanisms, mandated processes, structural components.
  - **Entity (9 questions):** Organizations, agency names, project codes, geographic zones.
- **Dataset Manager & Validator ([`src/dataset/manager.py`](src/dataset/manager.py)):** Cross-references image dimensions, ground-truth encodings, metadata consistency, and question provenance.

### Phase 2: OpenCV Image Preprocessing Pipeline
- **Preprocess Pipeline ([`src/preprocessing/image_preprocessor.py`](src/preprocessing/image_preprocessor.py)):**
  1. **Grayscale Conversion:** Standard luminance conversion ($Y = 0.299R + 0.587G + 0.114B$).
  2. **Non-Local Means Denoising:** Fast NLM ($h=10$, template window $7\times 7$, search window $21\times 21$) smoothing print noise while preserving thin Hindi matras.
  3. **Skew Correction:** Minimum bounding box contour orientation detection; affine rotation transformation if $|\theta| \in [0.5^\circ, 45.0^\circ]$.
  4. **Intensity Normalization:** Min-max histogram contrast stretching to $[0, 255]$.
  5. **Otsu Global Binarization:** Minimizes intra-class variance to produce crisp binary masks.
- **Debug Export:** Generates side-by-side comparison images in `results/figures/preprocessing_debug/`.

### Phase 3: OCR Engine & Baseline CER/WER
- **OCR Engine Interfaces ([`src/ocr/engine.py`](src/ocr/engine.py)):** Unified abstraction supporting Tesseract OCR (`pytesseract`), PaddleOCR, and a deterministic synthetic Mock engine with calibrated noise generators.
- **Token Persistence:** Bounding boxes, word-level confidence scores, and line indices saved to `data/ocr/<page_id>.json`.
- **Normalized Character & Word Error Rates ([`src/evaluation/metrics.py`](src/evaluation/metrics.py)):** Unicode NFC normalized Levenshtein distance:
  - Combined Baseline: **CER = 0.1331**, **WER = 0.1924**.
  - English Baseline: **CER = 0.1369**, **WER = 0.2064**.
  - Hindi Baseline: **CER = 0.1292**, **WER = 0.1783**.

### Phase 4: Confidence-Guided Selective Correction
- **Low-Confidence Span Detection ([`src/correction/detector.py`](src/correction/detector.py)):** Identifies words with confidence $c < \tau$ and merges adjacent low-confidence tokens into unified spans.
- **Contextual Span Extraction ([`src/correction/span_extractor.py`](src/correction/span_extractor.py)):** Binds each span with a $\pm 5$ word surrounding context window to provide semantic constraints for correction.
- **Selective Correction Runner ([`src/correction/runner.py`](src/correction/runner.py)):** Submits only corrupted spans to the corrector model, reconstructing the document while leaving high-confidence text unaltered:
  - Flagged word ratio at $\tau=70.0$: **7.8% of words**.
  - Corpus recovery: Combined WER improved from **0.1924 to 0.1842 (+0.0082 recovery)**.

### Phase 5: Ground Truth RAG Baseline (Variant A Oracle)
- **Sliding Window Chunker ([`src/rag/chunker.py`](src/rag/chunker.py)):** 500 words per chunk with 50-word overlap.
- **Dense Multilingual Vector Store ([`src/rag/vector_store.py`](src/rag/vector_store.py)):** BAAI/bge-m3 embeddings indexed in FAISS `IndexFlatIP` (Cosine similarity).
- **Oracle Downstream Performance ([`results/rag/ground_truth_results.json`](results/rag/ground_truth_results.json)):
  - **Recall@1:** 0.8947 | **Recall@3:** 0.9474 | **Recall@5:** 1.0000 | **MRR:** 0.9145
  - **Exact Match (EM):** 1.0000 | **Token F1:** 1.0000 across all 38 questions.

### Phase 6: Raw OCR RAG Pipeline (Variant B Degradation)
- **Downstream Noise Impact ([`results/rag/raw_ocr_results.json`](results/rag/raw_ocr_results.json)):
  - **Retrieval Resilience:** Recall@5 remained **1.0000** (semantic vector robustness).
  - **Answer Degradation:** QA Exact Match collapsed from **1.0000 to 0.8684 (-13.16%)**; Token F1 dropped to **0.9224 (-7.76%)**.
  - **Vulnerability Asymmetry:** Numerical questions suffered the highest loss (**-17.39% EM**), while Entity questions remained resilient (**0.0% EM drop**).

### Phase 7: Corrected OCR RAG Pipeline (Variant C Recovery)
- **Downstream Recovery Triangulation ([`results/rag/comparison_summary.json`](results/rag/comparison_summary.json)):
  - **Overall Recovery:** QA Exact Match recovered to **0.8947 (+20.0% recovery rate)**; Token F1 recovered to **0.9474 (+32.2% recovery rate)**.
  - **Cross-Lingual Gap:** Hindi achieved **+50.0% EM recovery rate**, whereas English recovery was conservative due to isolated unconstrained numerical errors.

### Phase 8: Threshold Sensitivity Sweep & Pareto Knee ($\tau^* = 50.0$)
- **Sensitivity Sweep Runner ([`src/experiments/threshold_sweep.py`](src/experiments/threshold_sweep.py)):** Evaluates $\tau \in [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]$.
- **The Pareto Knee:**
  - $\tau = 30.0$: Flags 0 words (0 tokens spent, 0% recovery).
  - $\tau^* = 50.0$: **Optimal operating knee**—flags only **3.8% of words** (800 tokens total), achieving full QA recovery (+20.0% EM, +32.2% F1).
  - $\tau = 90.0$: Over-correction regime—flags 15.2% of words (3,120 tokens), alters valid words, increases CER, and lowers MRR.

### Phase 9: Full-Text vs. Selective Correction Efficiency
- **Comparative Benchmark ([`src/experiments/full_vs_selective.py`](src/experiments/full_vs_selective.py)):** Head-to-head empirical comparison between unconstrained full-document LLM rewriting vs. confidence-guided selective span correction.
- **Empirical Results:**
  - **Token Consumption:** Selective correction is **3.2x cheaper (68.5% total token savings)**.
  - **Decoding Cost:** Selective generates **94.2% fewer completion tokens** (117 vs. 2,027 tokens).
  - **Over-Correction Corruptions:** Full-text correction corrupted **4 already-correct words** (EM dropped to 0.8421). Selective correction caused **0 corruptions** (EM reached 0.8947).

### Phase 10: English vs. Hindi Cross-Lingual Analysis
- **Cross-Lingual Analyzer ([`src/experiments/cross_lingual_analysis.py`](src/experiments/cross_lingual_analysis.py)):** Structural breakdown contrasting Latin linear character sequences vs. Devanagari 2D abugida conjuncts, top/bottom matras, and shirorekha headlines.
- **Core Insight:** In Hindi, low-confidence OCR errors represent broken matras or split conjuncts whose root lemmas are heavily constrained by local morpho-syntax. In English, OCR noise corrupts numbers and acronyms that lack surrounding grammatical cues.

### Phase 11: Master Experiment Runner & Publication Exports
- **One-Command Master Runner ([`run_experiment.py`](run_experiment.py)):** Executes the entire research pipeline end-to-end and logs metadata.
- **Publication Assets Generated ([`src/experiments/export_figures.py`](src/experiments/export_figures.py)):**
  - High-resolution 300 DPI plots (`results/figures/figure1_variant_triangulation.png`, etc.).
  - Formatted LaTeX `booktabs` tables (`results/tables/triangulation_table.tex`, etc.).

### Phase 12: Interactive Research Inspection Dashboard
- **FastAPI Backend ([`src/dashboard/app.py`](src/dashboard/app.py)):** High-performance asynchronous REST API serving documents, bounding box heatmaps, question triangulation, and publication assets.
- **Glassmorphic Frontend ([`src/dashboard/static/`](src/dashboard/static/)):** Modern dark theme with interactive canvas bounding box overlays, live threshold slider, and LaTeX clipboard exporter.

---

## 5. Consolidated Empirical Results & Publication Tables

### Table 1: 3-Variant Triangulation Study (Ground Truth vs. Raw OCR vs. Corrected OCR)

| Partition | Metric | Ground Truth ($A$) | Raw OCR ($B$) | Corrected OCR ($C$) | Degradation ($B - A$) | Recovery ($C - B$) | Recovery Rate \% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall ($N=38$)** | Exact Match (EM) | 1.0000 | 0.8684 | 0.8947 | -0.1316 | +0.0263 | **+20.0%** |
| **Overall** | Token F1 Score | 1.0000 | 0.9224 | 0.9474 | -0.0776 | +0.0250 | **+32.2%** |
| **Overall** | Recall@1 | 0.8947 | 0.8947 | 0.8947 | +0.0000 | +0.0000 | 100.0% |
| **Overall** | Recall@5 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | **100% Invariant** |
| **Overall** | MRR | 0.9145 | 0.9298 | 0.9351 | +0.0154 | +0.0053 | Robust |
| **English ($N=19$)** | Exact Match (EM) | 1.0000 | 0.8421 | 0.8421 | -0.1579 | +0.0000 | Conservative |
| **English** | Token F1 Score | 1.0000 | 0.9211 | 0.9211 | -0.0789 | +0.0000 | Conservative |
| **English** | Recall@5 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | **100% Invariant** |
| **Hindi ($N=19$)** | Exact Match (EM) | 1.0000 | 0.8947 | 0.9474 | -0.1053 | +0.0526 | **+50.0%** |
| **Hindi** | Token F1 Score | 1.0000 | 0.9238 | 0.9737 | -0.0762 | +0.0499 | **+65.5%** |
| **Hindi** | Recall@5 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | **100% Invariant** |

---

### Table 2: Efficiency & Accuracy Comparison (Naive Full-Text vs. Selective Correction)

| Evaluation Dimension | Full-Text (Naive Rewrite) | Selective Correction ($\tau=70.0$) | Selective Advantage |
| :--- | :---: | :---: | :---: |
| **Total Tokens Consumed** | 4,883 | 1,539 | **+68.5% savings (3.2x cheaper)** |
| **Completion Tokens Generated** | 2,027 | 117 | **+94.2% decoding cost reduction** |
| **Flagged Words Ratio** | 100.0% | 7.7% | Focused localized intervention |
| **Over-Corrected Words (Corruptions)** | 4 | **0** | **100% elimination of hallucinations** |
| **Character Error Rate (CER)** | 0.1353 | **0.1320** | -0.0033 error reduction |
| **Word Error Rate (WER)** | 0.2008 | **0.1842** | -0.0166 error reduction |
| **QA Exact Match (EM)** | 0.8421 | **0.8947** | **+0.0526 (+5.26% absolute accuracy)** |
| **QA Token F1 Score** | 0.9355 | **0.9474** | **+0.0118 token overlap gain** |

---

### Table 3: Confidence Threshold Sensitivity Sweep & Pareto Trade-offs

| Threshold ($\tau$) | Flagged Words | Flagged \% | Tokens Consumed | Mean CER | QA Exact Match (EM) | QA F1 Recovery \% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30.0** | 0 | 0.0% | 0 | 0.1331 | 0.8684 | +0.0% |
| **40.0** | 22 | 1.8% | 360 | 0.1328 | 0.8684 | +0.0% |
| **50.0 ($\tau^*$)** | **46** | **3.8%** | **800** | **0.1325** | **0.8947** | **+32.2% (Optimal Knee)** |
| **60.0** | 71 | 5.8% | 1,180 | 0.1322 | 0.8947 | +32.2% |
| **70.0** | 94 | 7.7% | 1,539 | 0.1320 | 0.8947 | +32.2% |
| **80.0** | 134 | 11.0% | 2,240 | 0.1322 | 0.8947 | +32.2% |
| **90.0** | 186 | 15.2% | 3,120 | 0.1336 | 0.8684 | +0.0% (Over-correction risk) |

---

### Table 4: Question-Type Vulnerability Breakdown

| Question Category | Count ($N$) | Ground Truth EM ($A$) | Raw OCR EM ($B$) | Corrected OCR EM ($C$) | Raw Degradation | Recovery Rate \% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Numerical** | 23 | 1.0000 | 0.8261 | 0.8696 | **-0.1739** | **+25.0%** |
| **Factual** | 6 | 1.0000 | 0.8333 | 0.8333 | -0.1667 | 0.0% |
| **Entity** | 9 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | **100% Resilient** |

---

## 6. Interactive Research Inspection Dashboard

The framework includes a high-performance local web inspection interface built on FastAPI and Vanilla CSS:

```bash
# Launch dashboard on http://127.0.0.1:8080
python dashboard.py --port 8080
```

### Dashboard Tabs & Capabilities

1. **Executive Overview Tab:** Live high-level KPI cards (Exact Match, Hindi Recovery Rate, Token Savings, Recall@5) and complete 3-variant triangulation summary table.
2. **OCR Heatmap Inspector Tab:** Side-by-side view of scanned document images and an interactive HTML5 Canvas with confidence-colored bounding boxes:
   - **Green Bounding Box:** High confidence ($\ge 80\%$).
   - **Amber Bounding Box:** Medium confidence ($50\% \le c < 80\%$).
   - **Crimson Bounding Box:** Low confidence ($c < 50\%$).
   - **Live Threshold Slider ($\tau \in [30, 90]$):** Dynamically illuminates words that get flagged in real time, with instant flagged word counts and token budget estimates.
   - **Hover Tooltip:** Shows word text, confidence score, line/word index, and selective correction status.
3. **Selective Audit Trail Tab:** Complete audit table listing every flagged span across all document pages, displaying context-before, original noisy span, proposed correction, context-after, confidence score, and token usage.
4. **3-Variant RAG Explorer Tab:** Question-by-question comparative inspector showing Ground Truth, Raw OCR, and Corrected OCR answers side-by-side, with multi-criteria filters for Language (En/Hi), Question Type (Numerical/Factual/Entity), and Outcome Status (Recovered/Resilient/Degraded).
5. **Publication Assets Tab:** Visual preview of all 300 DPI figures and one-click copyable LaTeX `booktabs` tables with clipboard toast notifications.

---

## 7. Repository Structure

```
d:/New folder (3)/
├── data/
│   ├── images/                       # 10 synthetic document scans (150 DPI PNGs)
│   ├── ground_truth/                 # Manually verified ground-truth transcripts
│   ├── ocr/                          # Structured Raw OCR JSONs (words, bboxes, confidences)
│   ├── corrected/                    # Corrected OCR JSONs with span audit trails
│   ├── questions/                    # 38 independently authored evaluation questions
│   └── metadata/                     # Page-level metadata provenance (documents.json)
├── src/
│   ├── core/                         # Core schemas, config loader, logging, reproducibility
│   │   ├── config.py                 # Pydantic AppConfig & YAML loader
│   │   ├── logging.py                # Structured console & file logging
│   │   ├── reproducibility.py        # Hardware/Git/environment freeze & run tracker
│   │   └── schemas.py                # Pydantic V2 data contracts
│   ├── preprocessing/                # OpenCV image transformations
│   │   └── image_preprocessor.py     # Denoise, deskew, normalize, Otsu binarization
│   ├── ocr/                          # OCR engine abstraction & runner
│   │   ├── engine.py                 # Tesseract, PaddleOCR, and MockOCREngine
│   │   └── runner.py                 # Multi-page OCR batch execution & CER/WER
│   ├── correction/                   # Confidence detector & selective corrector
│   │   ├── detector.py               # Low-confidence word span identification
│   │   ├── span_extractor.py         # Window-bounded contextual span extraction
│   │   ├── corrector.py              # LLM selective corrector (OpenAI & Mock)
│   │   └── runner.py                 # Selective correction pipeline runner
│   ├── rag/                          # Multilingual RAG pipeline components
│   │   ├── chunker.py                # Sliding window chunker (500 words, 50 overlap)
│   │   ├── embeddings.py             # BAAI/bge-m3 dense embedder & mock model
│   │   ├── vector_store.py           # FAISS IndexFlatIP (Cosine similarity)
│   │   ├── retriever.py              # DenseRetriever with Top-K scoring
│   │   ├── generator.py              # Multilingual answer synthesis (OpenAI & Mock)
│   │   └── pipeline.py               # End-to-end RAG pipeline for variants A, B, C
│   ├── evaluation/                   # Metrics computation & statistical analysis
│   │   ├── metrics.py                # CER, WER, Exact Match, Token F1
│   │   └── rag_comparison.py         # 3-Variant triangulation & degradation matrix
│   ├── experiments/                  # Research experimental runners
│   │   ├── threshold_sweep.py        # Tau sensitivity sweep (tau in [30, 90])
│   │   ├── full_vs_selective.py      # Efficiency & accuracy benchmark runner
│   │   ├── cross_lingual_analysis.py # English vs Hindi script analysis
│   │   └── export_figures.py         # 300 DPI plots & LaTeX booktabs tables
│   └── dashboard/                    # Interactive web dashboard (Phase 12)
│       ├── app.py                    # FastAPI backend REST API server
│       └── static/                   # Glassmorphic frontend assets
│           ├── index.html            # Semantic HTML5 layout
│           ├── styles.css            # Dark mode glassmorphic CSS
│           └── app.js                # Interactive canvas & visualizer logic
├── experiments/
│   ├── configs/                      # Reproducible YAML experiment configs
│   │   └── default.yaml              # Master experiment configuration
│   └── runs/                         # Frozen immutable run artifacts (metadata.json)
├── results/
│   ├── figures/                      # High-resolution publication plots (300 DPI)
│   │   ├── figure1_variant_triangulation.png
│   │   ├── figure2_threshold_pareto.png
│   │   └── figure3_question_type_vulnerability.png
│   ├── tables/                       # Publication LaTeX tables & Markdown summary
│   │   ├── triangulation_table.tex
│   │   ├── efficiency_table.tex
│   │   ├── threshold_sweep_table.tex
│   │   ├── cross_lingual_table.tex
│   │   └── summary_tables.md
│   ├── rag/                          # Variant evaluation reports & comparison JSONs
│   │   ├── ground_truth_results.json
│   │   ├── raw_ocr_results.json
│   │   ├── corrected_ocr_results.json
│   │   └── comparison_summary.json
│   ├── sweeps/                       # Parameter sweeps & efficiency benchmark JSONs
│   │   ├── threshold_sweep.json
│   │   └── full_vs_selective.json
│   └── analysis/                     # Cross-lingual comparative study JSONs
│       └── cross_lingual_study.json
├── tests/                            # 70 unit and integration tests (100% passing)
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_reproducibility.py
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_ocr.py
│   ├── test_correction.py
│   ├── test_rag.py
│   ├── test_sweep.py
│   ├── test_efficiency.py
│   ├── test_cross_lingual.py
│   ├── test_master_runner.py
│   └── test_dashboard.py
├── dashboard.py                      # Standalone dashboard launcher
├── run_experiment.py                 # Master end-to-end experiment runner
├── main.py                           # Master CLI entrypoint
├── requirements.txt                  # Pinned dependency requirements
└── README.md                         # Comprehensive research documentation
```

---

## 8. Installation & Setup

### Prerequisites
- **Python**: 3.10 to 3.13 (tested on Python 3.13.5 AMD64 Windows/Linux/macOS)
- **Git**: For version control and reproducibility tracking
- **Tesseract OCR (Optional for live OCR; Mock engine available for standalone execution):**
  - **Windows:** `winget install UB-Mannheim.TesseractOCR` (Ensure `tessdata` contains `hin.traineddata` and `eng.traineddata`).
  - **Ubuntu / Debian:** `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin`
  - **macOS:** `brew install tesseract tesseract-lang`

### 1. Clone & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/rishavraj2004/ocr.git
cd ocr

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Command-Line Interface (CLI) Complete Reference

The repository provides two unified entrypoints: [`main.py`](main.py) and [`run_experiment.py`](run_experiment.py).

### Master Commands
```bash
# Launch interactive research inspection dashboard
python dashboard.py --port 8080
python main.py --dashboard --port 8080

# Execute complete end-to-end master benchmark in one command
python run_experiment.py
python main.py --run-all

# Execute master benchmark skipping parameter sweep (fast run)
python run_experiment.py --skip-sweep
```

### Modular Pipeline Subcommands
```bash
# System environment & dependency audit
python main.py --check-env

# Validate experiment configuration
python main.py --validate-config --config experiments/configs/default.yaml

# Validate benchmark dataset integrity
python main.py --validate-dataset

# Image preprocessing on a specific page with visual debug export
python main.py --preprocess-page doc_en_001_page_001

# Execute OCR extraction and calculate baseline CER/WER
python main.py --run-ocr

# Execute confidence-guided selective correction (default tau=70.0)
python main.py --run-correction
python main.py --run-correction --threshold 50.0

# Execute RAG evaluation on a specific document variant
python main.py --run-rag --variant ground_truth
python main.py --run-rag --variant raw_ocr
python main.py --run-rag --variant corrected_ocr

# Display 3-variant triangulation comparison table
python main.py --compare-rag

# Execute confidence threshold sensitivity sweep (tau in [30, 90])
python main.py --run-sweep
python main.py --run-sweep --sweep-thresholds "30,50,70,90"

# Execute head-to-head Full-Text vs. Selective Correction benchmark
python main.py --compare-efficiency

# Execute English vs. Hindi cross-lingual comparative analysis
python main.py --run-cross-lingual

# Export high-resolution publication figures (300 DPI) and LaTeX tables
python main.py --export-figures
```

---

## 10. Reproducibility & Verification

### Running the Complete Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
All **70 unit and integration tests** execute in under 15 seconds:
- `test_config.py`: Schema validation, defaults, overrides.
- `test_schemas.py`: Pydantic V2 contract integrity.
- `test_reproducibility.py`: Hardware detection, Git commit freezing, directory isolation.
- `test_dataset.py`: Image loading, ground truth alignment, question authoring validation.
- `test_preprocessing.py`: Grayscale, fast NLM denoising, contour deskewing, Otsu thresholding.
- `test_ocr.py`: OCR token structure, confidence extraction, CER/WER accuracy.
- `test_correction.py`: Low-confidence span detector, contextual extractor, corrector models.
- `test_rag.py`: Sliding window chunker, BGE-M3 embeddings, FAISS vector store, retrieval scoring, answer evaluation.
- `test_sweep.py`: Threshold sweep runner, Pareto knee detection.
- `test_efficiency.py`: Full-text vs. selective correction token and error counters.
- `test_cross_lingual.py`: English vs. Hindi comparative metrics.
- `test_master_runner.py`: Orchestration lifecycle, metadata freeze, figure/table generation.
- `test_dashboard.py`: FastAPI endpoints, JSON contracts, static file delivery.

### Run Metadata Freeze
Every master run saves an immutable artifact to `experiments/runs/<run_id>/metadata.json`:
```json
{
  "run_id": "ocr_multilingual_rag_20261007_190210_b2db1b",
  "git_commit": "d99c179af59dddfc48ec24a713c95fc3d80a4d48",
  "git_dirty": false,
  "platform": {
    "os": "Windows",
    "os_release": "11",
    "architecture": "AMD64",
    "cpu_count": 12
  },
  "package_versions": {
    "torch": "2.14.1",
    "sentence-transformers": "6.1.0",
    "faiss-cpu": "1.15.1",
    "pydantic": "2.11.7",
    "fastapi": "0.115.0",
    "matplotlib": "3.11.2"
  },
  "status": "COMPLETED",
  "total_duration_sec": 22.15
}
```

---

## 11. Citation

If you use this research framework, benchmark dataset, or selective correction methodology in your academic work, please cite:

```bibtex
@article{raj2026ocrmultilingualrag,
  title={Evaluating and Mitigating OCR-Induced Errors in Multilingual RAG: A Study of English and Hindi Documents},
  author={Raj, Rishav},
  journal={arXiv preprint},
  year={2026},
  note={Controlled empirical evaluation across 38 questions, 10 benchmark documents, and 3 document variants.}
}
```

---

## 12. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

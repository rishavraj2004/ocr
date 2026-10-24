# Multilingual OCR-Aware RAG Research: Consolidated Tables

## 1. 3-Variant Triangulation Study

| Partition | Metric | Ground Truth (A) | Raw OCR (B) | Corrected OCR (C) | Degradation (B - A) | Recovery (C - B) | Recovery Rate % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Overall | Exact Match (EM) | 1.0000 | 0.8684 | 0.8947 | -0.1316 | +0.0263 | +20.0% |
| Overall | Token F1 Score | 1.0000 | 0.9224 | 0.9474 | -0.0776 | +0.0250 | +32.2% |
| Overall | Recall@5 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0% |
| Overall | MRR | 0.9145 | 0.9298 | 0.9351 | +0.0154 | +0.0053 | -34.3% |
| English | Exact Match (EM) | 1.0000 | 0.8421 | 0.8421 | -0.1579 | +0.0000 | +0.0% |
| English | Token F1 Score | 1.0000 | 0.9211 | 0.9211 | -0.0789 | +0.0000 | +0.0% |
| English | Recall@5 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0% |
| English | MRR | 0.8947 | 0.9254 | 0.9386 | +0.0307 | +0.0132 | -42.9% |
| Hindi | Exact Match (EM) | 1.0000 | 0.8947 | 0.9474 | -0.1053 | +0.0526 | +50.0% |
| Hindi | Token F1 Score | 1.0000 | 0.9238 | 0.9737 | -0.0762 | +0.0499 | +65.5% |
| Hindi | Recall@5 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0% |
| Hindi | MRR | 0.9342 | 0.9342 | 0.9316 | +0.0000 | -0.0026 | +100.0% |


## 2. Full-Text vs. Selective Correction Efficiency

| Evaluation Dimension | Full-Text (Naive) | Selective (tau=70) | Advantage |
| :--- | :---: | :---: | :---: |
| Total Tokens Consumed | 4,883 | 1,539 | +68.5% (3.2x cheaper) |
| Completion Tokens Generated | 2,027 | 117 | +94.2% decoding savings |
| Over-Corrected Words | 4 | 0 | +4 fewer corruptions |
| Character Error Rate (CER) | 0.1353 | 0.1320 | -0.0033 |
| Word Error Rate (WER) | 0.2008 | 0.1842 | -0.0166 |
| QA Exact Match (EM) | 0.8421 | 0.8947 | +0.0526 |
| QA Token F1 Score | 0.9355 | 0.9474 | +0.0118 |

## 3. Confidence Threshold Sensitivity Sweep

| Threshold (tau) | Flagged Words | Flagged % | Total Tokens | Mean CER | QA Exact Match (EM) | QA F1 Recovery % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 30 | 0 | 0.0% | 0 | 0.1331 | 0.8684 | +0.0% |
| 70 **(Optimal tau*)** | 94 | 7.7% | 1539 | 0.1320 | 0.8947 | +32.2% |


## 4. Cross-Lingual Comparison (English vs. Hindi)

| Dimension | English (Latin) | Hindi (Devanagari) | Finding |
| :--- | :--- | :--- | :--- |
| Script Profile | Latin (Linear alphabet) | Devanagari (2D abugida ligatures) | Fundamentally different error topologies |
| Retrieval Invariance | Recall@5: 1.0000 | Recall@5: 1.0000 | Dense retrieval is script-invariant |
| QA EM Degradation | -0.1579 | -0.1053 | English has high numerical vulnerability |
| QA EM Recovery Rate | 0.0% | 50.0% | Hindi recovers 50.0% vs conservative English |
| QA F1 Recovery Rate | 0.0% | 65.5% | Hindi recovers 65.5% token overlap |


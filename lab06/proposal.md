# Individual Project Proposal: RAG Document Chunking Optimization

## Student Information
- **Name:** Shcherbakov Oleh
- **Student ID:** 71663
- **Course:** Machine Learning Systems Engineering

---

## 1. Problem Statement
Evaluating chunking and retrieval strategies for Retrieval-Augmented Generation (RAG) over long technical PDFs containing complex tables and figures. The goal is to maximize factual faithfulness while minimizing hallucination rates during automated data extraction.

## 2. Dataset
- **Source:** Technical PDF documentation, reports, and manuals containing mixed text, tabular structures, and graphical captions.
- **Preprocessing:** Document parsing, table text extraction, and semantic chunking with configurable overlap boundaries.

## 3. Success Metric
- **Retrieval Recall@K** and **Faithfulness Score** (evaluated via LLM-as-a-judge and ground-truth validation sets).

## 4. Deployment Constraints
- **Inference Latency:** $\le 50\text{ ms}$ per retrieval query.
- **Model / Artifact Size:** $\le 5\text{ MB}$ for embedding and retrieval scoring modules.

## 5. Planned Optimization Techniques
- **Quantization:** Post-training quantization (PTQ) on embedding index weights to reduce memory footprint.
- **Pruning:** Removing redundant embedding dimensions to satisfy latency and size boundaries.
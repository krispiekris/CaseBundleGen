# CaseBundleGen: A Multimodal RAG Benchmark Generator

A research framework for generating synthetic case bundles and evaluating Retrieval-Augmented Generation (RAG) systems. This project generates realistic case documents and systematically benchmarks RAG model performance using BERTScore.

## Overview

This framework consists of three main components:

1. **Case Generation**: Creates diverse, realistic synthetic case bundles with structured documents
2. **RAG Evaluation**: Sets up and evaluates RAG systems using llama-index with Ollama integration
3. **Benchmarking**: Measures RAG performance using BERTScore, a state-of-the-art semantic similarity metric

## Prerequisites

- Python 3.10 or newer
- Ollama CLI installed and available on `PATH` (tested with Ollama `0.18.1`)
- HuggingFace models (automatically downloaded on first use)
- GPU recommended for embedding and benchmarking tasks

## Install Ollama (v0.18.1)

This project has been tested with **Ollama 0.18.1**.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.18.1 sh
```

### macOS

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.18.1 sh
```

### Windows (PowerShell)

```powershell
$env:OLLAMA_VERSION="0.18.1"
irm https://ollama.com/install.ps1 | iex
```

### Verify installation

```bash
ollama --version
```

Expected output:

```text
ollama version 0.18.1
```

## Quick Start

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Cases

By default, this script runs the `gemma3:12b` model and generates 100 synthetic cases:

```bash
python generate_case.py
```

Cases are stored in the `cases/` directory with the structure:
- `cases/case_001/`, `cases/case_002/`, etc.
- Each case contains multiple document types (emails, bank statements, reports, etc.)
- Includes `metadata.json` and `qa.json` for evaluation

### Evaluate RAG Systems

Open and run the Jupyter notebook in the `Evaluation/` folder:

```bash
jupyter notebook Evaluation/evaluation.ipynb
```

The notebook:
- Preprocesses case files and moves Q&A pairs to a dedicated folder
- Builds vector indices using HuggingFace embeddings (BAAI/bge-base-en-v1.5)
- Queries RAG systems using Ollama-powered LLMs
- Records predicted answers and metrics for benchmarking

### Benchmark with BERTScore

Once evaluation results are generated, use BERTScore for robust semantic evaluation:

```bash
cd bertscore
python run_bertscore_eval.py --input case_benchmark_results_bertscore_input.csv
```

**BERTScore** is the primary benchmarking tool used to evaluate RAG answer quality by measuring contextual similarity between predicted and reference answers. Unlike simple string matching, BERTScore captures semantic meaning and is more aligned with human judgment.

## Project Structure

```
├── generate_case.py              # Main pipeline for synthetic case generation
├── ollama_utils.py               # Utilities for Ollama integration
├── cases/                        # Generated synthetic case bundles
├── Evaluation/
│   ├── evaluation.ipynb          # Jupyter notebook for RAG setup and evaluation
│   ├── cases/                    # Symbolic links to generated cases
│   └── quesNAs/                  # Q&A pairs extracted from cases
├── bertscore/
│   ├── run_bertscore_eval.py     # BERTScore evaluation script
│   ├── trim_bertscore_columns.py # Utility for result processing
│   └── case_benchmark_results_*.csv # Benchmark results at various stages
├── prompts/                      # Prompt templates for case generation
├── samples/                      # Sample documents for style reference
└── README.md                     # This file
```

## Key Features

- **Ollama-Compatible**: Leverages Ollama for local LLM inference without external API dependencies
- **Configurable Case Generation**: Adjust the number of cases by modifying the batch size in `generate_case.py`
- **Semantic Evaluation**: Uses BERTScore for evaluation beyond lexical metrics
- **Reproducibility**: All components designed for research reproducibility and transparency

## Configuration

Environment variables for evaluation:

```bash
EMBED_DEVICE=cpu              # Embedding device (cpu, cuda, etc.)
EMBED_BATCH_SIZE=8            # Batch size for embeddings
INDEX_INSERT_BATCH_SIZE=16    # Batch size for index insertion
```

## Notes

- This repository is designed for research and experimental evaluation purposes
- Case generation times depend on model performance and available compute resources
- BERTScore computation can be memory-intensive for large batches
- For reviewers: Reduce the number of cases in `generate_case.py` for faster iteration
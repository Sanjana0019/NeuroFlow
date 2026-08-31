# NeuroFlow — End-to-End RAG Lifecycle & Optimization Platform

<div align="center">

![NeuroFlow Banner](https://img.shields.io/badge/NeuroFlow-RAG%20Lifecycle%20Platform-6366f1?style=for-the-badge&logo=cpu&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14%20App%20Router-black?style=for-the-badge&logo=next.js&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7%20Pub%2FSub-dc382d?style=for-the-badge&logo=redis&logoColor=white)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Observability-f5a800?style=for-the-badge&logo=opentelemetry&logoColor=white)

**Turn experimental retrieval into observable, measurable, and self-improving intelligence.**

[Overview](#overview) • [Key Features](#key-features) • [Architecture](#architecture) • [Tech Stack](#tech-stack) • [Quickstart](#quickstart) • [API Reference](#api-reference) • [Builder](#builder)

</div>

---

## Overview

Most Retrieval-Augmented Generation (RAG) implementations stop at connecting a document parser to a vector database. In production, teams face a black box: when answers degrade or hallucinate, there is zero visibility into whether chunking, vector embeddings, BM25 keyword rankings, rerankers, or prompt templates caused the failure.

**NeuroFlow** bridges the gap between simple prompt scripts and production AI infrastructure. It provides a complete, observable RAG lifecycle platform:
1. **Multi-Source Ingestion**: High-throughput parsing across PDF, DOCX, CSV, PPTX, OCR images, and web URLs.
2. **Hybrid Retrieval & Reranking**: Reciprocal Rank Fusion of dense semantic search (`pgvector`) and sparse keyword frequency (`BM25`), followed by cross-encoder reranking.
3. **Real-Time Streaming with Citations**: Token-by-token Server-Sent Events (SSE) with deterministic chunk-level citation attribution and source inspection.
4. **Automated LLM-as-Judge Evaluation**: Asynchronous background evaluation scoring Faithfulness, Answer Relevance, Context Precision, and Context Recall over Redis pub/sub.
5. **A/B Pipeline Benchmarking**: Side-by-side comparison across pipeline versions with live latency and score diffs.
6. **Closed-Loop Fine-Tuning Curation**: Automated extraction and validation (PII, length, citations) of top-scoring pairs for SFT and DPO JSONL export.

---

## Key Features

### 1. Multi-Format Ingestion Engine
* Extract and normalize documents across **PDF**, **DOCX**, **CSV**, **PPTX**, **OCR image scans** (Tesseract), and **Web URLs**.
* Configurable chunking strategies (recursive character splitting, semantic boundary detection) with configurable token overlap.
* Dual indexing: stores high-dimensional dense embeddings (`text-embedding-3-small` / open-source equivalents) alongside BM25 sparse keyword indices.

### 2. Hybrid Retrieval & Cross-Encoder Reranking
* Dense vector similarity via **PostgreSQL `pgvector`** for semantic conceptual matching.
* Sparse keyword frequency via **BM25** for exact terminology, acronyms, and product IDs.
* Reciprocal Rank Fusion (RRF) and **cross-encoder reranking** (`bge-reranker-large`) to guarantee high-signal context before generation.

### 3. Streaming Generation & Inline Attribution
* Real-time Server-Sent Events (SSE) streaming.
* Deterministic chunk-level citation tags (`[1]`, `[2]`) mapped directly to retrieved chunks.
* Interactive slide-over citation drawer showing source file, chunk text, and similarity confidence scores.

### 4. Automated LLM-as-Judge Evaluation
* Every user interaction is asynchronously evaluated across 4 core dimensions:
  * **Faithfulness**: Are claims strictly supported by retrieved context?
  * **Answer Relevance**: Does the response directly address the user query?
  * **Context Precision**: Are high-relevance chunks ranked at the top?
  * **Context Recall**: Was all necessary reference ground truth retrieved?
* Real-time Redis pub/sub streaming directly to the **Live Evaluation Feed** without adding latency to user responses.

### 5. Stateful Pipeline Management & A/B Comparison
* Create, configure, version, and clone independent RAG pipelines.
* Live 7-day query volume sparklines, P50/P95 latency percentiles, and quality distributions from PostgreSQL.
* Side-by-side **Compare Mode** to evaluate two pipeline configurations on identical prompts simultaneously.

### 6. Closed-Loop Dataset Curation (SFT & DPO)
* Automated candidate extraction for pairs scoring $\ge 80\%$ quality.
* Strict automated validation checklist: PII removal, citation verification, response token range (50–2,000 tokens).
* One-click dataset export to **SFT JSONL** (ChatML format) and **DPO JSONL** (prompt, chosen, rejected) for fine-tuning.

---

## Architecture

```
                                  ┌───────────────────────────────┐
                                  │      Next.js 14 Frontend      │
                                  │  (Playground, Analytics, SSE) │
                                  └───────────────┬───────────────┘
                                                  │ HTTP / SSE
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Backend Runtime                                    │
│                                                                                                  │
│   ┌────────────────────┐    ┌───────────────────────────────┐    ┌───────────────────────────┐   │
│   │  Ingestion Engine  │    │      Retrieval & Rerank       │    │    Streaming Generator    │   │
│   │ (PDF,DOCX,OCR,URL) │───▶│  (pgvector + BM25 + CrossEnc) │───▶│  (OpenAI, Anthropic, OR)  │   │
│   └────────────────────┘    └───────────────────────────────┘    └─────────────┬─────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┼─────────────────┘
                                                                                 │
                                                    ┌────────────────────────────┴─────────────────┐
                                                    ▼                                              ▼
                                     ┌─────────────────────────────┐                ┌──────────────────────────────┐
                                     │  PostgreSQL 16 + pgvector   │                │     Redis 7 + ARQ Worker     │
                                     │  • Documents & Chunks       │                │  • Asynchronous LLM Judge    │
                                     │  • Pipelines & Versions     │                │  • Pub/Sub Telemetry Stream  │
                                     │  • Query Runs & Evals       │                │  • SFT / DPO Curation Job    │
                                     └─────────────────────────────┘                └──────────────────────────────┘
```

---

## Tech Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 14 (App Router), React, TypeScript | Type-safe interactive dashboard, telemetry visualizations, Monaco editor |
| **Styling** | Tailwind CSS, Lucide Icons | Responsive modern dark-theme design system (`#0b0f19`) |
| **Backend API** | Python 3.11+, FastAPI, Pydantic V2 | High-throughput asynchronous REST & SSE streaming server |
| **Relational & Vector** | PostgreSQL 16 + pgvector | ACID relational persistence & high-dimensional vector search |
| **Keyword Search** | BM25 Engine | Sparse keyword frequency scoring & term matching |
| **Message Queue** | Redis 7 + ARQ Distributed Workers | Non-blocking background evaluation and dataset export jobs |
| **Observability** | OpenTelemetry, Prometheus, Jaeger, MLflow | Distributed tracing, latency percentiles, and model run tracking |
| **Containerization** | Docker, Docker Compose | Reproducible local and production infrastructure orchestration |

---

## Quickstart

### Prerequisites
* Docker & Docker Desktop
* Python 3.11+
* Node.js 18+ & npm

### 1. Clone the Repository
```bash
git clone https://github.com/Sanjana0019/NeuroFlow.git
cd NeuroFlow
```

### 2. Start Infrastructure (PostgreSQL + pgvector, Redis, MLflow)
```bash
docker compose -f infra/docker-compose.yml up -d
```

### 3. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run FastAPI backend server (Port 8000)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Subsystem health checks (Postgres, Redis, MLflow, Circuit Breakers) |
| `POST` | `/query` | Stream or execute RAG retrieval and generation with citations |
| `POST` | `/compare` | Execute simultaneous side-by-side query against two pipelines |
| `GET` | `/pipelines` | List all pipelines with 7-day query volume and latency percentiles |
| `POST` | `/pipelines` | Create or version an existing RAG pipeline configuration |
| `GET` | `/evaluations` | List recent evaluation runs and 4-metric judge scorecards |
| `GET` | `/evaluations/stream` | Server-Sent Events stream for real-time evaluation updates |
| `POST` | `/ingest` | Upload and parse multi-format documents into vector & BM25 indices |
| `GET` | `/finetune/readiness` | Validate dataset size, PII check, and eligibility counts |
| `GET` | `/finetune/datasets/export` | Download validated SFT or DPO datasets in JSONL format |

---

## Builder

**Sanjana Patil**  
*Full-Stack & AI Systems Engineer · Creator of NeuroFlow*

* **GitHub**: [@Sanjana0019](https://github.com/Sanjana0019)
* **LinkedIn**: [sanjana-patil-dev](https://www.linkedin.com/in/sanjana-patil-dev/)
* **X (Twitter)**: [@sanjana_p0019](https://x.com/sanjana_p0019)

---

<div align="center">
  <sub>NeuroFlow is built with open standards for observable, measurable retrieval intelligence.</sub>
</div>

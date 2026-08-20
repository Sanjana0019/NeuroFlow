# NeuroFlow System Architecture

## 1. Overview

NeuroFlow is an AI platform that allows organizations to ingest internal data, retrieve relevant information, generate answers using large language models, evaluate answer quality, and prepare high-quality examples for potential fine-tuning.

The system is organized into five major subsystems:

1. Ingestion
2. Retrieval
3. Generation
4. Evaluation
5. Fine-Tuning

A backend API provides a controlled interface between the frontend and these subsystems.

---

## 2. High-Level Architecture

```text
                         NEUROFLOW
                            │
                     ┌──────┴──────┐
                     │   Frontend  │
                     └──────┬──────┘
                            │
                            ↓
                     ┌──────────────┐
                     │  Backend API │
                     └──────┬───────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ↓                 ↓                 ↓
      Ingestion         Retrieval        Generation
          │                 │                 │
          ↓                 ↓                 ↓
      Processing       Ranked Context       LLM
          │                 │                 │
          └──────────┐      │      ┌──────────┘
                     ↓      ↓      ↓
                PostgreSQL + pgvector
                            │
                            ↓
                       Evaluation
                            │
                            ↓
                       Fine-Tuning
```

The architecture separates data preparation, information retrieval, answer generation, quality evaluation, and model adaptation into distinct subsystems.

---

## 3. Ingestion Subsystem

### Purpose

The ingestion subsystem converts raw organizational data into searchable information.

Supported sources may include:

* PDF files
* DOCX files
* Images
* CSV files
* Web URLs

### Flow

```text
Raw File / URL
      ↓
Content Extraction
      ↓
Chunking
      ↓
Metadata
      ↓
Embedding
      ↓
PostgreSQL + pgvector
```

### Responsibilities

* Accept documents and URLs.
* Extract usable content.
* Split content into smaller chunks.
* Attach metadata such as document name, page, and source.
* Generate embeddings.
* Store chunks, metadata, and vectors.
* Detect and prevent duplicate ingestion where applicable.

The output of ingestion is searchable data that can later be consumed by the retrieval subsystem.

---

## 4. Retrieval Subsystem

### Purpose

The retrieval subsystem finds the most relevant information for a user's question.

### Flow

```text
User Question
      ↓
 ┌────┼──────────────┐
 ↓    ↓              ↓
Vector Keyword  Metadata
Search  Search   Filtering
 ↓       ↓           ↓
 └───────┼───────────┘
         ↓
     RRF Fusion
         ↓
Candidate Results
         ↓
Cross-Encoder Reranker
         ↓
Ranked Context
```

### Responsibilities

* Convert the query into an embedding.
* Perform semantic vector search.
* Perform keyword-based search.
* Apply metadata filters where required.
* Combine retrieval rankings using Reciprocal Rank Fusion (RRF).
* Rerank candidate results using a cross-encoder.
* Return the highest-quality context to the generation subsystem.

---

## 5. Generation Subsystem

### Purpose

The generation subsystem uses the user's question and retrieved context to generate an answer.

### Flow

```text
User Question
      +
Retrieved Context
      ↓
Prompt Assembly
      ↓
LLM Router
      ↓
Selected LLM
      ↓
Streaming Response
      ↓
Answer
```

### Responsibilities

* Assemble the user question and retrieved context into a prompt.
* Select an appropriate model through the LLM router.
* Send the request to the selected LLM.
* Stream the generated response when supported.
* Record relevant generation information for evaluation and observability.

The generation subsystem should use retrieved context to ground responses and reduce unsupported answers.

---

## 6. Evaluation Subsystem

### Purpose

The evaluation subsystem measures the quality of generated answers and retrieved context.

### Flow

```text
Question
   +
Retrieved Context
   +
Generated Answer
        ↓
     Evaluator
        ↓
 Quality Metrics
        ↓
     Store Scores
```

### Metrics

The system can evaluate:

* Faithfulness
* Answer Relevance
* Context Precision
* Context Recall

An LLM-as-a-Judge approach may be used together with evaluation frameworks such as RAGAS.

Evaluation results are stored so that system quality can be monitored over time and different versions of the system can be compared.

Evaluation scores are treated as quality signals rather than absolute truth.

---

## 7. Fine-Tuning Subsystem

### Purpose

The fine-tuning subsystem prepares high-quality examples and uses them to adapt model behavior when fine-tuning is appropriate.

### Flow

```text
High-Quality Examples
        ↓
Example Extraction
        ↓
JSONL Dataset
        ↓
Fine-Tuning Job
        ↓
Fine-Tuned Model
        ↓
MLflow Tracking
        ↓
Model Management
        ↓
LLM Router
```

### Responsibilities

* Identify high-quality examples.
* Prepare training datasets in JSONL format.
* Create and track fine-tuning jobs.
* Track experiments and model artifacts.
* Register or manage resulting models.
* Make suitable fine-tuned models available to the LLM routing layer.

Fine-tuning is treated as a separate model adaptation process rather than a replacement for retrieval.

---

## 8. Data Storage

PostgreSQL is used as the primary relational data store.

The pgvector extension allows vector embeddings to be stored and searched within PostgreSQL.

The database may contain:

* Documents
* Chunks
* Embeddings
* Metadata
* Queries
* Evaluation results
* Pipeline information
* Fine-tuning job information

The exact schema will be defined during implementation.

---

## 9. Backend API Layer

The backend API acts as the controlled interface between the frontend and NeuroFlow's internal services.

```text
Frontend
   ↓
Backend API
   ↓
Application Services
   ↓
Database / AI Subsystems
```

The API layer is responsible for:

* Request validation
* Authentication
* Authorization
* Rate limiting
* Routing requests
* Error handling
* Returning structured responses

The frontend should not access the database directly.

---

## 10. Security and Reliability

NeuroFlow is designed with production-oriented concerns in mind.

Key areas include:

* Authentication and authorization
* Rate limiting
* Input validation
* Secure handling of uploaded files
* Error handling
* Logging
* Monitoring
* Health checks
* Metrics
* Automated testing

Security and reliability mechanisms will be implemented incrementally as the project moves from architecture to implementation.

---

## 11. End-to-End Data Flow

### Ingestion Flow

```text
File / URL
    ↓
Backend API
    ↓
Ingestion
    ↓
Extraction
    ↓
Chunking
    ↓
Embedding
    ↓
PostgreSQL + pgvector
```

### Question Answering Flow

```text
User Question
    ↓
Backend API
    ↓
Retrieval
    ↓
Relevant Context
    ↓
Generation
    ↓
LLM
    ↓
Answer
    ↓
Evaluation
    ↓
Quality Scores
```

### Fine-Tuning Flow

```text
High-Quality Interactions
    ↓
Example Extraction
    ↓
JSONL Dataset
    ↓
Fine-Tuning Job
    ↓
Fine-Tuned Model
    ↓
Model Tracking
    ↓
LLM Router
```

---

## 12. Design Principles

The NeuroFlow architecture follows these principles:

1. **Separation of responsibilities** — each subsystem has a clear purpose.
2. **API-first communication** — frontend communication occurs through backend APIs.
3. **Retrieval-grounded generation** — generated answers are supported by retrieved context.
4. **Evaluation-driven improvement** — system changes can be measured using evaluation results.
5. **Model flexibility** — the LLM router allows different models to be used based on application needs.
6. **Production awareness** — security, observability, testing, and reliability are considered from the beginning.

# API Contracts

## POST `/ingest`

### Purpose

Accept company data such as a file or URL and start the ingestion pipeline.

The ingestion pipeline extracts the content, chunks it, generates embeddings, and stores the processed data for later retrieval.

### Request

The client sends either:

* a file to be ingested, or
* a URL to be ingested

Optional metadata may also be provided.

Example:

```json
{
  "source_type": "url",
  "source": "https://example.com/company-document",
  "metadata": {
    "document_name": "Company Policy"
  }
}
```

### Response

If the request is accepted, the API returns a document identifier and processing status.

Example:

```json
{
  "document_id": "doc_12345",
  "status": "processing"
}
```

The `document_id` allows the client to identify the document while processing continues.

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `400`  | Invalid request                    |
| `401`  | Authentication required or invalid |
| `413`  | File is too large                  |
| `415`  | Unsupported file type              |
| `429`  | Rate limit exceeded                |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

The client must provide valid credentials before starting an ingestion request.

### Rate Limits

Ingestion requests are rate-limited per authenticated user to prevent excessive resource consumption.

## POST `/query`

### Purpose

Accept a user's question, retrieve relevant information from the knowledge base, and generate an answer using an LLM.

### Request

The client provides the user's question.

Example:

```json
{
  "query": "What is our company's leave policy?"
}
```

Optional parameters may be added later for model selection, retrieval configuration, or conversation context.

### Response

The API returns the generated answer and an identifier for the query.

Example:

```json
{
  "query_id": "query_12345",
  "answer": "Employees are entitled to 20 days of annual leave per year.",
  "status": "completed"
}
```

The `query_id` can be used to identify and track the query.

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `400`  | Invalid or empty query             |
| `401`  | Authentication required or invalid |
| `429`  | Rate limit exceeded                |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Query requests are rate-limited per authenticated user to control resource usage and prevent abuse.

## GET `/query/{query_id}/stream`

### Purpose

Stream the generated answer for a previously submitted query.

The endpoint allows the client to receive the LLM response progressively instead of waiting for the complete answer.

### Request

The client provides the `query_id` as a path parameter.

Example:

```text
GET /query/query_12345/stream
```

### Response

The server streams the generated answer progressively to the client.

Example stream:

```text
"Employees "
"are entitled "
"to 20 days "
"of annual leave."
```

The exact streaming protocol will be defined during implementation.

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `401`  | Authentication required or invalid |
| `404`  | Query not found                    |
| `429`  | Rate limit exceeded                |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Streaming requests are rate-limited per authenticated user to prevent excessive resource consumption.
## GET `/evaluations`

### Purpose

Retrieve evaluation results for previously generated answers.

The results allow NeuroFlow to inspect the quality of individual RAG responses.

### Request

No request body is required.

Optional query parameters may be added later for filtering by date, query, user, model, or evaluation metric.

Example:

```text
GET /evaluations
```

### Response

Returns a list of evaluation results.

Example:

```json
{
  "evaluations": [
    {
      "evaluation_id": "eval_001",
      "query_id": "query_12345",
      "faithfulness": 0.91,
      "answer_relevance": 0.88,
      "context_precision": 0.84,
      "context_recall": 0.80
    }
  ]
}
```

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `401`  | Authentication required or invalid |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Requests are rate-limited per authenticated user.

---

## GET `/evaluations/aggregate`

### Purpose

Retrieve aggregated evaluation statistics across multiple evaluated responses.

This endpoint provides an overall view of NeuroFlow's answer quality.

### Request

No request body is required.

Example:

```text
GET /evaluations/aggregate
```

### Response

Returns aggregated evaluation metrics.

Example:

```json
{
  "total_evaluations": 100,
  "average_scores": {
    "faithfulness": 0.90,
    "answer_relevance": 0.87,
    "context_precision": 0.82,
    "context_recall": 0.79
  }
}
```

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `401`  | Authentication required or invalid |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Requests are rate-limited per authenticated user.

## POST `/pipelines`

### Purpose

Create a pipeline configuration that defines a sequence of processing steps used by NeuroFlow.

### Request

The client provides the pipeline name and configuration.

Example:

```json id="t8oh6g"
{
  "name": "default-rag-pipeline",
  "steps": [
    "retrieval",
    "reranking",
    "generation",
    "evaluation"
  ]
}
```

### Response

Returns the newly created pipeline identifier and its status.

Example:

```json id="m0b3sp"
{
  "pipeline_id": "pipeline_001",
  "name": "default-rag-pipeline",
  "status": "created"
}
```

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `400`  | Invalid pipeline configuration     |
| `401`  | Authentication required or invalid |
| `409`  | Pipeline already exists            |
| `429`  | Rate limit exceeded                |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Pipeline creation requests are rate-limited per authenticated user.

---

## GET `/pipelines/{id}/runs`

### Purpose

Retrieve execution history for a specific pipeline.

A pipeline can be executed many times, so this endpoint allows NeuroFlow to inspect previous runs and their statuses.

### Request

The client provides the pipeline identifier as a path parameter.

Example:

```text id="f4zv0v"
GET /pipelines/pipeline_001/runs
```

### Response

Returns a list of pipeline runs.

Example:

```json id="9y8g9d"
{
  "pipeline_id": "pipeline_001",
  "runs": [
    {
      "run_id": "run_001",
      "status": "completed",
      "started_at": "2026-08-11T10:00:00Z",
      "completed_at": "2026-08-11T10:02:15Z"
    },
    {
      "run_id": "run_002",
      "status": "running",
      "started_at": "2026-08-11T11:00:00Z",
      "completed_at": null
    }
  ]
}
```

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `401`  | Authentication required or invalid |
| `404`  | Pipeline not found                 |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Requests are rate-limited per authenticated user.

## POST `/finetune/jobs`

### Purpose

Create and start a fine-tuning job using a prepared high-quality dataset.

### Request

The client provides the dataset and fine-tuning configuration.

Example:

```json id="g6y2o4"
{
  "dataset_id": "dataset_001",
  "base_model": "base-model",
  "configuration": {
    "epochs": 3
  }
}
```

### Response

Returns the fine-tuning job identifier and initial status.

Example:

```json id="v9c8f1"
{
  "job_id": "ft_job_001",
  "status": "queued"
}
```

A job may move through states such as `queued`, `running`, `completed`, or `failed`.

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `400`  | Invalid fine-tuning configuration  |
| `401`  | Authentication required or invalid |
| `404`  | Dataset not found                  |
| `429`  | Rate limit exceeded                |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Fine-tuning job creation is rate-limited per authenticated user.

---

## GET `/finetune/jobs/{id}`

### Purpose

Retrieve the current status and details of a fine-tuning job.

### Request

The client provides the fine-tuning job identifier as a path parameter.

Example:

```text id="hj3y9s"
GET /finetune/jobs/ft_job_001
```

### Response

Returns the job status and relevant details.

Example:

```json id="r5w2kp"
{
  "job_id": "ft_job_001",
  "status": "completed",
  "base_model": "base-model",
  "model_id": "model_001"
}
```

The `model_id` is returned when a fine-tuning job successfully produces a model.

### Errors

| Status | Meaning                            |
| ------ | ---------------------------------- |
| `401`  | Authentication required or invalid |
| `404`  | Fine-tuning job not found          |
| `500`  | Internal server error              |

### Authentication

Authentication is required.

### Rate Limits

Requests are rate-limited per authenticated user.

## GET `/health`

### Purpose

Check whether the NeuroFlow backend is running and available.

### Request

No request body is required.

Example:

```text id="lqz8ne"
GET /health
```

### Response

Returns the current health status of the application.

Example:

```json id="3v6j6h"
{
  "status": "healthy"
}
```

### Errors

| Status | Meaning             |
| ------ | ------------------- |
| `503`  | Service unavailable |

### Authentication

Authentication is not required.

### Rate Limits

The endpoint should be protected from excessive requests but may be accessible without user authentication.

---

## GET `/metrics`

### Purpose

Expose application and system metrics for monitoring and observability.

### Request

No request body is required.

Example:

```text id="lqj5fw"
GET /metrics
```

### Response

Returns metrics collected by the NeuroFlow backend.

Example:

```text id="9a0r1e"
requests_total 1245
errors_total 12
request_latency_seconds 0.23
```

The exact metric format and names will be finalized during implementation.

### Errors

| Status | Meaning               |
| ------ | --------------------- |
| `500`  | Internal server error |

### Authentication

Access to metrics should be restricted in production because metrics can expose internal system information.

### Rate Limits

Metrics access should be protected from excessive requests.

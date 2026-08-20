# ADR 001: Vector Store

## Context

NeuroFlow needs to store document chunks, metadata, embeddings, queries, evaluation results, and other application data.

The retrieval subsystem also needs to perform vector similarity search over document embeddings.

Using separate databases for relational data and vector data would increase system complexity for the initial project.

## Decision

We will use PostgreSQL with the pgvector extension as the primary data store and vector search system.

PostgreSQL will store application data such as documents, chunks, metadata, queries, evaluations, pipeline information, and fine-tuning job information.

The pgvector extension will store and search embedding vectors for semantic retrieval.

## Consequences

### Positive

* Relational and vector data can be stored in one database.
* The architecture is simpler for the initial project.
* PostgreSQL is mature and widely used.
* pgvector supports vector similarity search.
* Fewer infrastructure components are required to develop and maintain locally.

### Negative

* A specialized vector database may provide additional vector-search capabilities at larger scale.
* Vector search performance may require additional optimization as the dataset grows.

### Future Consideration

If NeuroFlow grows to a scale where PostgreSQL and pgvector are no longer sufficient, a specialized vector database can be evaluated as an alternative.

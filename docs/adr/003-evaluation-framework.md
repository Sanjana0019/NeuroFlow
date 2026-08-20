# ADR 003: Evaluation Framework

## Context

NeuroFlow needs to evaluate the quality of generated answers and retrieved context.

Manual evaluation alone is difficult to scale and makes it harder to compare different versions of the retrieval and generation system.

The evaluation system should measure important RAG quality characteristics such as:

* Faithfulness
* Answer Relevance
* Context Precision
* Context Recall

The framework should also allow evaluation results to be stored and compared over time.

## Decision

NeuroFlow will use RAGAS as the primary evaluation framework for RAG-specific evaluation.

The evaluation subsystem will remain modular so that additional evaluation metrics or evaluation methods can be introduced later.

Evaluation results will be stored in the application's database and associated with the corresponding query, retrieved context, and generated answer where appropriate.

LLM-as-a-Judge may be used for metrics that require language-model-based evaluation.

Evaluation scores will be treated as quality signals rather than absolute truth.

## Consequences

### Positive

* Provides established metrics for evaluating RAG systems.
* Makes evaluation more systematic than manual inspection alone.
* Allows different versions of the system to be compared.
* Supports automated evaluation pipelines.
* A modular design allows additional metrics to be introduced later.

### Negative

* LLM-based evaluation can itself contain errors or biases.
* Evaluation quality depends on the evaluator model and evaluation configuration.
* Some metrics may require additional computation and increase processing time.
* Evaluation scores do not perfectly represent human judgment.

### Future Consideration

The evaluation framework can be extended with custom metrics, human evaluation, regression testing, and model-specific evaluation strategies as NeuroFlow evolves.

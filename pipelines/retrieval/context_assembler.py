import tiktoken
from pipelines.retrieval.models import AssembledContext, RetrievalResult


class ContextAssembler:
    """Assembles and formats retrieved chunks into context respecting a strict token budget."""

    def __init__(self, token_budget: int = 4000, model_encoding: str = "cl100k_base"):
        self.token_budget = token_budget
        try:
            self.tokenizer = tiktoken.get_encoding(model_encoding)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count exact tokens in text string."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def format_chunk(self, source_idx: int, chunk: RetrievalResult) -> str:
        """Format an individual chunk with source attribution header."""
        doc_name = chunk.filename or "document"
        page_info = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        header = f"[Source {source_idx} — {doc_name}{page_info}]"
        return f"{header}\n{chunk.content.strip()}"

    def assemble(
        self,
        chunks: list[RetrievalResult],
        custom_budget: int | None = None,
    ) -> AssembledContext:
        """Assemble top chunks into context without exceeding the token limit or splitting chunks."""
        budget = custom_budget if custom_budget is not None else self.token_budget

        if not chunks or budget <= 0:
            return AssembledContext(
                context="",
                chunks_used=[],
                total_tokens=0,
                sources=[],
            )

        selected_blocks: list[str] = []
        chunks_used: list[RetrievalResult] = []
        sources: list[dict] = []
        current_tokens = 0

        for idx, chunk in enumerate(chunks, start=1):
            formatted_block = self.format_chunk(idx, chunk)
            block_tokens = self.count_tokens(formatted_block)

            # Account for separator tokens "\n\n" between blocks
            separator_tokens = self.count_tokens("\n\n") if selected_blocks else 0
            candidate_total = current_tokens + separator_tokens + block_tokens

            if candidate_total > budget:
                # Stop before adding chunk that would exceed token budget; do not split mid-chunk
                break

            selected_blocks.append(formatted_block)
            chunks_used.append(chunk)
            current_tokens = candidate_total
            sources.append(
                {
                    "source_index": idx,
                    "document_id": str(chunk.document_id),
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
            )

        final_context = "\n\n".join(selected_blocks)
        exact_tokens = self.count_tokens(final_context)

        return AssembledContext(
            context=final_context,
            chunks_used=chunks_used,
            total_tokens=exact_tokens,
            sources=sources,
        )

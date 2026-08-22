import asyncio
import copy
import inspect
import re
from typing import Any
from uuid import UUID

import tiktoken

from pipelines.ingestion.models import Chunk, ExtractedPage


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vector embeddings."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class Chunker:
    """Structure-aware, semantic, and hierarchical chunker with configurable token size and overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize chunker with size, overlap, and tokenizer encoding."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Return exact token count for the given text."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def select_strategy(self, pages: list[ExtractedPage]) -> str:
        """Automatically select chunking strategy based on document structure, type, and size."""
        if not pages:
            return "fixed_size"

        # 1. Table content -> fixed_size
        if any(p.content_type == "table" for p in pages):
            return "fixed_size"

        # 2. DOCX with headings -> hierarchical
        has_headings = any(
            p.metadata.get("level") is not None
            or (p.metadata.get("source") == "paragraph" and "section" in p.metadata)
            or p.metadata.get("source") == "header"
            for p in pages
        )
        if has_headings:
            return "hierarchical"

        # 3. PDF with > 50 pages -> semantic
        is_pdf = any(
            p.metadata.get("source") == "pdf" or "page" in p.metadata for p in pages
        )
        if is_pdf and len(pages) > 50:
            return "semantic"

        # 4. Otherwise default to fixed_size
        return "fixed_size"

    def chunk(
        self,
        pages: list[ExtractedPage],
        document_id: str | UUID | None = None,
        strategy: str = "auto",
        embedder: Any = None,
        similarity_threshold: float = 0.7,
        sentence_embeddings: list[list[float]] | None = None,
    ) -> list[Chunk]:
        """Convert extracted pages into chunks using automatic or explicit strategy."""
        selected_strategy = strategy
        if selected_strategy == "auto":
            selected_strategy = self.select_strategy(pages)

        if selected_strategy == "fixed_size":
            return self.chunk_fixed_size(pages, document_id=document_id)
        elif selected_strategy == "hierarchical":
            return self.chunk_hierarchical(pages, document_id=document_id)
        elif selected_strategy == "semantic":
            return self.chunk_semantic(
                pages,
                document_id=document_id,
                similarity_threshold=similarity_threshold,
                embedder=embedder,
                sentence_embeddings=sentence_embeddings,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {selected_strategy}")

    def chunk_fixed_size(
        self,
        pages: list[ExtractedPage],
        document_id: str | UUID | None = None,
    ) -> list[Chunk]:
        """Fixed-size strategy: preserves sentence/paragraph boundaries and applies token fallback."""
        chunks: list[Chunk] = []
        global_chunk_index = 0

        for page in pages:
            if not page.content or not page.content.strip():
                continue

            page_content = page.content.strip()
            page_tokens = self.count_tokens(page_content)

            base_metadata: dict[str, Any] = copy.deepcopy(page.metadata)
            base_metadata.setdefault("page_number", page.page_number)
            base_metadata.setdefault("content_type", page.content_type)
            base_metadata["strategy"] = "fixed_size"
            if document_id is not None:
                base_metadata["document_id"] = str(document_id)

            if page_tokens <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=page_content,
                        chunk_index=global_chunk_index,
                        token_count=page_tokens,
                        page_number=page.page_number,
                        metadata=copy.deepcopy(base_metadata),
                    )
                )
                global_chunk_index += 1
                continue

            page_chunks = self._chunk_structured_text(page_content)

            for chunk_text in page_chunks:
                chunk_token_count = self.count_tokens(chunk_text)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_index=global_chunk_index,
                        token_count=chunk_token_count,
                        page_number=page.page_number,
                        metadata=copy.deepcopy(base_metadata),
                    )
                )
                global_chunk_index += 1

        return chunks

    def chunk_hierarchical(
        self,
        pages: list[ExtractedPage],
        document_id: str | UUID | None = None,
    ) -> list[Chunk]:
        """Hierarchical strategy: groups content by heading levels and tracks parent-child hierarchy."""
        chunks: list[Chunk] = []
        global_chunk_index = 0

        current_parent_section: str | None = None
        current_level: str | None = None
        hierarchy_path: list[str] = []

        for page in pages:
            if not page.content or not page.content.strip():
                continue

            page_content = page.content.strip()
            level = page.metadata.get("level")
            section_title = page.metadata.get("section")

            if level is not None or section_title is not None:
                current_level = level or "h1"
                current_parent_section = section_title or page_content

                if current_level == "h1":
                    hierarchy_path = [current_parent_section]
                else:
                    if hierarchy_path:
                        hierarchy_path = [hierarchy_path[0], current_parent_section]
                    else:
                        hierarchy_path = [current_parent_section]

            base_metadata: dict[str, Any] = copy.deepcopy(page.metadata)
            base_metadata.setdefault("page_number", page.page_number)
            base_metadata.setdefault("content_type", page.content_type)
            base_metadata["strategy"] = "hierarchical"
            if current_parent_section:
                base_metadata["parent_section"] = current_parent_section
            if current_level:
                base_metadata["heading_level"] = current_level
            if hierarchy_path:
                base_metadata["hierarchy_path"] = list(hierarchy_path)

            if document_id is not None:
                base_metadata["document_id"] = str(document_id)

            page_tokens = self.count_tokens(page_content)
            if page_tokens <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=page_content,
                        chunk_index=global_chunk_index,
                        token_count=page_tokens,
                        page_number=page.page_number,
                        metadata=copy.deepcopy(base_metadata),
                    )
                )
                global_chunk_index += 1
            else:
                page_chunks = self._chunk_structured_text(page_content)
                for chunk_text in page_chunks:
                    chunk_token_count = self.count_tokens(chunk_text)
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            chunk_index=global_chunk_index,
                            token_count=chunk_token_count,
                            page_number=page.page_number,
                            metadata=copy.deepcopy(base_metadata),
                        )
                    )
                    global_chunk_index += 1

        return chunks

    def chunk_semantic(
        self,
        pages: list[ExtractedPage],
        document_id: str | UUID | None = None,
        similarity_threshold: float = 0.7,
        embedder: Any = None,
        sentence_embeddings: list[list[float]] | None = None,
    ) -> list[Chunk]:
        """Semantic strategy: splits based on topic shifts using sentence similarity drop below threshold."""
        chunks: list[Chunk] = []
        global_chunk_index = 0

        for page in pages:
            if not page.content or not page.content.strip():
                continue

            page_content = page.content.strip()
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", page_content)
                if s.strip()
            ]

            if not sentences:
                continue

            base_metadata: dict[str, Any] = copy.deepcopy(page.metadata)
            base_metadata.setdefault("page_number", page.page_number)
            base_metadata.setdefault("content_type", page.content_type)
            base_metadata["strategy"] = "semantic"
            if document_id is not None:
                base_metadata["document_id"] = str(document_id)

            if len(sentences) == 1 or self.count_tokens(page_content) <= self.chunk_size and sentence_embeddings is None and embedder is None:
                # If short single block and no embedder provided
                pass

            # Resolve embeddings for sentences
            embeddings = sentence_embeddings
            if embeddings is None:
                if embedder is None:
                    try:
                        from backend.providers.client import get_client
                        client = get_client()
                        embedder = client
                    except Exception as exc:
                        raise RuntimeError(
                            f"Semantic chunking requires an available embedding provider: {exc}"
                        ) from exc

                # Fetch embeddings via embedder (sync or async)
                try:
                    embed_func = getattr(embedder, "embed", None) or getattr(embedder, "embed_chunks", None)
                    if embed_func is None:
                        raise RuntimeError("Provided embedder has no embed method")

                    if inspect.iscoroutinefunction(embed_func):
                        # Running in an existing event loop or new loop
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    embeddings = pool.submit(lambda: asyncio.run(embed_func(sentences))).result()
                            else:
                                embeddings = loop.run_until_complete(embed_func(sentences))
                        except RuntimeError:
                            embeddings = asyncio.run(embed_func(sentences))
                    else:
                        embeddings = embed_func(sentences)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to generate sentence embeddings for semantic chunking: {exc}"
                    ) from exc

            if len(embeddings) != len(sentences):
                raise ValueError(
                    f"Sentence count ({len(sentences)}) and embedding count ({len(embeddings)}) mismatch"
                )

            # Group sentences based on similarity threshold
            sentence_groups: list[list[str]] = []
            current_group: list[str] = [sentences[0]]

            for i in range(len(sentences) - 1):
                sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
                curr_group_tokens = self.count_tokens(" ".join(current_group + [sentences[i + 1]]))

                # Split if similarity drops below threshold or chunk exceeds size limit
                if sim < similarity_threshold or curr_group_tokens > self.chunk_size:
                    sentence_groups.append(current_group)
                    current_group = [sentences[i + 1]]
                else:
                    current_group.append(sentences[i + 1])

            if current_group:
                sentence_groups.append(current_group)

            for group in sentence_groups:
                group_text = " ".join(group).strip()
                group_tokens = self.count_tokens(group_text)

                if group_tokens <= self.chunk_size:
                    chunks.append(
                        Chunk(
                            content=group_text,
                            chunk_index=global_chunk_index,
                            token_count=group_tokens,
                            page_number=page.page_number,
                            metadata=copy.deepcopy(base_metadata),
                        )
                    )
                    global_chunk_index += 1
                else:
                    fallback_chunks = self._chunk_structured_text(group_text)
                    for chunk_text in fallback_chunks:
                        c_tokens = self.count_tokens(chunk_text)
                        chunks.append(
                            Chunk(
                                content=chunk_text,
                                chunk_index=global_chunk_index,
                                token_count=c_tokens,
                                page_number=page.page_number,
                                metadata=copy.deepcopy(base_metadata),
                            )
                        )
                        global_chunk_index += 1

        return chunks

    def _chunk_structured_text(self, text: str) -> list[str]:
        """Split text along structural boundaries respecting chunk_size and chunk_overlap."""
        atomic_units = self._split_into_atomic_units(text)
        if not atomic_units:
            return []

        if len(atomic_units) == 1 and self.count_tokens(atomic_units[0]) <= self.chunk_size:
            return atomic_units

        packed_chunks: list[str] = []
        current_units: list[str] = []
        i = 0

        while i < len(atomic_units):
            unit = atomic_units[i]

            if not current_units:
                current_units.append(unit)
                i += 1
            else:
                candidate = "\n\n".join(current_units + [unit])
                candidate_tokens = self.count_tokens(candidate)

                if candidate_tokens <= self.chunk_size:
                    current_units.append(unit)
                    i += 1
                else:
                    emitted_text = "\n\n".join(current_units)
                    packed_chunks.append(emitted_text)

                    overlap_units: list[str] = []
                    for u in reversed(current_units):
                        cand_overlap = (
                            "\n\n".join([u] + overlap_units) if overlap_units else u
                        )
                        cand_tokens = self.count_tokens(cand_overlap)
                        if (
                            cand_tokens <= self.chunk_overlap
                            and cand_tokens < self.chunk_size
                        ):
                            overlap_units.insert(0, u)
                        else:
                            break

                    current_units = list(overlap_units)

        if current_units:
            final_text = "\n\n".join(current_units)
            if not packed_chunks or final_text != packed_chunks[-1]:
                packed_chunks.append(final_text)

        return packed_chunks

    def _split_into_atomic_units(self, text: str) -> list[str]:
        """Recursively break text down into chunks that are each <= chunk_size tokens."""
        if self.count_tokens(text) <= self.chunk_size:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(paragraphs) > 1:
            units: list[str] = []
            for p in paragraphs:
                units.extend(self._split_into_atomic_units(p))
            return units

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            units = []
            for line in lines:
                units.extend(self._split_into_atomic_units(line))
            return units

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
        ]
        if len(sentences) > 1:
            units = []
            for s in sentences:
                units.extend(self._split_into_atomic_units(s))
            return units

        return self._token_based_split(text)

    def _token_based_split(self, text: str) -> list[str]:
        """Split unbroken text into token-sized windows respecting overlap."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        stride = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            window_tokens = tokens[start:end]
            decoded = self._tokenizer.decode(window_tokens).strip()
            if decoded:
                chunks.append(decoded)
            if end >= len(tokens):
                break
            start += stride

        return chunks

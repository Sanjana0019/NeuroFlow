import pytest

from pipelines.generation.prompt_builder import (
    BASE_SYSTEM_PROMPT,
    PromptBuilder,
)


def test_prompt_builder_factual_prompt():
    """Factual prompt includes base instructions and direct answer guidance."""
    builder = PromptBuilder()
    messages = builder.build(
        query="What is the capital of France?",
        context="[Source 1 — doc.pdf, page 1]\nParis is the capital of France.",
        query_type="factual",
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert BASE_SYSTEM_PROMPT in messages[0].content
    assert "Provide a direct, concise answer" in messages[0].content

    assert messages[1].role == "user"
    assert "<context>" in messages[1].content
    assert "Paris is the capital of France." in messages[1].content
    assert "</context>" in messages[1].content
    assert "What is the capital of France?" in messages[1].content


def test_prompt_builder_analytical_prompt():
    """Analytical prompt includes synthesis and contradiction identification guidance."""
    builder = PromptBuilder()
    messages = builder.build(
        query="Analyze the economic trends",
        context="Some context",
        query_type="analytical",
    )

    assert "Analyze and synthesize across the provided sources" in messages[0].content


def test_prompt_builder_comparative_prompt():
    """Comparative prompt includes structured comparison instructions."""
    builder = PromptBuilder()
    messages = builder.build(
        query="Compare SQL vs NoSQL",
        context="Some context",
        query_type="comparative",
    )

    assert "Organize your response as a structured comparison" in messages[0].content


def test_prompt_builder_procedural_prompt():
    """Procedural prompt includes numbered steps requirement."""
    builder = PromptBuilder()
    messages = builder.build(
        query="How to configure pgvector?",
        context="Some context",
        query_type="procedural",
    )

    assert "Provide numbered steps. Each step must be cited." in messages[0].content


def test_prompt_builder_context_tag_wrapping():
    """Context is strictly wrapped inside <context> ... </context> tags before the query."""
    builder = PromptBuilder()
    context_text = "Doc 1 text.\n\nDoc 2 text."
    query_text = "Summarize the key points."
    user_message = builder.build_user_message(query=query_text, context=context_text)

    expected = f"<context>\n{context_text}\n</context>\n\n{query_text}"
    assert user_message == expected

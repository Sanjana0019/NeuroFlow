from backend.providers.base import ChatMessage

BASE_SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer fully, say so explicitly.
For every factual claim, include a citation in the format [Source N].
Do not introduce information not present in the context."""

QUERY_TYPE_INSTRUCTIONS = {
    "factual": "Provide a direct, concise answer. If multiple sources agree, cite all of them.",
    "analytical": "Analyze and synthesize across the provided sources. Identify agreements and contradictions.",
    "comparative": "Organize your response as a structured comparison. Use a table if appropriate.",
    "procedural": "Provide numbered steps. Each step must be cited.",
}


class PromptBuilder:
    """Builds grounded RAG prompts tailored to the classified query type."""

    def __init__(self, base_system_prompt: str = BASE_SYSTEM_PROMPT):
        self.base_system_prompt = base_system_prompt

    def build_system_prompt(self, query_type: str = "factual") -> str:
        """Construct system prompt with query-type specific guidance."""
        type_key = (query_type or "factual").lower()
        type_instruction = QUERY_TYPE_INSTRUCTIONS.get(type_key, QUERY_TYPE_INSTRUCTIONS["factual"])
        return f"{self.base_system_prompt}\n\n{type_instruction}"

    def build_user_message(self, query: str, context: str) -> str:
        """Format user query with context delimited by <context> tags."""
        clean_context = context.strip() if context else "No context available."
        clean_query = query.strip()
        return f"<context>\n{clean_context}\n</context>\n\n{clean_query}"

    def build(
        self,
        query: str,
        context: str,
        query_type: str = "factual",
    ) -> list[ChatMessage]:
        """Generate structured ChatMessages for LLM completion."""
        system_prompt = self.build_system_prompt(query_type)
        user_message = self.build_user_message(query, context)

        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]

import asyncio
import os
import sys

from backend.config import settings
from backend.providers.base import ChatMessage
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.providers.openai_provider import OpenAIProvider


async def verify():
    print("========================================")
    print("NEUROFLOW PROVIDER VERIFICATION")
    print("========================================")

    api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        provider_type = "OpenAI" if api_key else None
    else:
        provider_type = "OpenRouter"

    if not api_key:
        print("Status: No API key configured in .env (OPENROUTER_API_KEY is not set).")
        print("Expected Model: nvidia/nemotron-3-embed-1b:free")
        print("Expected Dimension: 2048")
        print("LLM Model: meta-llama/llama-3.2-3b-instruct:free")
        print("Provider Layer Status: Initialized and ready for key.")
        return

    print(f"Provider: {provider_type}")

    if provider_type == "OpenRouter":
        provider = OpenRouterProvider(
            api_key=api_key,
            model=settings.openrouter_llm_model,
            embedding_model=settings.openrouter_embedding_model,
            base_url=settings.openrouter_base_url,
        )
    else:
        provider = OpenAIProvider(
            api_key=api_key,
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        )

    print(f"LLM Model: {provider.model}")
    print(f"Embedding Model: {provider.embedding_model}")

    # 1. Test Embedding
    try:
        sample_text = ["NeuroFlow hybrid RAG architectural verification."]
        embeddings = await provider.embed(sample_text)
        dim = len(embeddings[0])
        print(f"Embedding Test: SUCCESS")
        print(f"Returned Embedding Dimension: {dim}")
    except Exception as exc:
        print(f"Embedding Test: FAILED ({type(exc).__name__}: {exc})")

    # 2. Test LLM Completion
    try:
        messages = [ChatMessage(role="user", content="Respond with the single word 'OK'.")]
        result = await provider.complete(messages, max_tokens=10)
        print(f"LLM Completion Test: SUCCESS")
        print(f"LLM Response Content: {result.content.strip()!r}")
    except Exception as exc:
        print(f"LLM Completion Test: FAILED ({type(exc).__name__}: {exc})")

    print("========================================")

if __name__ == "__main__":
    asyncio.run(verify())

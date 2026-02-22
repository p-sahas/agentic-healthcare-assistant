"""
Embedding model provider.

Routes through OpenRouter when PROVIDER=openrouter, otherwise direct OpenAI.
"""

from typing import Any
from langchain_openai import OpenAIEmbeddings

from infrastructure.config import EMBEDDING_MODEL, PROVIDER, OPENROUTER_BASE_URL, get_api_key


def get_default_embeddings(
    batch_size: int = 100,
    show_progress: bool = False,
    **kwargs: Any
) -> OpenAIEmbeddings:
    """
    Get an OpenAIEmbeddings instance configured for the active provider.

    When PROVIDER=openrouter, requests are routed through the OpenRouter
    unified API so that model IDs resolve correctly.

    Args:
        batch_size: Number of texts to embed per API call.
        show_progress: Show progress bar during embedding.
        **kwargs: Additional arguments forwarded to OpenAIEmbeddings.

    Returns:
        A ready-to-use OpenAIEmbeddings instance.
    """
    llm_kwargs: dict[str, Any] = dict(
        model=EMBEDDING_MODEL,
        show_progress_bar=show_progress,
        **kwargs,
    )

    if PROVIDER == "openrouter":
        llm_kwargs["openai_api_base"] = OPENROUTER_BASE_URL
        llm_kwargs["openai_api_key"] = get_api_key("openrouter")

    return OpenAIEmbeddings(**llm_kwargs)

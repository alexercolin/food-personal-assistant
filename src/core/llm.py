from langchain_ollama import ChatOllama

from src.core.config import settings

_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=1024,
        )
    return _llm

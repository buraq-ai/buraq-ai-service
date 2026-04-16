import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm():
    """
    Returns a LangChain LLM instance based on the LLM_PROVIDER environment variable.

    Environment variables:
        LLM_PROVIDER: "openai" or "ollama" (default: "openai")
        OPENAI_API_KEY: Required if LLM_PROVIDER is "openai"
        OLLAMA_BASE_URL: Optional, defaults to "http://localhost:11434"
        OLLAMA_MODEL: Optional, defaults to "llama3.2"
    """

    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            api_key=api_key,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        )

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        )

    else:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {provider}. Must be 'openai' or 'ollama'")

import os
import pytest
from config.llm_config import get_llm


class TestGetLLM:

    def test_ollama_provider_creates_chat_ollama(self):
        """Verify that setting LLM_PROVIDER=ollama returns a ChatOllama instance."""
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = "llama3.2"

        llm = get_llm()

        assert "ChatOllama" in type(llm).__name__

    def test_invalid_provider_raises_value_error(self):
        """Verify that an invalid provider raises a ValueError."""
        os.environ["LLM_PROVIDER"] = "invalid_provider"

        with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
            get_llm()

    def test_openai_missing_key_raises_value_error(self):
        """Verify that OpenAI provider without API key raises ValueError."""
        os.environ["LLM_PROVIDER"] = "openai"
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm()
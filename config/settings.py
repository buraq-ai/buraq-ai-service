from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file if it exists

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Configuration
    LLM_PROVIDER: str = "openai"          # or "ollama"
    OPENAI_API_KEY: str = ""
    OLLAMA_MODEL: str = "llama3.2"

    # ChromaDB Configuration (local dev)
    CHROMA_DB_PATH: str = "./chroma_db"
    
    # --- NEW FOR BURAQ-23: ChromaDB Docker HTTP Connection ---
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION_NAME: str = "buraq_documents"
    
    # --- NEW FOR BURAQ-23: Spring Boot Callback Configuration ---
    SPRING_BOOT_URL: str = "http://localhost:8080"
    INTERNAL_SERVICE_KEY: str = "change_this_secret_key"

    # JWT (will be used later to validate requests from Java backend)
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"

    # Application
    DEBUG: bool = True

settings = Settings()
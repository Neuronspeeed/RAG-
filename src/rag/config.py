"""Configuration for the RAG system."""
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables first to maintain compatibility
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    # API Keys - Use environment variables directly for backward compatibility
    openai_api_key: str = Field(default_factory=lambda: os.environ.get('OPENAI_API_KEY'))
    tavily_api_key: str = Field(default_factory=lambda: os.environ.get('TAVILY_API_KEY'))
    
    # Model Settings
    default_model: str = Field(default="gpt-4-turbo-preview")
    
    # Vector Store Settings
    persist_directory: Path = Field(
        default=Path("./cache"),
        description="Directory for persisting vector store"
    )
    
    # Search Settings
    content_weight: float = Field(default=0.7)
    metadata_weight: float = Field(default=0.3)
    default_top_k: int = Field(default=5)
    min_similarity_score: float = Field(default=0.5)
    
    class Config:
        env_file = ".env"
        env_prefix = "RAG_"
        
    def validate_environment(self) -> None:
        """Validate required environment variables."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")

# Create settings instance
settings = Settings()

# Export API keys for backward compatibility
OPENAI_API_KEY = settings.openai_api_key
TAVILY_API_KEY = settings.tavily_api_key
DEFAULT_MODEL = settings.default_model

# Validate on import
settings.validate_environment()
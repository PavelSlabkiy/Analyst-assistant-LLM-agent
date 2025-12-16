"""
Configuration module for loading environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class Config:
    """Application configuration."""
    
    # OpenRouter API
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # LLM Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "kwaipilot/kat-coder-pro:free")
    
    # Data paths
    DATA_PATH: str = os.getenv("DATA_PATH", "data.json")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "metadata.json")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        errors = []
        
        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY is not set")
        
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is not set")
        
        if errors:
            for error in errors:
                print(f"❌ Config Error: {error}")
            return False
        
        return True


config = Config()

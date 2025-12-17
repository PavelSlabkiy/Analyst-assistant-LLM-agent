"""
Модуль конфигурации для загрузки переменных окружения.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class Config:
    """Конфигурация приложения."""
    
    # OpenRouter API
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # Telegram-бот
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # Настройки LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "kwaipilot/kat-coder-pro:free")
    LLM_FORMATTER_MODEL: str = os.getenv("LLM_FORMATTER_MODEL", "google/gemma-3-4b-it:free")
    
    # Пути к файлам данных
    DATA_PATH: str = os.getenv("DATA_PATH", "data.json")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "metadata.json")
    
    @classmethod
    def validate(cls) -> bool:
        """Проверка наличия всех обязательных параметров конфигурации."""
        errors = []
        
        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY не установлен")
        
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN не установлен")
        
        if errors:
            for error in errors:
                print(f"❌ Ошибка конфигурации: {error}")
            return False
        
        return True


config = Config()

"""Настройки приложения.

Источник значений: переменные окружения и файл .env (см. .env.example).
Один объект `settings` импортируется везде — никаких os.environ по коду.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    genius_access_token: SecretStr
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql://genius:genius@localhost:5432/genius"

    openrouter_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-4o-mini"


settings = Settings()

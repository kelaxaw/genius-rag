"""Application settings.

Values come from environment variables and .env (see .env.example).
A single `settings` object is imported everywhere; no os.environ access elsewhere.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: SecretStr
    genius_access_token: SecretStr
    postgres_dsn: str = "postgresql://genius:genius@localhost:5432/genius"

    openrouter_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-4o-mini"


settings = Settings()

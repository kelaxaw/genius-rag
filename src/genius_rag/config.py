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

    # Langfuse tracing: disabled without keys; tests turn it off via LANGFUSE_TRACING_ENABLED.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_tracing_environment: str = "development"
    langfuse_tracing_enabled: bool = True


settings = Settings()

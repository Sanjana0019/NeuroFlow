from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "neuroflow"
    postgres_user: str = "neuroflow"
    postgres_password: str

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str

    mlflow_tracking_uri: str = "http://localhost:5000"

    otel_exporter_endpoint: str = "http://localhost:4318/v1/traces"

    # Optional LLM provider credentials.
    # NeuroFlow can start without these credentials.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # OpenRouter configuration
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_llm_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    openrouter_embedding_model: str = "nvidia/nemotron-3-embed-1b:free"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        extra="ignore",
    )


settings = Settings()
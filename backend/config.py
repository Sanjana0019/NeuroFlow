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

    model_config = SettingsConfigDict(
        env_file="../.env",
        extra="ignore",
    )


settings = Settings()
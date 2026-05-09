"""Runtime configuration. Env-driven via Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QORBA_",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    api_base_url: str = "http://localhost:8000"
    web_origin: str = "http://localhost:3000"

    database_url: str = "postgresql+psycopg://qorba:qorba@localhost:5432/qorba"

    session_secret: str = Field(default="dev-secret-change-me-32-bytes-min")
    session_cookie_name: str = "qorba_session"
    session_max_age_seconds: int = 60 * 60 * 24 * 30

    allow_registration: bool = True

    caissa_api_key: str | None = None
    caissa_base_url: str = "https://client-api.caissallc.com"

    anthropic_api_key: str | None = None
    max_llm_cost_per_upload_usd: float = 0.50
    max_llm_cost_per_month_usd: float = 50.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

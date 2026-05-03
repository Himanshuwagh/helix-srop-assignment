import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = ""
    # Gemini API embedContent model (not Vertex names like text-embedding-004).
    gemini_embedding_model: str = "models/gemini-embedding-001"
    # Prefer flash-lite on the free tier (higher RPD than 2.5 Flash; 2.0 Flash quotas are often exhausted/deprecated).
    adk_model: str = "gemini-2.0-flash-lite"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10


settings = Settings()


def _sync_google_api_key_to_os_environ() -> None:
    """google.genai / ADK read GOOGLE_API_KEY from os.environ; pydantic only loads .env into Settings."""
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key


_sync_google_api_key_to_os_environ()

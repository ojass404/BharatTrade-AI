from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="BharatTrade AI", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")

    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        alias="CORS_ORIGINS",
    )

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/stock_ai",
        alias="DATABASE_URL",
    )

    upstox_analytics_token: str | None = Field(
        default=None,
        alias="UPSTOX_ANALYTICS_TOKEN",
    )

    model_directory: Path = Field(
        default=Path("ml_pipeline/artifacts"),
        alias="MODEL_DIRECTORY",
    )
    conv1d_model_path: Path = Field(
        default=Path("ml_pipeline/artifacts/conv1d_model.onnx"),
        alias="CONV1D_MODEL_PATH",
    )
    lstm_model_path: Path = Field(
        default=Path("ml_pipeline/artifacts/lstm_model.onnx"),
        alias="LSTM_MODEL_PATH",
    )
    scaler_path: Path = Field(
        default=Path("ml_pipeline/artifacts/feature_scaler.joblib"),
        alias="SCALER_PATH",
    )

    initial_virtual_capital: float = Field(
        default=100000.0,
        alias="INITIAL_VIRTUAL_CAPITAL",
        gt=0,
    )
    max_position_percent: float = Field(
        default=10.0,
        alias="MAX_POSITION_PERCENT",
        gt=0,
        le=100,
    )
    daily_loss_limit_percent: float = Field(
        default=3.0,
        alias="DAILY_LOSS_LIMIT_PERCENT",
        gt=0,
        le=100,
    )
    default_stop_loss_percent: float = Field(
        default=2.0,
        alias="DEFAULT_STOP_LOSS_PERCENT",
        gt=0,
        le=100,
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Convert the comma-separated CORS value into a clean list."""

        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Create and cache one application-settings object."""

    return Settings()
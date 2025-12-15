import logging
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LOG_LEVEL: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info"
    )

    MUSIC_LIBRARIES: list[str] = Field(
        default_factory=list,
        description="List of music library directories to scan for music files"
    )

    SCAN_INTERVAL: int = Field(
        default=3600,
        description="Interval in seconds between automatic scans for new music files"
    )

    BATCH_SIZE: int = Field(
        default=32,
        gt=0,
        le=128,
        description="Batch size for processing embeddings"
    )

    @property
    def log_level(self) -> int:
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        return level_map[self.LOG_LEVEL]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()  # type: ignore

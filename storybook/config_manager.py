import os
from typing import Optional

from pydantic import BaseSettings

class Config(BaseSettings):
    """Config for the app."""

    model: Optional[str] = "gpt-4-turbo-preview"
    mode: Optional[str] = "Story"
    debug: Optional[bool] = False
    voice: Optional[str] = "nova"
    
class ConfigManager:
    """Manage the configuration for the app."""

    def __init__(self) -> None:
        """Initialize the config manager."""
        self._config = Config()

    def get_config(self) -> Config:
        """Get the config."""
        return self._config

# storybook/config/models_config.py

"""
Configuration for language models used in the Storybook system.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Configuration for a specific language model."""
    provider: str = Field(..., description="Model provider (e.g., 'openai', 'anthropic')")
    name: str = Field(..., description="Model name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    
class ModelsConfig(BaseModel):
    """Configuration for all language models used in the system."""
    default_model: str = Field(default="gpt-4", description="Default model to use")
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="All available models")
    
    def get_model_config(self, model_name: Optional[str] = None) -> ModelConfig:
        """Get configuration for a specific model."""
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in configuration")
        return self.models[model_name]

# Default models configuration
DEFAULT_MODELS_CONFIG = ModelsConfig(
    default_model="gpt-4",
    models={
        "gpt-4": ModelConfig(
            provider="openai",
            name="gpt-4",
            parameters={
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        ),
        "gpt-3.5-turbo": ModelConfig(
            provider="openai", 
            name="gpt-3.5-turbo",
            parameters={
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        ),
        "claude-3-opus": ModelConfig(
            provider="anthropic",
            name="claude-3-opus",
            parameters={
                "temperature": 0.7,
                "max_tokens": 4000,
            }
        )
    }
)

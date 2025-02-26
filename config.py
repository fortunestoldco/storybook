import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class storybookConfig(BaseModel):
    """Configuration for the novel generation system."""
    # LLM Configuration
    llm_model: str = Field(default="gpt-4", description="The LLM model to use")
    temperature: float = Field(default=0.7, description="Temperature for agent LLM calls")
    
    # Project Settings
    project_name: str = Field(default="New Novel Project", description="Name of the novel project")
    genre: str = Field(default="", description="Primary genre of the novel")
    target_word_count: int = Field(default=80000, description="Target word count")
    target_audience: str = Field(default="Adult", description="Target audience age range")
    
    # Quality Thresholds
    min_thematic_coherence: float = Field(default=0.75, description="Minimum thematic coherence score")
    min_character_consistency: float = Field(default=0.8, description="Minimum character consistency score")
    min_narrative_engagement: float = Field(default=0.7, description="Minimum narrative engagement score")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # System Settings
    max_revision_cycles: int = Field(default=3, description="Maximum number of revision cycles")
    parallel_chapter_generation: bool = Field(default=True, description="Whether to generate chapters in parallel")
    
    def __init__(self, **data: Any):
        """Initialize with environment variables if not provided."""
        if "openai_api_key" not in data:
            data["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
        super().__init__(**data)
    
    def get_llm_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for initializing an LLM."""
        return {
            "model": self.llm_model,
            "temperature": self.temperature,
        }

# Default configuration
default_config = storybookConfig()

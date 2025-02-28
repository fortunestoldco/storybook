from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class LLMProvider(str, Enum):
    """Available LLM providers and their models."""
    # OpenAI Models
    GPT4 = "openai/gpt-4"
    GPT4_TURBO = "openai/gpt-4-turbo-preview"
    GPT35 = "openai/gpt-3.5-turbo"
    
    # Anthropic Models
    CLAUDE = "anthropic/claude-3-sonnet"
    CLAUDE_OPUS = "anthropic/claude-3-opus"
    CLAUDE_HAIKU = "anthropic/claude-3-haiku"
    
    # Hugging Face Models (require model name)
    HUGGINGFACE = "huggingface"
    
    # Replicate Models (require model name)
    REPLICATE = "replicate"
    
    # Local Models (require model name)
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"

    @property
    def provider(self) -> str:
        """Get the provider name without model."""
        return self.value.split('/')[0] if '/' in self.value else self.value

    @property
    def requires_model(self) -> bool:
        """Check if this provider requires explicit model specification."""
        return self in [
            LLMProvider.HUGGINGFACE,
            LLMProvider.REPLICATE,
            LLMProvider.OLLAMA,
            LLMProvider.LLAMACPP
        ]

    @property
    def get_default_model(self) -> Optional[str]:
        """Get default model name if available."""
        return self.value.split('/')[1] if '/' in self.value else None

class ManuscriptState(BaseModel):
    """Core manuscript state."""
    title: str
    manuscript: str
    notes: str
    llm_provider: LLMProvider
    llm_model: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

class AgentOutput(BaseModel):
    """Standard agent output structure."""
    content: Dict[str, Any]
    timestamp: datetime 
    agent_id: str
    
    model_config = {"arbitrary_types_allowed": True}

class State(BaseModel):
    """Overall graph state."""
    title: str
    manuscript: str
    notes: str = ""
    llm_provider: LLMProvider = Field(default=LLMProvider.CLAUDE)
    llm_model: Optional[str] = None
    market_analysis: Optional[AgentOutput] = None
    content_analysis: Optional[AgentOutput] = None
    characters: Optional[AgentOutput] = None
    dialogue: Optional[AgentOutput] = None
    world_building: Optional[AgentOutput] = None
    subplots: Optional[AgentOutput] = None
    story_arc: Optional[AgentOutput] = None
    language: Optional[AgentOutput] = None
    quality_review: Optional[AgentOutput] = None
    current_step: str = "start"
    
    model_config = {"arbitrary_types_allowed": True}
    
    def get_manuscript_state(self) -> ManuscriptState:
        """Create a ManuscriptState instance from current state."""
        return ManuscriptState(
            title=self.title,
            manuscript=self.manuscript,
            notes=self.notes,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model
        )

class InputState(BaseModel):
    """Input state definition."""
    title: str = Field(..., description="The title of the manuscript")
    manuscript: str = Field(..., description="The main manuscript text")
    notes: str = Field("", description="Additional notes or context")
    llm_provider: LLMProvider = Field(
        default=LLMProvider.CLAUDE,
        description="The LLM provider to use for processing"
    )
    llm_model: Optional[str] = Field(
        None,
        description="Custom model identifier (required for HuggingFace, Replicate, Ollama, or Llama.cpp)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "title": "My Story",
                "manuscript": "Once upon a time...",
                "notes": "Historical fiction set in 18th century London",
                "llm_provider": "anthropic/claude-3-sonnet",
                "llm_model": None
            }, {
                "title": "AI Generated Tale",
                "manuscript": "In the distant future...",
                "notes": "Science fiction",
                "llm_provider": "replicate",
                "llm_model": "meta/llama-2-70b-chat"
            }]
        }
    }

    @property
    def get_model_name(self) -> str:
        """Get the appropriate model name based on provider."""
        if not self.llm_provider.requires_model:
            return self.llm_provider.value.split('/')[1] if '/' in self.llm_provider.value else ""
        if not self.llm_model:
            raise ValueError(f"Model name required for {self.llm_provider.value}")
        return self.llm_model
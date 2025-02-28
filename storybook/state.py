from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class ManuscriptState(BaseModel):
    """Core manuscript state."""
    title: str
    manuscript: str
    notes: str
    llm_provider: LLMProvider
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
    manuscript: ManuscriptState
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

class LLMProvider(str, Literal):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"

class InputState(BaseModel):
    """Input state definition."""
    title: str = Field(..., description="The title of the manuscript")
    manuscript: str = Field(..., description="The main manuscript text")
    notes: str = Field("", description="Additional notes or context")
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="The LLM provider to use for processing"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "title": "My Story",
                "manuscript": "Once upon a time...",
                "notes": "Historical fiction set in 18th century London",
                "llm_provider": "anthropic"
            }]
        }
    }
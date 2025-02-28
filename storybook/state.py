from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ManuscriptState(BaseModel):
    """Core manuscript state."""
    text: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
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

class InputState(BaseModel):
    """Input state definition."""
    manuscript_text: str
    metadata: Dict[str, Any]
    
    model_config = {"arbitrary_types_allowed": True}
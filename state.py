from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal, cast, Union
import operator
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

# Define the project types and input states
class ProjectType:
    NEW = "new"
    EXISTING = "existing"

class NewProjectInput(TypedDict):
    title: str
    synopsis: str
    manuscript: str
    notes: Optional[Dict[str, Any]]

class ExistingProjectInput(TypedDict):
    project_id: str

class ProjectData(TypedDict):
    id: str
    title: str
    synopsis: str
    manuscript: str
    manuscript_chunks: List[Dict[str, Any]]
    notes: Optional[Dict[str, Any]]
    type: str
    quality_assessment: Dict[str, Any]
    created_at: str

class InputState(TypedDict):
    project_type: str
    project_data: Dict[str, Any]
    task: str

# Define the research state classes
class ResearchState(TypedDict):
    query: str
    results: List[Dict[str, Any]]
    summary: str

class DomainResearchState(ResearchState):
    domain_specific_data: Dict[str, Any]

class CulturalResearchState(ResearchState):
    cultural_context: Dict[str, Any]

class MarketResearchState(ResearchState):
    market_trends: Dict[str, Any]

class FactVerificationState(ResearchState):
    verification_status: Dict[str, bool]

# State for the storybook application
class AgentState(TypedDict):
    # storybook states
    project: Optional[ProjectData]
    phase: Optional[str]
    phase_history: Optional[Dict[str, List[Dict[str, Any]]]]
    current_input: Optional[Dict[str, Any]]
    messages: Optional[List[Dict[str, Any]]]
    # Tracking state
    count: Annotated[int, operator.add]
    lnode: str

class Configuration(BaseModel):
    quality_gates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_runnable_config(cls, config: Dict[str, Any]) -> 'Configuration':
        """Extract configuration from a runnable config."""
        configurable = config.get("configurable", {})
        return cls(
            quality_gates=configurable.get("quality_gates", {})
        )

class storybookConfig(BaseModel):
    model_name: str
    temperature: float
    max_tokens: Optional[int] = None

    model_config = {
        "extra": "forbid"
    }
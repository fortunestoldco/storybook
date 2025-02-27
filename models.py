from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str
    backstory: str = ""
    motivations: List[str] = Field(default_factory=list)
    arc: str = ""
    traits: List[str] = Field(default_factory=list)


class Setting(BaseModel):
    name: str
    description: str
    history: str = ""
    cultures: List[str] = Field(default_factory=list)
    environment: str = ""


class Subplot(BaseModel):
    title: str
    description: str
    connected_characters: List[str] = Field(default_factory=list)
    resolution: str = ""


class Novel(BaseModel):
    title: str
    author: str
    manuscript: str
    current_stage: str = "initial"
    characters: Dict[str, Character] = Field(default_factory=dict)
    settings: Dict[str, Setting] = Field(default_factory=dict)
    subplots: List[Subplot] = Field(default_factory=list)
    main_plot: str = ""
    revision_history: List[Dict] = Field(default_factory=list)
    latest_feedback: str = ""
    
    
class AgentInput(BaseModel):
    novel: Novel
    instructions: Optional[str] = None
    focus_areas: Optional[List[str]] = None


class AgentOutput(BaseModel):
    novel: Novel
    notes: str = ""
    changes_made: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

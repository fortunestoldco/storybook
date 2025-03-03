from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Union
from pydantic import BaseModel, Field


class TeamState(BaseModel):
    """State for a specific team."""
    tasks_pending: List[Dict] = Field(default_factory=list)
    tasks_completed: List[Dict] = Field(default_factory=list)
    work_artifacts: Dict = Field(default_factory=dict)
    quality_metrics: Dict = Field(default_factory=dict)


class DirectorState(BaseModel):
    """State for a director-level agent."""
    teams: Dict[str, TeamState] = Field(default_factory=dict)
    current_focus: str = ""
    strategic_direction: str = ""
    quality_metrics: Dict = Field(default_factory=dict)


class ProjectState(BaseModel):
    """Global state for the entire project."""
    # Basic project information
    project_id: str
    title: str
    genre: str
    target_audience: str
    word_count_target: int
    
    # Phase tracking
    current_phase: str = "initialization"
    phase_history: List[Dict] = Field(default_factory=list)
    
    # Director-level states
    executive_director: Dict = Field(default_factory=dict)
    creative_director: DirectorState = Field(default_factory=DirectorState)
    content_development_director: DirectorState = Field(default_factory=DirectorState)
    editorial_director: DirectorState = Field(default_factory=DirectorState)
    market_alignment_director: DirectorState = Field(default_factory=DirectorState)
    
    # Overall manuscript state
    manuscript: Dict = Field(default_factory=dict)
    
    # Quality and progress tracking
    quality_assessment: Dict = Field(default_factory=dict)
    progress_metrics: Dict = Field(default_factory=dict)
    
    # Human feedback
    human_feedback: List[Dict] = Field(default_factory=list)
    
    def update_phase(self, new_phase: str) -> None:
        """Update the current phase and record in history."""
        self.phase_history.append({
            "phase": self.current_phase,
            "timestamp": str(datetime.now())
        })
        self.current_phase = new_phase
    
    def get_team_state(self, director: str, team: str) -> TeamState:
        """Get the state for a specific team under a director."""
        director_state = getattr(self, director)
        return director_state.teams.get(team, TeamState())
    
    def set_team_state(self, director: str, team: str, state: TeamState) -> None:
        """Set the state for a specific team under a director."""
        director_state = getattr(self, director)
        director_state.teams[team] = state


# Define the state record that gets passed between agents
class NovelSystemState(TypedDict):
    """State record for the novel writing system."""
    project: ProjectState
    current_input: Dict
    current_output: Dict
    messages: List[Dict]
    errors: List[Dict]


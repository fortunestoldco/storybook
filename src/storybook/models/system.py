from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class NovelSystemState:
    """Represents the current state of the novel writing system."""
    
    project_id: str
    phase: str
    status: str = "active"
    current_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_state(self, agent: str, action: str, result: Dict[str, Any]) -> None:
        """Update the system state with new information."""
        self.current_agent = agent
        self.updated_at = datetime.utcnow()
        
        # Add to history
        self.history.append({
            "agent": agent,
            "action": action,
            "result": result,
            "timestamp": self.updated_at
        })
        
        # Update metadata based on result
        if "metadata_updates" in result:
            self.metadata.update(result["metadata_updates"])
    
    def get_agent_history(self, agent: str) -> List[Dict[str, Any]]:
        """Get all historical actions for a specific agent."""
        return [entry for entry in self.history if entry["agent"] == agent]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for storage."""
        return {
            "project_id": self.project_id,
            "phase": self.phase,
            "status": self.status,
            "current_agent": self.current_agent,
            "metadata": self.metadata,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
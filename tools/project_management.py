from typing import Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel
from datetime import datetime, timedelta
from langchain_core.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langsmith.run_helpers import traceable

class ProjectInput(BaseModel):
    title: str
    manuscript: str
    target_completion: Optional[datetime] = None
    milestones: Optional[List[str]] = None

@tool
def create_project_timeline(input_data: ProjectInput) -> Dict:
    """Creates a detailed project timeline with milestones."""
    start_date = datetime.now()
    estimated_completion = input_data.target_completion or (start_date + timedelta(days=30))
    
    # Detailed milestone breakdown
    milestones = input_data.milestones or ["Development", "Creation", "Refinement"]
    detailed_milestones = []
    for milestone in milestones:
        if milestone == "Development":
            detailed_milestones.append({
                "name": "Development",
                "duration": "10 days",
                "sub_tasks": ["Character profiles", "Plot outline"],
                "dependencies": [],
                "resource_allocation": ["Team A"],
                "progress_tracking": "0%"
            })
        elif milestone == "Creation":
            detailed_milestones.append({
                "name": "Creation",
                "duration": "14 days",
                "sub_tasks": ["First draft", "Review points"],
                "dependencies": ["Development"],
                "resource_allocation": ["Team B"],
                "progress_tracking": "0%"
            })
        elif milestone == "Refinement":
            detailed_milestones.append({
                "name": "Refinement",
                "duration": "6 days",
                "sub_tasks": ["Editing", "Final polish"],
                "dependencies": ["Creation"],
                "resource_allocation": ["Team C"],
                "progress_tracking": "0%"
            })
    
    return {
        "timeline": {
            "start_date": start_date.isoformat(),
            "estimated_completion": estimated_completion.isoformat(),
            "phases": detailed_milestones
        }
    }

@tool
def analyze_market_trends(input_data: ProjectInput) -> Dict:
    """Analyzes current market trends for story positioning."""
    return {
        "market_analysis": {
            "trending_genres": ["Urban Fantasy", "Contemporary Romance"],
            "audience_preferences": ["Strong character development", "Diverse representation"],
            "market_gaps": ["Unique magic systems", "Cross-genre innovation"],
            "recommendations": [
                "Consider incorporating trending elements",
                "Focus on unique selling points"
            ]
        }
    }

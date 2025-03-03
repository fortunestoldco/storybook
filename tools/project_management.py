from typing import Dict, List, Optional
from langchain.tools import tool
from pydantic import BaseModel
from datetime import datetime, timedelta

class ProjectInput(BaseModel):
    title: str
    manuscript: str
    target_completion: Optional[datetime] = None
    milestones: Optional[List[str]] = None

@tool
def create_project_timeline(input_data: ProjectInput) -> Dict:
    """Creates a detailed project timeline with milestones."""
    return {
        "timeline": {
            "start_date": datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(days=30)).isoformat(),
            "phases": [
                {
                    "name": "Development",
                    "duration": "10 days",
                    "milestones": ["Character profiles", "Plot outline"]
                },
                {
                    "name": "Creation",
                    "duration": "14 days",
                    "milestones": ["First draft", "Review points"]
                },
                {
                    "name": "Refinement",
                    "duration": "6 days",
                    "milestones": ["Editing", "Final polish"]
                }
            ]
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
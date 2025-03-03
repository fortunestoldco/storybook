from typing import Dict, List, Optional
from langchain.tools import tool
from pydantic import BaseModel

class StoryAnalysisInput(BaseModel):
    title: str
    manuscript: str
    criteria: Optional[List[str]] = None

@tool
def analyze_story_structure(input_data: StoryAnalysisInput) -> Dict:
    """Analyzes the story's structure including plot, pacing, and narrative flow."""
    return {
        "structure_score": 0.85,
        "elements": {
            "plot": "Three-act structure present",
            "pacing": "Good rhythm with minor issues",
            "narrative": "Clear progression"
        },
        "recommendations": [
            "Tighten middle section",
            "Add more tension points"
        ]
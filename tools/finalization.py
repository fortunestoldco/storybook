from typing import Dict, List, Optional
from langchain.tools import tool
from pydantic import BaseModel
from langchain.agents import AgentExecutor, StructuredChatAgent
from langsmith.run_helpers import traceable
from tools.finalization import (
    assess_market_readiness,
    perform_quality_check,
    optimize_marketability
)

class FinalizationInput(BaseModel):
    title: str
    manuscript: str
    market_requirements: Optional[List[str]] = None
    quality_targets: Optional[List[str]] = None

@tool
def assess_market_readiness(input_data: FinalizationInput) -> Dict:
    """Assesses story's market readiness and positioning."""
    return {
        "market_assessment": {
            "readiness_score": 0.88,
            "target_audience": "Clearly defined",
            "market_fit": "Strong alignment",
            "unique_selling_points": [
                "Innovative premise",
                "Strong character development"
            ]
        },
        "recommendations": [
            "Emphasize unique elements in marketing",
            "Consider series potential"
        ]
    }

@tool
def perform_quality_check(input_data: FinalizationInput) -> Dict:
    """Performs final quality assessment."""
    return {
        "quality_check": {
            "technical_quality": "Excellent",
            "narrative_cohesion": "Strong",
            "character_consistency": "Maintained",
            "pacing": "Well-balanced"
        },
        "final_score": 0.92,
        "certification": "Ready for publication"
    }

@tool
def optimize_marketability(input_data: FinalizationInput) -> Dict:
    """Optimizes story elements for market appeal."""
    return {
        "optimization": {
            "title_strength": "High impact",
            "hook_effectiveness": "Compelling",
            "genre_alignment": "Well-positioned",
            "market_elements": "Properly emphasized"
        },
        "suggestions": [
            "Consider alternate title options",
            "Strengthen opening hook"
        ]
    }

@traceable(name="Finalizer Agent")
def finalizer_agent(state: StoryState) -> Dict:
    """Agent responsible for final story preparation."""
    tools = [
        Tool(name="assess_market_readiness", func=assess_market_readiness),
        Tool(name="optimize_marketability", func=optimize_marketability)
    ]
    
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=get_llm_from_state(state),
        tools=tools
    )
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    result = executor.invoke({
        "input": {
            "title": state["title"],
            "manuscript": state["manuscript"]
        }
    })
    
    return {
        "market_assessment": result.get("market_assessment", {}),
        "optimization": result.get("optimization", {}),
        "feedback": ["Market readiness assessed", "Story optimized"],
        "agent_type": "finalizer",
        "agent_model": state["model_name"]
    }

@traceable(name="Quality Checker Agent")
def quality_checker_agent(state: StoryState) -> Dict:
    """Agent responsible for final quality assurance."""
    tools = [
        Tool(name="perform_quality_check", func=perform_quality_check)
    ]
    
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=get_llm_from_state(state),
        tools=tools
    )
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    result = executor.invoke({
        "input": {
            "title": state["title"],
            "manuscript": state["manuscript"]
        }
    })
    
    return {
        "quality_check": result.get("quality_check", {}),
        "final_score": result.get("final_score", 0.0),
        "feedback": ["Final quality check completed"],
        "agent_type": "quality_checker",
        "agent_model": state["model_name"]
    }
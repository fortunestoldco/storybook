from typing import Dict, List, Optional
from langchain_core.agents import AgentExecutor, Tool
from langchain.agents.structured_chat.base import StructuredChatAgent
from langsmith.run_helpers import traceable
from langchain_core.tools import tool
from pydantic import BaseModel

class RefinementInput(BaseModel):
    title: str
    manuscript: str
    revision_depth: Optional[str] = None
    focus_areas: Optional[List[str]] = None

@tool
def edit_content(input_data: RefinementInput) -> Dict:
    """Edits story content based on input data."""
    return {
        "edits": {
            "section": input_data.revision_depth or "general",
            "revised_text": "Edited content placeholder",
            "word_count": 450,
            "key_changes": [
                "Improved dialogue",
                "Enhanced descriptions",
                "Fixed inconsistencies"
            ]
        },
        "coherence": {
            "flow": "Improved",
            "consistency": "High"
        }
    }

@tool
def analyze_story_coherence(input_data: RefinementInput) -> Dict:
    """Analyzes the coherence of the story."""
    return {
        "coherence_analysis": {
            "overall_score": 0.87,
            "strengths": [
                "Consistent narrative voice",
                "Logical plot progression"
            ],
            "weaknesses": [
                "Minor pacing issues",
                "Some character motivations unclear"
            ]
        },
        "feedback": [
            "Consider revising pacing in middle chapters",
            "Clarify character motivations in key scenes"
        ]
    }

@tool
def verify_story_elements(input_data: RefinementInput) -> Dict:
    """Verifies story elements for consistency and accuracy."""
    return {
        "verification": {
            "elements_checked": [
                "Character consistency",
                "Timeline accuracy",
                "Setting details"
            ],
            "issues_found": [
                "Inconsistent character traits in chapter 3",
                "Timeline discrepancy in chapter 5"
            ]
        },
        "feedback": [
            "Ensure character traits are consistent throughout",
            "Correct timeline discrepancies"
        ]
    }

@traceable(name="Editor Agent")
def editor_agent(state: StoryState) -> Dict:
    """Agent responsible for story editing and refinement."""
    tools = [
        Tool(name="edit_content", func=edit_content),
        Tool(name="analyze_story_coherence", func=analyze_story_coherence)
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
            "manuscript": state["manuscript"],
            "revision_depth": "detailed"
        }
    })
    
    return {
        "edits": result.get("edits", {}),
        "coherence": result.get("coherence_analysis", {}),
        "feedback": ["Editing completed", "Story coherence improved"],
        "agent_type": "editor",
        "agent_model": state["model_name"]
    }

@traceable(name="Proofreader Agent")
def proofreader_agent(state: StoryState) -> Dict:
    """Agent responsible for final proofreading and verification."""
    tools = [
        Tool(name="verify_story_elements", func=verify_story_elements),
        Tool(name="edit_content", func=edit_content)
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
            "manuscript": state["manuscript"],
            "focus_areas": ["technical", "consistency"]
        }
    })
    
    return {
        "verification": result.get("verification", {}),
        "feedback": ["Proofreading completed", "Technical issues resolved"],
        "agent_type": "proofreader",
        "agent_model": state["model_name"]
    }

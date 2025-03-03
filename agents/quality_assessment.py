from typing import Dict, List, Any
from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langsmith.run_helpers import traceable
from tools.quality_assessment import (
    analyze_story_structure,
    evaluate_character_arcs,
    assess_narrative_coherence
)

@traceable(name="Quality Assessment Director")
def quality_assessment_director_agent(state: StoryState) -> Dict:
    """Quality Assessment Director agent responsible for story evaluation."""
    tools = [
        Tool(
            name="analyze_story_structure",
            func=analyze_story_structure,
            description="Analyzes the overall structure of the story"
        ),
        Tool(
            name="evaluate_character_arcs",
            func=evaluate_character_arcs,
            description="Evaluates character development and arcs"
        ),
        Tool(
            name="assess_narrative_coherence",
            func=assess_narrative_coherence,
            description="Assesses narrative coherence and flow"
        )
    ]
    
    # Create agent executor
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=get_llm_from_state(state),
        tools=tools,
        verbose=True
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    # Execute analysis
    result = agent_executor.invoke({
        "input": {
            "title": state["title"],
            "manuscript": state["manuscript"],
            "criteria": ["structure", "characters", "coherence"]
        }
    })
    
    # Format response
    return {
        "feedback": result["output"],
        "scores": {
            "structure": result.get("structure_score", 0.0),
            "characters": result.get("character_score", 0.0),
            "coherence": result.get("coherence_score", 0.0)
        },
        "agent_type": "quality_assessment_director",
        "agent_model": state["model_name"]
    }
import os
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain.agents import AgentExecutor  # Change: Updated import
from langchain_core.agents import AgentAction, AgentFinish  # Change: Updated imports

from ..agents.base import BaseAgent
from ..models.system import NovelSystemState
from ..storage.research import ResearchStorage
from .tools import (
    execute_research,
    analyze_research_quality,
    identify_knowledge_gaps,
    generate_followup_queries
)
from .states import ResearchState
from .config import validate_api_configuration, get_api_key
from .graphs import create_research_subgraph

class ResearchAgent:
    """Base class for research-focused agents using LangGraph."""
    
    def __init__(
        self,
        name: str,
        research_graph: StateGraph,
        state_class: Type[ResearchState],
        config: Dict[str, Any]
    ):
        self.name = name
        self.research_graph = research_graph 
        self.state_class = state_class
        self.config = config

    async def process(self, state: NovelSystemState) -> Dict[str, Any]:
        """Process the current state using research graph."""
        messages = [
            SystemMessage(content=self.config.get("system_prompt", "")),
            HumanMessage(content=state.current_input)
        ]
        
        # Create research state
        research_state = self.state_class(
            project_id=state.project_id,
            messages=messages,
            config=self.config
        )
        
        # Execute research graph
        result = await self.research_graph.ainvoke(research_state)
        return result

# Specialized research agents
class DomainKnowledgeSpecialist(ResearchAgent):
    """Specialist in domain-specific research."""
    pass

class CulturalAuthenticityExpert(ResearchAgent):
    """Expert in cultural research and authenticity verification."""
    pass

class MarketAlignmentDirector(ResearchAgent):
    """Director of market research and alignment."""
    pass

class FactVerificationSpecialist(ResearchAgent):
    """Specialist in fact checking and verification."""
    pass

__all__ = [
    "ResearchAgent",
    "DomainKnowledgeSpecialist",
    "CulturalAuthenticityExpert",
    "MarketAlignmentDirector",
    "FactVerificationSpecialist"
]
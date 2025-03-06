from typing import Dict, Any, List, Literal, Type
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

from ..utils.vector_search import VectorSearch
from .states import *
from .prompts import *
from .tools import *
from .configuration import Configuration
from .search import select_and_execute_search
from ..utils import load_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver

def create_research_subgraph(
    research_type: str,
    state_class: Type[ResearchState],
    config: Dict[str, Any]
) -> StateGraph:
    """Create a research focused subgraph."""
    
    builder = StateGraph(state_class)
    
    # Add tool nodes
    builder.add_node("execute_research", execute_research)
    builder.add_node("analyze_quality", analyze_research_quality) 
    builder.add_node("identify_gaps", identify_knowledge_gaps)
    builder.add_node("generate_queries", generate_followup_queries)

    # Add edges
    builder.add_edge("execute_research", "analyze_quality")
    builder.add_edge("analyze_quality", "identify_gaps")
    builder.add_edge("identify_gaps", "generate_queries")
    
    # Add conditional routing
    def should_continue_research(state: ResearchState) -> str:
        if state.iterations >= config.get("max_iterations", 3):
            return END
        if state.quality_score >= config.get("quality_threshold", 0.8):
            return END
        return "execute_research"
        
    builder.add_conditional_edges(
        "generate_queries",
        should_continue_research
    )
    
    return builder.compile()

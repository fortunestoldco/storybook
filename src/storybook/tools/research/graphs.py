from typing import Dict, Any, List, Literal, Type, Optional
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
from langchain.agents import AgentExecutor
from langchain.graphs import GraphStore

def create_research_subgraph(
    config: Optional[Dict[str, Any]] = None
) -> AgentExecutor:
    """Creates a research subgraph for managing research workflows"""
    if config is None:
        config = {}
    
    graph = GraphStore()
    
    # Add research nodes
    graph.add_node("ResearchQuery", {"type": "query"})
    graph.add_node("ResearchResult", {"type": "result"})
    graph.add_node("ResearchReport", {"type": "report"})
    
    # Add relationships
    graph.add_edge("ResearchQuery", "ResearchResult", "PRODUCES")
    graph.add_edge("ResearchResult", "ResearchReport", "CONTRIBUTES_TO")
    
    return graph

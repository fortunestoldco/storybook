from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

from langgraph.server import Server
from langgraph.runtime import RuntimeEnvironment
from langgraph.graph import StateGraph

from agents import AgentFactory
from mongodb import MongoDBManager
from state import NovelSystemState
from workflows import get_phase_workflow

# Load environment variables
load_dotenv()

# Initialize MongoDB and agent factory
mongo_manager = MongoDBManager()
agent_factory = AgentFactory(mongo_manager)

# Get workflow configuration from environment
workflow_module = os.getenv("LANGGRAPH_WORKFLOW_MODULES")
graph_functions = os.getenv("LANGGRAPH_GRAPH_FUNCTIONS", "").split("|")

# Create graph configurations
graph_configs = [f"{workflow_module}:{func}" for func in graph_functions]

# Initialize server
server = Server(
    graphs_config=",".join(graph_configs),
    runtime=RuntimeEnvironment(
        python_dependencies=[
            "langchain-anthropic",
            "langchain-openai",
            "langchain-mongodb",
            "langgraph"
        ],
        memory_limit=os.getenv("LANGGRAPH_RUNTIME_MEMORY_LIMIT", "4G"),
        timeout=int(os.getenv("LANGGRAPH_RUNTIME_TIMEOUT", "600"))
    )
)

# Register graphs as endpoints
@server.register("/initialize/{project_id}")
def get_initialization_graph(project_id: str) -> StateGraph:
    return get_phase_workflow("initialization", project_id, agent_factory)

@server.register("/develop/{project_id}")
def get_development_graph(project_id: str) -> StateGraph:
    """Get the development phase graph for a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The development phase graph.
    """
    return get_phase_workflow("development", project_id, agent_factory)

@server.register("/create/{project_id}")
def get_creation_graph(project_id: str) -> StateGraph:
    """Get the creation phase graph for a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The creation phase graph.
    """
    return get_phase_workflow("creation", project_id, agent_factory)

@server.register("/refine/{project_id}")
def get_refinement_graph(project_id: str) -> StateGraph:
    """Get the refinement phase graph for a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The refinement phase graph.
    """
    return get_phase_workflow("refinement", project_id, agent_factory)

@server.register("/finalize/{project_id}")
def get_finalization_graph(project_id: str) -> StateGraph:
    """Get the finalization phase graph for a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The finalization phase graph.
    """
    return get_phase_workflow("finalization", project_id, agent_factory)

if __name__ >= "__main__":
    server.serve(
        host="0.0.0.0",
        port=8000,
        workers=1
    )

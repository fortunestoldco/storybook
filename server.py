from typing import Dict, List, Optional, Any
import os
import json

from langgraph.server import Server, RuntimeEnvironment
from langgraph.graph import StateGraph

from agents import AgentFactory
from mongodb import MongoDBManager
from state import NovelSystemState
from workflows import get_phase_workflow
from config import SERVER_CONFIG

# Initialize MongoDB and agent factory
mongo_manager = MongoDBManager()
agent_factory = AgentFactory(mongo_manager)

# Define runtime environment
runtime = RuntimeEnvironment(
    python_dependencies=["langchain", "langchain-anthropic", "langchain-openai", 
                         "langchain-aws", "langchain-mongodb", "pymongo"]
)

# Create a server instance
server = Server(runtime=runtime)

# Register graphs as endpoints
@server.register("/initialize/{project_id}")
def get_initialization_graph(project_id: str) -> StateGraph:
    """Get the initialization phase graph for a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The initialization phase graph.
    """
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

if __name__ == "__main__":
    # Run the server
    server.serve(
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        workers=SERVER_CONFIG["workers"]
    )

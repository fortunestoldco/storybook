from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from langgraph.server import Server
from langgraph_runtime import RuntimeEnvironment
from langgraph.graph import StateGraph

from agents.factory import AgentFactory
from mongodb import MongoDBManager
from state import NovelSystemState
from workflows import get_phase_workflow

# Load environment variables
load_dotenv()

# Initialize MongoDB and agent factory
mongo_manager = MongoDBManager()
agent_factory = AgentFactory(mongo_manager)

# Initialize server
server = Server(
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

app = FastAPI()

@server.register
def initialize(project_id: str) -> StateGraph:
    """Get the initialization phase graph for a project."""
    return get_phase_workflow("initialization", project_id, agent_factory)

@server.register
def develop(project_id: str) -> StateGraph:
    """Get the development phase graph for a project."""
    return get_phase_workflow("development", project_id, agent_factory)

@server.register
def create(project_id: str) -> StateGraph:
    """Get the creation phase graph for a project."""
    return get_phase_workflow("creation", project_id, agent_factory)

@server.register
def refine(project_id: str) -> StateGraph:
    """Get the refinement phase graph for a project."""
    return get_phase_workflow("refinement", project_id, agent_factory)

@server.register
def finalize(project_id: str) -> StateGraph:
    """Get the finalization phase graph for a project."""
    return get_phase_workflow("finalization", project_id, agent_factory)

@server.register
def complete_novel(project_id: str) -> StateGraph:
    """Get the complete novel workflow graph for a project."""
    return get_phase_workflow("complete", project_id, agent_factory)

if __name__ == "__main__":
    server.serve(
        host="0.0.0.0",
        port=8000,
        workers=1
    )


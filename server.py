from typing import Dict, List, Optional, Any
import os
import json

from langgraph.serve import Server, RuntimeEnvironment
from langgraph.graph import StateGraph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from agents import AgentFactory
from mongodb import MongoDBManager
from state import NovelSystemState
from workflows import get_phase_workflow
from config import SERVER_CONFIG
from backend import get_default_backend_config

# Initialize MongoDB and agent factory
mongo_manager = MongoDBManager()
backend_config = get_default_backend_config()
agent_factory = AgentFactory(mongo_manager, backend_config)

# Define runtime environment
runtime = RuntimeEnvironment(
    python_dependencies=[
        "langchain", "langchain-anthropic", "langchain-openai",
        "langchain-aws", "langchain-mongodb", "pymongo",
        "langchain-google-vertexai", "langchain-azure-openai",
        "langchain-community", "python-dotenv"
    ]
)

# Initialize server
server = Server(runtime=runtime)

# Add CORS middleware to the langgraph server's app
server.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register graphs as endpoints
@server.register("/initialize/{project_id}")
def get_initialization_graph(project_id: str) -> StateGraph:
    """Get the initialization phase graph for a project."""
    config = {
        "metadata": {
            "project_id": project_id,
            "phase": "initialization",
            "agent_factory": agent_factory
        }
    }
    return get_phase_workflow(config)

@server.register("/develop/{project_id}")
def get_development_graph(project_id: str) -> StateGraph:
    """Get the development phase graph for a project."""
    config = {
        "metadata": {
            "project_id": project_id,
            "phase": "development",
            "agent_factory": agent_factory
        },
        "configurable": {
            "graph_name": f"storybook_{project_id}_development"
        }
    }
    return get_phase_workflow(config)

@server.register("/create/{project_id}")
def get_creation_graph(project_id: str) -> StateGraph:
    """Get the creation phase graph for a project."""
    config = {
        "metadata": {
            "project_id": project_id,
            "phase": "creation",
            "agent_factory": agent_factory
        },
        "configurable": {
            "graph_name": f"storybook_{project_id}_creation"
        }
    }
    return get_phase_workflow(config)

@server.register("/refine/{project_id}")
def get_refinement_graph(project_id: str) -> StateGraph:
    """Get the refinement phase graph for a project."""
    config = {
        "metadata": {
            "project_id": project_id,
            "phase": "refinement",
            "agent_factory": agent_factory
        },
        "configurable": {
            "graph_name": f"storybook_{project_id}_refinement"
        }
    }
    return get_phase_workflow(config)

@server.register("/finalize/{project_id}")
def get_finalization_graph(project_id: str) -> StateGraph:
    """Get the finalization phase graph for a project."""
    config = {
        "metadata": {
            "project_id": project_id,
            "phase": "finalization",
            "agent_factory": agent_factory
        },
        "configurable": {
            "graph_name": f"storybook_{project_id}_finalization"
        }
    }
    return get_phase_workflow(config)

# Add assistants API endpoint - Required for the assistants UI
@server.app.get("/api/v1/assistants")
async def get_assistants():
    """Get available assistants information for UI."""
    assistants = [
        {
            "id": "exec_director",
            "name": "Executive Director",
            "description": "Overall project manager for the novel writing process",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "creative_director",
            "name": "Creative Director",
            "description": "Manages creative aspects of the novel",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "content_director",
            "name": "Content Development Director",
            "description": "Manages content creation and drafting",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "editorial_director",
            "name": "Editorial Director",
            "description": "Manages editing and refinement",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "market_director",
            "name": "Market Alignment Director",
            "description": "Manages market positioning and audience targeting",
            "model": agent_factory.backend_config.provider
        }
    ]
    return assistants

# Try to mount static files (UI) if the directory exists
try:
    static_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.exists(static_directory):
        server.app.mount("/", StaticFiles(directory=static_directory, html=True), name="static")
        print(f"Mounted static files from {static_directory}")
except Exception as e:
    print(f"Could not mount static files: {str(e)}")

if __name__ == "__main__":
    # Run the server directly if this file is executed
    server.serve(
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"]
    )
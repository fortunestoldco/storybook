from typing import Dict, List, Optional, Any
import os
import json

from langgraph.server import LangGraphServer, RuntimeEnvironment
from langgraph.graph import StateGraph

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
graph_server = LangGraphServer()

# Register graphs as endpoints
@graph_server.register("/initialize/{project_id}")
def get_initialization_graph(project_id: str) -> StateGraph:
    """Get the initialization phase graph for a project."""
    config = {
        "metadata": {
            "project_id": project_id,
            "phase": "initialization",
            "agent_factory": agent_factory
        }
    }
    app = get_phase_workflow(config)
    return app

@graph_server.register("/develop/{project_id}")
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

@graph_server.register("/create/{project_id}")
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

@graph_server.register("/refine/{project_id}")
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

@graph_server.register("/finalize/{project_id}")
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

if __name__ == "__main__":
    # Run the server with the new format
    graph_server.serve(
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"]
    )
    serve.mount_asgi_app(
        app,
        server,
        runtime,
        path_prefix="/api/v1",
        include_middleware=True
    )

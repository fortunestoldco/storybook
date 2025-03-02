from typing import List, Dict, Optional
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from server import server
from agents import AgentFactory
from backend import get_default_backend_config
from config import SERVER_CONFIG

# Initialize the agent factory
agent_factory = AgentFactory(backend_config=get_default_backend_config())

# Get the FastAPI app instance
app = FastAPI(title="Novel Writing System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount LangGraph API with correct prefix - this is critical
app.mount("/api/v1", server.app)

# Define model for Assistant responses
class Assistant(BaseModel):
    id: str
    name: str
    description: str
    model: str

class AssistantSearchResponse(BaseModel):
    data: List[Assistant]
    has_more: bool = False
    object: str = "list"

# Add an assistant API directly to the FastAPI app for redundancy
@app.get("/assistants", response_model=AssistantSearchResponse)
async def get_assistants():
    """Get available assistants information for UI."""
    assistants_data = [
        {
            "id": "exec_director",
            "name": "Executive Director",
            "description": "Overall project manager for the novel writing process",
            "model": str(agent_factory.backend_config.provider)
        },
        {
            "id": "creative_director",
            "name": "Creative Director",
            "description": "Manages creative aspects of the novel",
            "model": str(agent_factory.backend_config.provider)
        },
        {
            "id": "content_director",
            "name": "Content Development Director",
            "description": "Manages content creation and drafting",
            "model": str(agent_factory.backend_config.provider)
        },
        {
            "id": "editorial_director",
            "name": "Editorial Director",
            "description": "Manages editing and refinement",
            "model": str(agent_factory.backend_config.provider)
        },
        {
            "id": "market_director",
            "name": "Market Alignment Director",
            "description": "Manages market positioning and audience targeting",
            "model": str(agent_factory.backend_config.provider)
        }
    ]
    
    # Convert to Assistant models
    assistants = [Assistant(**data) for data in assistants_data]
    
    return AssistantSearchResponse(data=assistants)

# Add a root redirect
@app.get("/")
async def root():
    return RedirectResponse(url="/index.html")

if __name__ == "__main__":
    import uvicorn
    # Try to mount static files if directory exists
    try:
        static_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        if os.path.exists(static_directory):
            app.mount("/", StaticFiles(directory=static_directory, html=True), name="static")
            print(f"Mounted static files from {static_directory}")
    except Exception as e:
        print(f"Could not mount static files: {str(e)}")
        
    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"]
    )
import json
import uuid
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ValidationError

from agents import AgentFactory
from state import ProjectState, NovelSystemState
from mongodb import MongoDBManager
from workflows import get_phase_workflow
from utils import generate_id, current_timestamp

import logging

app = FastAPI(title="Storybook API")
db = MongoDBManager()
agent_factory = AgentFactory(db)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectCreate(BaseModel):
class ProjectRequest(BaseModel):
    """Request model for creating a new project."""
    title: str
    genre: str
    target_audience: str
    word_count_target: int
    description: Optional[str] = None


class TaskRequest(BaseModel):
    """Request model for running a task."""
    task: str
    content: Optional[str] = None
    phase: Optional[str] = None
    editing_type: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for providing human feedback."""
    content: str
    type: str = "general"
    quality_scores: Optional[Dict[str, int]] = None


@app.post("/projects", response_model=Dict)
async def create_project(request: ProjectRequest) -> Dict:
    """Create a new project.
    
    Args:
        request: The project request.
        
    Returns:
        The created project.
    """
    try:
        project_id = generate_id()
        
        # Create initial project state
        project_state = ProjectState(
            project_id=project_id,
            title=request.title,
            genre=request.genre,
            target_audience=request.target_audience,
            word_count_target=request.word_count_target
        )
        
        # Save to MongoDB
        mongo_manager.save_state(project_id, project_state.dict())
        
        logger.info(f"Project created with ID: {project_id}")
        
        return {
            "project_id": project_id,
            "title": request.title,
            "status": "created",
            "current_phase": "initialization"
        }
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/projects/{project_id}", response_model=Dict)
async def get_project(project_id: str) -> Dict:
    """Get a project by ID."""
    try:
        project = mongo_manager.load_state(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_id}/run", response_model=Dict)
async def run_task(project_id: str, request: TaskRequest, background_tasks: BackgroundTasks) -> Dict:
    """Run a task for a project."""
    try:
        project_data = mongo_manager.load_state(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        phase = request.phase or project_data.get("current_phase", "initialization")
        workflow = get_phase_workflow(phase, project_id, agent_factory)
        
        background_tasks.add_task(
            workflow.invoke,
            {
                "title": project_data["title"],
                "task": request.task,
                "content": request.content,
                "phase": phase
            }
        )
        
        return {
            "project_id": project_id,
            "status": "running",
            "task": request.task,
            "phase": phase
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_id}/feedback", response_model=Dict)
async def add_feedback(project_id: str, request: FeedbackRequest) -> Dict:
    """Add human feedback to a project."""
    try:
        feedback = {
            "project_id": project_id,
            "content": request.content,
            "type": request.type,
            "quality_scores": request.quality_scores,
            "timestamp": current_timestamp()
        }
        mongo_manager.save_feedback(feedback)
        return {"status": "feedback_added", "feedback_id": str(feedback.get("_id"))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/status", response_model=Dict)
async def get_project_status(project_id: str) -> Dict:
    """Get project status."""
    try:
        project = mongo_manager.load_state(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {
            "project_id": project_id,
            "status": project.get("status", "unknown"),
            "current_phase": project.get("current_phase", "initialization"),
            "last_update": project.get("last_update", None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/manuscript", response_model=Dict)
async def get_manuscript(project_id: str) -> Dict:
    """Get project manuscript."""
    try:
        project = mongo_manager.load_state(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {
            "project_id": project_id,
            "title": project.get("title", ""),
            "manuscript": project.get("manuscript", ""),
            "version": project.get("version", "1.0")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ >= "__main__":
    import uvicorn
    from config import SERVER_CONFIG
    
    uvicorn.run("main:app", 
                host=SERVER_CONFIG["host"], 
                port=SERVER_CONFIG["port"], 
                workers=SERVER_CONFIG["workers"],
                log_level="info")

from typing import Dict, Any
from workflow import NovelWorkflow
from state import ProjectState
from mongodb import MongoDBManager

def create_novel(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for novel creation process."""
    # Initialize state
    state = ProjectState(
        project_id=input_data.get("project_id", "default"),
        title=input_data["title"],
        genre=input_data.get("genre", "fiction"),
        target_audience=input_data.get("target_audience", "general"),
        word_count_target=input_data.get("word_count_target", 50000)
    )
    
    # Initialize workflow
    workflow = NovelWorkflow(state)
    
    # Execute workflow until completion
    while state.current_phase != "complete":
        try:
            result = workflow.execute()
            print(f"Completed phase: {state.current_phase}")
            if result.get("status") == "error":
                break
        except Exception as e:
            print(f"Error in phase {state.current_phase}: {str(e)}")
            break
    
    return {
        "status": "complete" if state.current_phase == "complete" else "error",
        "final_state": state.dict(),
        "metrics": state.progress_metrics
    }

if __name__ == "__main__":
    # Example usage
    input_data = {
        "title": "The Test Novel",
        "genre": "science fiction",
        "target_audience": "young adult",
        "word_count_target": 60000
    }
    
    result = create_novel(input_data)
    print(f"Novel creation completed with status: {result['status']}")
    print(f"Final metrics: {result['metrics']}")

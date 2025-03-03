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

app = FastAPI(title="NovelSystem Langgraph Server")

mongo_manager = MongoDBManager()
agent_factory = AgentFactory(mongo_manager)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Get a project by ID.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The project.
    """
    try:
        # Load from MongoDB
        project_data = mongo_manager.load_state(project_id)
        
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        logger.info(f"Project retrieved with ID: {project_id}")
        
        return {
            "project_id": project_id,
            "title": project_data.get("title", ""),
            "current_phase": project_data.get("current_phase", "initialization"),
            "status": "active" if project_data.get("current_phase") != "complete" else "complete"
        }
    except Exception as e:
        logger.error(f"Error retrieving project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/projects/{project_id}/run", response_model=Dict)
async def run_task(project_id: str, request: TaskRequest, background_tasks: BackgroundTasks) -> Dict:
    """Run a task for a project.
    
    Args:
        project_id: ID of the project.
        request: The task request.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Status of the task execution.
    """
    try:
        # Load from MongoDB
        project_data = mongo_manager.load_state(project_id)
        
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create project state
        project_state = ProjectState(**project_data)
        
        # Use the phase from the request or the current phase
        phase = request.phase or project_state.current_phase
        
        # Get the workflow for the phase
        try:
            workflow = get_phase_workflow(phase, project_id, agent_factory)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Create initial state
        initial_state: NovelSystemState = {
            "project": project_state,
            "current_input": {
                "task": request.task,
                "content": request.content or "",
                "editing_type": request.editing_type or ""
            },
            "current_output": {},
            "messages": [],
            "errors": []
        }
        
        # Create a unique run ID
        run_id = generate_id()
        
        # Execute the workflow in the background
        def execute_workflow():
            try:
                # Run the workflow
                final_state = workflow.invoke(initial_state, {"recursion_limit": 25})
                
                # Save the final state
                mongo_manager.save_state(project_id, final_state["project"].dict())
                
                # Save execution record
                execution_record = {
                    "run_id": run_id,
                    "project_id": project_id,
                    "phase": phase,
                    "task": request.task,
                    "status": "completed",
                    "timestamp": current_timestamp(),
                    "messages": final_state["messages"]
                }
                mongo_manager.save_document(execution_record)
                
                logger.info(f"Task {request.task} completed for project {project_id}")
                
            except Exception as e:
                # Log the error
                error_message = str(e)
                logger.error(f"Error executing workflow: {error_message}")
                
                # Save error record
                error_record = {
                    "run_id": run_id,
                    "project_id": project_id,
                    "phase": phase,
                    "task": request.task,
                    "status": "failed",
                    "error": error_message,
                    "timestamp": current_timestamp()
                }
                mongo_manager.save_document(error_record)
        
        background_tasks.add_task(execute_workflow)
        
        logger.info(f"Task {request.task} started for project {project_id}")
        
        return {
            "project_id": project_id,
            "run_id": run_id,
            "task": request.task,
            "status": "running",
            "phase": phase
        }
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Error running task: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/projects/{project_id}/feedback", response_model=Dict)
async def add_feedback(project_id: str, request: FeedbackRequest) -> Dict:
    """Add human feedback to a project.
    
    Args:
        project_id: ID of the project.
        request: The feedback request.
        
    Returns:
        Status of the feedback submission.
    """
    try:
        # Load from MongoDB
        project_data = mongo_manager.load_state(project_id)
        
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create feedback document
        feedback = {
            "project_id": project_id,
            "content": request.content,
            "type": request.type,
            "quality_scores": request.quality_scores or {},
            "timestamp": current_timestamp()
        }
        
        # Save to MongoDB
        mongo_manager.save_feedback(feedback)
        
        # Update the project state with human feedback
        project_state = ProjectState(**project_data)
        project_state.human_feedback.append(feedback)
        
        # If quality scores are provided, update the quality assessment
        if request.quality_scores:
            for key, value in request.quality_scores.items():
                project_state.quality_assessment[key] = value
            
            # Also set human approval if high scores
            average_score = sum(request.quality_scores.values()) / len(request.quality_scores)
            if average_score >= 85:  # High score threshold
                project_state.quality_assessment["human_approved"] = True
        
        # Save updated state
        mongo_manager.save_state(project_id, project_state.dict())
        
        logger.info(f"Feedback added for project {project_id}")
        
        return {
            "project_id": project_id,
            "status": "feedback_added"
        }
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/projects/{project_id}/status", response_model=Dict)
async def get_project_status(project_id: str) -> Dict:
    """Get the status of a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        Status of the project.
    """
    try:
        # Load from MongoDB
        project_data = mongo_manager.load_state(project_id)
        
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_state = ProjectState(**project_data)
        
        logger.info(f"Status retrieved for project {project_id}")
        
        return {
            "project_id": project_id,
            "title": project_state.title,
            "current_phase": project_state.current_phase,
            "progress_metrics": project_state.progress_metrics,
            "quality_assessment": project_state.quality_assessment
        }
    except Exception as e:
        logger.error(f"Error retrieving project status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/projects/{project_id}/manuscript", response_model=Dict)
async def get_manuscript(project_id: str) -> Dict:
    """Get the manuscript for a project.
    
    Args:
        project_id: ID of the project.
        
    Returns:
        The manuscript.
    """
    try:
        # Load from MongoDB
        project_data = mongo_manager.load_state(project_id)
        
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_state = ProjectState(**project_data)
        
        logger.info(f"Manuscript retrieved for project {project_id}")
        
        return {
            "project_id": project_id,
            "title": project_state.title,
            "manuscript": project_state.manuscript
        }
    except Exception as e:
        logger.error(f"Error retrieving manuscript: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ >= "__main__":
    import uvicorn
    from config import SERVER_CONFIG
    
    uvicorn.run("main:app", 
                host=SERVER_CONFIG["host"], 
                port=SERVER_CONFIG["port"], 
                workers=SERVER_CONFIG["workers"],
                log_level="info")

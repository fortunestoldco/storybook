import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langserve import add_routes
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from models import Novel
from graph import storybook_graph, NovelState
from db_utils import MongoDBManager

# Initialize FastAPI app
app = FastAPI(title="Storybook Langgraph API")

# Initialize the MongoDB manager
db_manager = MongoDBManager()

# Add LangChain routes for the graph
add_routes(
    app,
    storybook_graph,
    path="/api/v1/novel-development",
)

class NovelInput(BaseModel):
    title: str
    author: str
    manuscript: str
    max_iterations: int = 3

class TaskStatus(BaseModel):
    task_id: str
    status: str
    current_stage: str = None
    error: str = None

# Store for tracking background tasks
task_store = {}

async def process_novel(novel_input: NovelInput, task_id: str):
    """Background task to process a novel through the graph"""
    try:
        # Create the novel object
        novel = Novel(
            title=novel_input.title,
            author=novel_input.author,
            manuscript=novel_input.manuscript
        )
        
        # Store the initial novel in the database
        novel_id = db_manager.store_novel(novel)
        
        # Update task status
        task_store[task_id] = {
            "status": "processing",
            "current_stage": "initial",
            "novel_id": novel_id,
            "error": None
        }
        
        # Set up the initial state for the graph
        initial_state: NovelState = {
            "novel": novel,
            "current_agent": "",
            "feedback": "",
            "error": "",
            "completed_agents": [],
            "iterations": 0,
            "max_iterations": novel_input.max_iterations
        }
        
        # Execute the graph
        for event in storybook_graph.stream(initial_state):
            current_state = event.get("state", {})
            
            # Update the task status
            task_store[task_id]["current_stage"] = current_state.get("current_agent", "processing")
            
            # Check for errors
            if current_state.get("error"):
                task_store[task_id]["error"] = current_state["error"]
                task_store[task_id]["status"] = "failed"
                return
        
        # Get the final state
        final_state = event.get("state", {})
        
        # Update the novel in the database
        db_manager.update_novel(novel_id, final_state["novel"])
        
        # Update task status to completed
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["current_stage"] = "finished"
        
        # Wait for all tracers to finish
        wait_for_all_tracers()
        
    except Exception as e:
        # Update task status to failed with error
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)

@app.post("/api/v1/novels", response_model=TaskStatus)
async def create_novel(novel_input: NovelInput, background_tasks: BackgroundTasks):
    """Start a Storybook task"""
    # Generate a task ID
    task_id = f"task_{len(task_store) + 1}"
    
    # Initialize task in store
    task_store[task_id] = {
        "status": "starting",
        "current_stage": "initial",
        "error": None
    }
    
    # Add the task to background tasks
    background_tasks.add_task(process_novel, novel_input, task_id)
    
    return TaskStatus(
        task_id=task_id,
        status="starting",
        current_stage="initial"
    )

@app.get("/api/v1/novels/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a Storybook task"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_store[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task_info["status"],
        current_stage=task_info["current_stage"],
        error=task_info["error"]
    )

@app.get("/api/v1/novels/{task_id}")
async def get_novel(task_id: str):
    """Get the processed novel"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_store[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Novel processing not completed")
    
    # Retrieve the novel from the database
    novel = db_manager.get_novel(task_info.get("novel_id"))
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found in database")
    
    return novel.model_dump()

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
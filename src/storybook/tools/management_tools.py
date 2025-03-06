from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS
from models.project import ProjectTimeline
from models.communication import Communication

@tool
def project_timeline_tracker(action: str, details: Dict[str, Any] = None) -> str:
    """View and adjust project deadlines.

    Args:
        action: The action to perform (view/update/add).
        details: Additional details for the action.

    Returns:
        Response about the timeline action.
    """
    timeline_collection = get_collection(COLLECTIONS["timelines"])
    
    if action == "view":
        project_id = details.get("project_id")
        if not project_id:
            return "Project ID is required to view timeline"
        
        timeline_doc = timeline_collection.find_one({"project_id": project_id})
        if not timeline_doc:
            return f"No timeline found for project {project_id}"
        
        timeline = ProjectTimeline.from_dict(timeline_doc)
        return f"Timeline for project {project_id}: {timeline.model_dump_json()}"
    
    elif action == "add":
        if not details:
            return "Details are required to add a timeline"
        
        timeline = ProjectTimeline(**details)
        timeline_collection.insert_one(timeline.to_dict())
        return f"Timeline added for project {timeline.project_id}"
    
    elif action == "update":
        project_id = details.get("project_id")
        if not project_id:
            return "Project ID is required to update timeline"
        
        # Remove project_id from details to avoid duplication
        update_details = details.copy()
        update_details.pop("project_id", None)
        
        # Update the updated_at field
        update_details["updated_at"] = datetime.utcnow()
        
        result = timeline_collection.update_one(
            {"project_id": project_id}, 
            {"$set": update_details}
        )
        
        if result.matched_count == 0:
            return f"No timeline found for project {project_id}"
        
        return f"Timeline updated for project {project_id}"
    
    else:
        return f"Unknown action: {action}"

@tool
def team_communication_hub(message: str, recipient: str, project_id: str = None, priority: int = 1) -> str:
    """Send directives to other agents.

    Args:
        message: The message to send.
        recipient: The recipient agent name.
        project_id: Optional project ID.
        priority: Message priority (1-5).

    Returns:
        Confirmation of message delivery.
    """
    comm_collection = get_collection(COLLECTIONS["communications"])
    
    communication = Communication(
        communication_id=str(uuid.uuid4()),
        project_id=project_id,
        sender="executive_director",
        recipient=recipient,
        message=message,
        message_type="directive",
        priority=priority,
        status="sent",
        timestamp=datetime.utcnow()
    )
    
    comm_collection.insert_one(communication.to_dict())
    
    return f"Message sent to {recipient}: {message}"

@tool
def progress_dashboard(project_id: str, component: str = None) -> str:
    """Monitor completion status of project components.

    Args:
        project_id: The project ID.
        component: Optional specific component to check.

    Returns:
        Progress status information.
    """
    # Check project status
    projects_collection = get_collection(COLLECTIONS["projects"])
    project = projects_collection.find_one({"project_id": project_id})
    
    if not project:
        return f"No project found with ID {project_id}"
    
    # Check chapters status if component is chapters or not specified
    if component in (None, "chapters"):
        chapters_collection = get_collection(COLLECTIONS["chapters"])
        chapters = list(chapters_collection.find({"project_id": project_id}))
        
        total_chapters = len(chapters)
        completed_chapters = sum(1 for ch in chapters if ch.get("status") == "completed")
        
        chapters_progress = f"Chapters: {completed_chapters}/{total_chapters} completed"
    else:
        chapters_progress = ""
    
    # Check character development if component is characters or not specified
    if component in (None, "characters"):
        characters_collection = get_collection(COLLECTIONS["characters"])
        characters = list(characters_collection.find({"project_id": project_id}))
        
        total_characters = len(characters)
        complete_profiles = sum(1 for ch in characters if ch.get("psychological_profile") is not None)
        
        characters_progress = f"Characters: {complete_profiles}/{total_characters} fully developed"
    else:
        characters_progress = ""
    
    # Check world building if component is world or not specified
    if component in (None, "world"):
        world_collection = get_collection(COLLECTIONS["world_building"])
        world_entries = list(world_collection.find({"project_id": project_id}))
        
        world_progress = f"World Building: {len(world_entries)} entries created"
    else:
        world_progress = ""
    
    # Compile the progress report
    overall_status = project.get("status", "unknown")
    
    progress_report = f"Project: {project.get('title', 'Untitled')}\n"
    progress_report += f"Status: {overall_status}\n"
    
    if chapters_progress:
        progress_report += f"{chapters_progress}\n"
    
    if characters_progress:
        progress_report += f"{characters_progress}\n"
    
    if world_progress:
        progress_report += f"{world_progress}\n"
    
    return progress_report

@tool
def resource_allocation_tool(project_id: str, resource: str, allocation: Dict[str, float]) -> str:
    """Assign resources to different aspects of the project.

    Args:
        project_id: The project ID.
        resource: The resource to allocate.
        allocation: Dictionary mapping project aspects to allocation percentages.

    Returns:
        Confirmation of resource allocation.
    """
    projects_collection = get_collection(COLLECTIONS["projects"])
    project = projects_collection.find_one({"project_id": project_id})
    
    if not project:
        return f"No project found with ID {project_id}"
    
    # Validate allocation percentages sum to 100%
    total_allocation = sum(allocation.values())
    if abs(total_allocation - 100.0) > 0.01:  # Allow for small floating-point errors
        return f"Total allocation must be 100%, got {total_allocation}%"
    
    # Update project metadata with the resource allocation
    if "metadata" not in project:
        project["metadata"] = {}
    
    if "resources" not in project["metadata"]:
        project["metadata"]["resources"] = {}
    
    project["metadata"]["resources"][resource] = allocation
    project["updated_at"] = datetime.utcnow()
    
    projects_collection.update_one(
        {"project_id": project_id},
        {"$set": {
            "metadata": project["metadata"],
            "updated_at": project["updated_at"]
        }}
    )
    
    return f"Resource '{resource}' allocated successfully for project {project_id}"

from ..db_config import get_collection, COLLECTIONS
from ..configuration import Configuration

@tool
def project_management_tool(action: str, project_id: str, project_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Manage project metadata and configuration."""
    collection = get_collection(COLLECTIONS["projects"])
    
    if action == "create":
        project_data["created_at"] = datetime.utcnow()
        project_data["updated_at"] = datetime.utcnow()
        result = collection.insert_one(project_data)
        return {"status": "success", "project_id": str(result.inserted_id)}
        
    elif action == "update":
        if not project_data:
            return {"status": "error", "message": "No update data provided"}

# Register tools with the registry
from storybook.tools import tool_registry
tool_registry.register_tool(project_timeline_tracker, "executive_director")
tool_registry.register_tool(team_communication_hub, "executive_director")
tool_registry.register_tool(progress_dashboard, "executive_director")
tool_registry.register_tool(resource_allocation_tool, "executive_director")
tool_registry.register_tool(project_management_tool, "executive_director")
from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def interactive_gantt_chart(project_id: str, action: str = "view", timeline: Dict[str, Any] = None) -> Dict[str, Any]:
    """Visualize the complete project timeline.
    
    Args:
        project_id: The project ID.
        action: Action to perform (view, create, update).
        timeline: Timeline data for create/update actions.
        
    Returns:
        Project timeline visualization.
    """
    timeline_collection = get_collection(COLLECTIONS["timelines"])
    
    if action == "view":
        # Retrieve existing timeline
        project_timeline = timeline_collection.find_one({"project_id": project_id}, {"_id": 0})
        
        if not project_timeline:
            return {
                "error": f"No timeline found for project {project_id}",
                "status": "error"
            }
        
        return {
            "timeline": project_timeline,
            "status": "success"
        }
    
    elif action == "create":
        if not timeline:
            # Create default timeline structure
            phases = ["planning", "outlining", "drafting", "revising", "finalizing"]
            tasks = []
            start_date = datetime.utcnow().isoformat()
            
            # Generate default tasks for each phase
            for i, phase in enumerate(phases):
                phase_start = i * 14  # 2 weeks per phase
                phase_end = (i + 1) * 14
                
                tasks.append({
                    "task_id": str(uuid.uuid4()),
                    "name": f"{phase.capitalize()} phase",
                    "phase": phase,
                    "start_day": phase_start,
                    "end_day": phase_end,
                    "progress": 0,
                    "dependencies": [] if i == 0 else [tasks[i-1]["task_id"]],
                    "assignee": "author"
                })
            
            critical_path = [task["task_id"] for task in tasks]
            
            timeline = {
                "project_id": project_id,
                "start_date": start_date,
                "tasks": tasks,
                "critical_path": critical_path,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        
        # Check if timeline already exists
        existing = timeline_collection.find_one({"project_id": project_id})
        if existing:
            return {
                "error": f"Timeline already exists for project {project_id}",
                "status": "error"
            }
        
        # Add creation timestamp if not present
        if "created_at" not in timeline:
            timeline["created_at"] = datetime.utcnow().isoformat()
        
        timeline["updated_at"] = datetime.utcnow().isoformat()
        
        # Insert new timeline
        timeline_collection.insert_one(timeline)
        
        return {
            "timeline": timeline,
            "status": "success"
        }
    
    elif action == "update":
        if not timeline:
            return {
                "error": "Timeline data required for update action",
                "status": "error"
            }
        
        # Check if timeline exists
        existing = timeline_collection.find_one({"project_id": project_id})
        if not existing:
            return {
                "error": f"No timeline found for project {project_id}",
                "status": "error"
            }
        
        # Update timestamp
        timeline["updated_at"] = datetime.utcnow().isoformat()
        
        # Update timeline
        timeline_collection.update_one(
            {"project_id": project_id},
            {"$set": timeline}
        )
        
        # Get updated timeline
        updated_timeline = timeline_collection.find_one({"project_id": project_id}, {"_id": 0})
        
        return {
            "timeline": updated_timeline,
            "status": "success"
        }
    
    else:
        return {
            "error": f"Unknown action: {action}",
            "status": "error"
        }

@tool
def milestone_tracker(project_id: str, action: str = "list", milestone: Dict[str, Any] = None) -> Dict[str, Any]:
    """Monitor critical deadlines and achievements.
    
    Args:
        project_id: The project ID.
        action: Action to perform (list, add, update, complete).
        milestone: Milestone data for add/update actions.
        
    Returns:
        Project milestones information.
    """
    milestone_collection = get_collection("project_milestones")
    
    if action == "list":
        # Retrieve all milestones for the project
        milestones = list(milestone_collection.find({"project_id": project_id}, {"_id": 0}))
        
        # Calculate progress
        total = len(milestones)
        completed = sum(1 for m in milestones if m.get("status") == "completed")
        progress = {
            "total": total,
            "completed": completed,
            "percentage": (completed / total * 100) if total > 0 else 0
        }
        
        # Generate alerts for approaching/overdue milestones
        alerts = []
        now = datetime.utcnow()
        
        for m in milestones:
            if m.get("status") != "completed" and "due_date" in m:
                try:
                    due_date = datetime.fromisoformat(m["due_date"])
                    days_remaining = (due_date - now).days
                    
                    if days_remaining < 0:
                        alerts.append({
                            "milestone_id": m.get("milestone_id"),
                            "name": m.get("name"),
                            "type": "overdue",
                            "days": abs(days_remaining)
                        })
                    elif days_remaining < 7:
                        alerts.append({
                            "milestone_id": m.get("milestone_id"),
                            "name": m.get("name"),
                            "type": "approaching",
                            "days": days_remaining
                        })
                except (ValueError, TypeError):
                    # Handle invalid date format
                    pass
        
        return {
            "milestones": milestones,
            "progress": progress,
            "alerts": alerts,
            "status": "success"
        }
    
    elif action == "add":
        if not milestone:
            return {
                "error": "Milestone data required for add action",
                "status": "error"
            }
        
        # Generate ID if not provided
        if "milestone_id" not in milestone:
            milestone["milestone_id"] = str(uuid.uuid4())
        
        # Add project ID
        milestone["project_id"] = project_id
        
        # Set defaults
        if "status" not in milestone:
            milestone["status"] = "pending"
        
        if "created_at" not in milestone:
            milestone["created_at"] = datetime.utcnow().isoformat()
        
        milestone["updated_at"] = datetime.utcnow().isoformat()
        
        # Insert milestone
        milestone_collection.insert_one(milestone)
        
        return {
            "milestone": milestone,
            "status": "success"
        }
    
    elif action == "update":
        if not milestone or "milestone_id" not in milestone:
            return {
                "error": "Milestone ID required for update action",
                "status": "error"
            }
        
        milestone_id = milestone["milestone_id"]
        
        # Check if milestone exists
        existing = milestone_collection.find_one({
            "milestone_id": milestone_id,
            "project_id": project_id
        })
        
        if not existing:
            return {
                "error": f"Milestone {milestone_id} not found",
                "status": "error"
            }
        
        # Update timestamp
        milestone["updated_at"] = datetime.utcnow().isoformat()
        
        # Update milestone
        milestone_collection.update_one(
            {"milestone_id": milestone_id, "project_id": project_id},
            {"$set": milestone}
        )
        
        # Get updated milestone
        updated_milestone = milestone_collection.find_one(
            {"milestone_id": milestone_id},
            {"_id": 0}
        )
        
        return {
            "milestone": updated_milestone,
            "status": "success"
        }
    
    elif action == "complete":
        if not milestone or "milestone_id" not in milestone:
            return {
                "error": "Milestone ID required for complete action",
                "status": "error"
            }
        
        milestone_id = milestone["milestone_id"]
        
        # Check if milestone exists
        existing = milestone_collection.find_one({
            "milestone_id": milestone_id,
            "project_id": project_id
        })
        
        if not existing:
            return {
                "error": f"Milestone {milestone_id} not found",
                "status": "error"
            }
        
        # Update milestone status
        milestone_collection.update_one(
            {"milestone_id": milestone_id, "project_id": project_id},
            {
                "$set": {
                    "status": "completed",
                    "completion_date": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )
        
        # Get updated milestone
        updated_milestone = milestone_collection.find_one(
            {"milestone_id": milestone_id},
            {"_id": 0}
        )
        
        return {
            "milestone": updated_milestone,
            "status": "success"
        }
    
    else:
        return {
            "error": f"Unknown action: {action}",
            "status": "error"
        }

@tool
def resource_allocation_calendar(project_id: str, action: str = "view", resources: Dict[str, Any] = None) -> Dict[str, Any]:
    """Manage time allocation across tasks.
    
    Args:
        project_id: The project ID.
        action: Action to perform (view, update).
        resources: Resource allocation data for update action.
        
    Returns:
        Resource allocation calendar.
    """
    resources_collection = get_collection("resource_allocations")
    
    if action == "view":
        # Retrieve existing allocations
        allocations = resources_collection.find_one({"project_id": project_id}, {"_id": 0})
        
        if not allocations:
            # Create default allocations
            default_allocations = {
                "project_id": project_id,
                "allocations": {
                    "writing": {"percentage": 60, "hours_per_week": 20},
                    "research": {"percentage": 20, "hours_per_week": 7},
                    "editing": {"percentage": 15, "hours_per_week": 5},
                    "planning": {"percentage": 5, "hours_per_week": 2}
                },
                "conflicts": [],
                "recommendations": [
                    "Allocate more time to research in early phases",
                    "Increase editing time in later phases"
                ],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            resources_collection.insert_one(default_allocations)
            allocations = default_allocations
        
        return {
            "allocations": allocations,
            "status": "success"
        }
    
    elif action == "update":
        if not resources:
            return {
                "error": "Resource allocation data required for update action",
                "status": "error"
            }
        
        # Check if allocations exist
        existing = resources_collection.find_one({"project_id": project_id})
        
        resources["updated_at"] = datetime.utcnow().isoformat()
        
        if existing:
            # Update existing allocations
            resources_collection.update_one(
                {"project_id": project_id},
                {"$set": resources}
            )
        else:
            # Create new allocations
            resources["project_id"] = project_id
            resources["created_at"] = datetime.utcnow().isoformat()
            resources_collection.insert_one(resources)
        
        # Get updated allocations
        updated_allocations = resources_collection.find_one({"project_id": project_id}, {"_id": 0})
        
        return {
            "allocations": updated_allocations,
            "status": "success"
        }
    
    else:
        return {
            "error": f"Unknown action: {action}",
            "status": "error"
        }

@tool
def dependency_mapper(project_id: str, tasks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Identify task relationships and critical paths.
    
    Args:
        project_id: The project ID.
        tasks: Optional list of tasks to analyze.
        
    Returns:
        Task dependency analysis.
    """
    dependency_collection = get_collection("task_dependencies")
    
    # Get tasks if not provided
    if not tasks:
        timeline_collection = get_collection(COLLECTIONS["timelines"])
        timeline = timeline_collection.find_one({"project_id": project_id})
        
        if not timeline:
            return {
                "error": f"No timeline found for project {project_id}",
                "status": "error"
            }
        
        tasks = timeline.get("tasks", [])
    
    # Build dependency map
    dependencies = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            continue
        
        dependencies[task_id] = {
            "name": task.get("name", "Unnamed Task"),
            "depends_on": task.get("dependencies", []),
            "duration": task.get("end_day", 0) - task.get("start_day", 0),
            "slack": 0  # Will calculate later
        }
    
    # Identify critical path
    # This is a simplified algorithm - a real implementation would use CPM
    critical_path = []
    bottlenecks = []
    
    # Find tasks with no dependencies
    start_tasks = [tid for tid, t in dependencies.items() if not t["depends_on"]]
    
    # Find terminal tasks (those that no other task depends on)
    terminal_tasks = []
    for tid in dependencies:
        is_terminal = True
        for _, t in dependencies.items():
            if tid in t["depends_on"]:
                is_terminal = False
                break
        if is_terminal:
            terminal_tasks.append(tid)
    
    # Identify a simple critical path - the longest path from start to finish
    # This is a placeholder implementation
    if start_tasks and terminal_tasks:
        # Find the path with the maximum total duration
        max_duration = 0
        max_path = []
        
        def find_paths(current, path, duration):
            path = path + [current]
            duration += dependencies[current]["duration"]
            
            if current in terminal_tasks:
                return [(path, duration)]
            
            paths = []
            for tid, task in dependencies.items():
                if current in task["depends_on"] and tid not in path:
                    paths.extend(find_paths(tid, path, duration))
            
            return paths or [(path, duration)]
        
        for start in start_tasks:
            paths = find_paths(start, [], 0)
            for path, duration in paths:
                if duration > max_duration:
                    max_duration = duration
                    max_path = path
        
        critical_path = max_path
    
    # Identify bottlenecks - tasks with multiple dependents
    for tid, task in dependencies.items():
        dependent_count = sum(1 for _, t in dependencies.items() if tid in t["depends_on"])
        
        if dependent_count > 1:
            bottlenecks.append({
                "task_id": tid,
                "name": task["name"],
                "dependent_count": dependent_count
            })
    
    # Store the analysis in MongoDB
    dependency_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "dependencies": dependencies,
        "critical_path": critical_path,
        "bottlenecks": bottlenecks,
        "created_at": datetime.utcnow().isoformat()
    }
    
    dependency_collection.insert_one(dependency_analysis)
    
    return {
        "dependencies": dependencies,
        "critical_path": critical_path,
        "bottlenecks": bottlenecks,
        "analysis_id": dependency_analysis["analysis_id"],
        "status": "success"
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    interactive_gantt_chart, milestone_tracker,
    resource_allocation_calendar, dependency_mapper
]:
    tool_registry.register_tool(tool_func, "timeline")
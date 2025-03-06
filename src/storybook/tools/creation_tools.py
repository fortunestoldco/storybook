from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from ..db_config import get_collection, COLLECTIONS
from ..models.chapter import Chapter
from ..models.scene import Scene
from ..models.dialogue import Dialogue

@tool
def chapter_structure_template(chapter_type: str, project_id: str) -> Dict[str, Any]:
    """Access approved chapter frameworks.
    
    Args:
        chapter_type: Type of chapter structure to use.
        project_id: The project ID.
        
    Returns:
        Template structure for the chapter type.
    """
    templates_collection = get_collection(COLLECTIONS["story_structures"])
    
    # Basic chapter templates
    templates = {
        "exposition": {
            "name": "Exposition Chapter",
            "sections": [
                {"name": "Setting Introduction", "description": "Establish time and place"},
                {"name": "Character Introduction", "description": "Introduce key characters"},
                {"name": "Background Information", "description": "Provide necessary context"}
            ]
        },
        "rising_action": {
            "name": "Rising Action Chapter",
            "sections": [
                {"name": "Problem Intensification", "description": "Escalate the conflict"},
                {"name": "Character Development", "description": "Show character growth or change"},
                {"name": "Subplot Development", "description": "Advance secondary storylines"}
            ]
        },
        "climax": {
            "name": "Climax Chapter",
            "sections": [
                {"name": "Main Confrontation", "description": "Major collision of forces"},
                {"name": "Crisis Point", "description": "Moment of highest tension"},
                {"name": "Decision/Action", "description": "Character's crucial choice"}
            ]
        },
        "falling_action": {
            "name": "Falling Action Chapter",
            "sections": [
                {"name": "Consequences", "description": "Results of the climax"},
                {"name": "Loose End Resolution", "description": "Addressing remaining questions"},
                {"name": "Character Reflection", "description": "Processing what has occurred"}
            ]
        },
        "resolution": {
            "name": "Resolution Chapter",
            "sections": [
                {"name": "New Normal", "description": "The changed status quo"},
                {"name": "Character Transformation", "description": "Final character state"},
                {"name": "Thematic Conclusion", "description": "Final statement on theme"}
            ]
        }
    }
    
    # Check if the requested chapter type exists
    if chapter_type not in templates:
        return {"error": f"Chapter type '{chapter_type}' not found. Available types: {', '.join(templates.keys())}"}
    
    # Save the template to MongoDB
    template_id = str(uuid.uuid4())
    template_data = {
        "template_id": template_id,
        "project_id": project_id,
        "template_type": "chapter",
        "chapter_type": chapter_type,
        "name": templates[chapter_type]["name"],
        "sections": templates[chapter_type]["sections"],
        "created_at": datetime.utcnow()
    }
    
    templates_collection.insert_one(template_data)
    
    return template_data

@tool
def scene_purpose_identifier(scene_content: str, project_id: str) -> Dict[str, Any]:
    """Clarify each scene's narrative function.
    
    Args:
        scene_content: The content of the scene.
        project_id: The project ID.
        
    Returns:
        Analysis of the scene's purpose.
    """
    scenes_collection = get_collection(COLLECTIONS["scenes"])
    
    # Create a record of the scene analysis
    scene_id = str(uuid.uuid4())
    
    # In a real implementation, you'd use NLP or an LLM to analyze
    # Here we're just using placeholder logic
    primary_purpose = "character_development"
    secondary_purposes = ["exposition", "foreshadowing"]
    
    # Store the analysis in MongoDB
    scene_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "scene_id": scene_id,
        "project_id": project_id,
        "content_snippet": scene_content[:200] + "..." if len(scene_content) > 200 else scene_content,
        "primary_purpose": primary_purpose,
        "secondary_purposes": secondary_purposes,
        "created_at": datetime.utcnow()
    }
    
    scenes_collection.insert_one(scene_analysis)
    
    return {
        "primary_purpose": primary_purpose,
        "secondary_purposes": secondary_purposes,
        "analysis_id": scene_analysis["analysis_id"]
    }

@tool
def dialogue_purpose_checker(dialogue: str, character_id: str, project_id: str) -> Dict[str, Any]:
    """Ensure dialogue serves narrative functions.
    
    Args:
        dialogue: The dialogue content.
        character_id: ID of the character speaking.
        project_id: The project ID.
        
    Returns:
        Analysis of the dialogue's purpose and effectiveness.
    """
    dialogue_collection = get_collection("dialogue_analysis")
    
    # In a real implementation, you'd use NLP or an LLM to analyze
    # Here we're using placeholder values
    purpose = "character_revelation"
    effectiveness = 0.85
    
    dialogue_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "character_id": character_id,
        "dialogue_content": dialogue,
        "purpose": purpose,
        "effectiveness": effectiveness,
        "recommendations": [
            "Consider adding more subtext",
            "Reflect character's background more clearly"
        ],
        "created_at": datetime.utcnow()
    }
    
    dialogue_collection.insert_one(dialogue_analysis)
    
    return {
        "purpose": purpose,
        "effectiveness": effectiveness,
        "recommendations": dialogue_analysis["recommendations"],
        "analysis_id": dialogue_analysis["analysis_id"]
    }

@tool
def continuity_database(action: str, project_id: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Track all established story facts.
    
    Args:
        action: Action to perform (add, get, update, list).
        project_id: The project ID.
        data: Continuity data for create/update actions.
        
    Returns:
        Action result.
    """
    continuity_collection = get_collection(COLLECTIONS["continuity_facts"])
    
    if action == "add":
        if not data:
            return {"error": "Continuity data is required for add action"}
        
        fact_id = str(uuid.uuid4())
        
        continuity_fact = {
            "fact_id": fact_id,
            "project_id": project_id,
            "entity_id": data.get("entity_id"),  # character_id, location_id, etc.
            "entity_type": data.get("entity_type", "general"),
            "category": data.get("category", "general"),
            "fact": data.get("fact", ""),
            "established_in": data.get("established_in", {}),  # e.g., {"chapter_id": "123", "scene_id": "456"}
            "last_referenced_in": data.get("last_referenced_in", {}),
            "importance": data.get("importance", "medium"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        continuity_collection.insert_one(continuity_fact)
        
        return {
            "action": action,
            "status": "success",
            "fact_id": fact_id
        }
    
    elif action == "get":
        if not data or "fact_id" not in data:
            return {"error": "Fact ID is required for get action"}
        
        fact_id = data["fact_id"]
        continuity_fact = continuity_collection.find_one(
            {"fact_id": fact_id, "project_id": project_id},
            {"_id": 0}
        )
        
        if not continuity_fact:
            return {"error": f"Fact {fact_id} not found"}
        
        return {
            "action": action,
            "status": "success",
            "fact": continuity_fact
        }
    
    elif action == "update":
        if not data or "fact_id" not in data:
            return {"error": "Fact ID is required for update action"}
        
        fact_id = data["fact_id"]
        
        # Remove fact_id from update data
        update_data = data.copy()
        update_data.pop("fact_id", None)
        update_data["updated_at"] = datetime.utcnow()
        
        result = continuity_collection.update_one(
            {"fact_id": fact_id, "project_id": project_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            return {"error": f"Fact {fact_id} not found"}
        
        return {
            "action": action,
            "status": "success",
            "fact_id": fact_id
        }
    
    elif action == "list":
        entity_id = data.get("entity_id") if data else None
        category = data.get("category") if data else None
        
        query = {"project_id": project_id}
        
        if entity_id:
            query["entity_id"] = entity_id
        
        if category:
            query["category"] = category
        
        continuity_facts = list(continuity_collection.find(query, {"_id": 0}))
        
        return {
            "action": action,
            "status": "success",
            "facts": continuity_facts
        }
    
    else:
        return {"error": f"Unknown action: {action}"}

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    chapter_structure_template,
    scene_purpose_identifier,
    dialogue_purpose_checker,
    continuity_database,
]:
    tool_registry.register_tool(tool_func, "creation")
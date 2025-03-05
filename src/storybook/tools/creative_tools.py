from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool, tool
import uuid
from datetime import datetime

from db_config import get_collection, COLLECTIONS, VECTOR_COLLECTIONS
from models.style import StyleGuide
from vector_search import VectorSearchManager
from ..utils.vector_search import VectorSearch, SearchResult

@tool
def creative_vision_board(action: str, project_id: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
    """Document and share artistic direction.
    
    Args:
        action: Action to perform (add, get, update, list).
        project_id: The project ID.
        content: Vision element content for add/update actions.
        
    Returns:
        Action result.
    """
    vision_collection = get_collection("creative_vision")
    
    if action == "add":
        if not content:
            return {"error": "Content is required for add action"}
        
        vision_element = {
            "element_id": str(uuid.uuid4()),
            "project_id": project_id,
            "type": content.get("type", "general"),
            "title": content.get("title", "Untitled Element"),
            "description": content.get("description", ""),
            "references": content.get("references", []),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        vision_collection.insert_one(vision_element)
        
        return {
            "action": action,
            "status": "success",
            "element_id": vision_element["element_id"]
        }
    
    elif action == "get":
        if not content or "element_id" not in content:
            return {"error": "Element ID is required for get action"}
        
        element_id = content["element_id"]
        vision_element = vision_collection.find_one(
            {"element_id": element_id, "project_id": project_id},
            {"_id": 0}
        )
        
        if not vision_element:
            return {"error": f"Element {element_id} not found"}
        
        return {
            "action": action,
            "status": "success",
            "vision_element": vision_element
        }
    
    elif action == "update":
        if not content or "element_id" not in content:
            return {"error": "Element ID is required for update action"}
        
        element_id = content["element_id"]
        
        # Remove element_id from update data
        update_data = content.copy()
        update_data.pop("element_id", None)
        update_data["updated_at"] = datetime.utcnow()
        
        result = vision_collection.update_one(
            {"element_id": element_id, "project_id": project_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            return {"error": f"Element {element_id} not found"}
        
        return {
            "action": action,
            "status": "success",
            "element_id": element_id
        }
    
    elif action == "list":
        element_type = content.get("type") if content else None
        
        query = {"project_id": project_id}
        if element_type:
            query["type"] = element_type
        
        vision_elements = list(vision_collection.find(query, {"_id": 0}))
        
        return {
            "action": action,
            "status": "success",
            "vision_elements": vision_elements
        }
    
    else:
        return {"error": f"Unknown action: {action}"}

@tool
def style_guide_creator(element: str, project_id: str, guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """Establish consistent creative standards.
    
    Args:
        element: Type of element (prose, dialogue, description, etc.).
        project_id: The project ID.
        guidelines: Style guidelines to establish.
        
    Returns:
        Created or updated style guide.
    """
    style_collection = get_collection(COLLECTIONS["style_guides"])
    
    # Check if a style guide already exists for this element
    existing_guide = style_collection.find_one({
        "project_id": project_id,
        "element": element
    })
    
    if existing_guide:
        # Update existing guide
        guide_id = existing_guide["guide_id"]
        
        style_collection.update_one(
            {"guide_id": guide_id},
            {"$set": {
                "guidelines": guidelines.get("guidelines", {}),
                "examples": guidelines.get("examples", []),
                "references": guidelines.get("references", []),
                "updated_at": datetime.utcnow()
            }}
        )
    else:
        # Create new guide
        guide_id = str(uuid.uuid4())
        
        style_guide = StyleGuide(
            guide_id=guide_id,
            project_id=project_id,
            element=element,
            guidelines=guidelines.get("guidelines", {}),
            examples=guidelines.get("examples", []),
            references=guidelines.get("references", [])
        )
        
        style_collection.insert_one(style_guide.to_dict())
    
    # Retrieve the updated/created guide
    style_guide = style_collection.find_one({"guide_id": guide_id}, {"_id": 0})
    
    return style_guide

@tool
def inspiration_repository(action: str, project_id: str, reference: Dict[str, Any] = None) -> Dict[str, Any]:
    """Store and organize creative references.
    
    Args:
        action: Action to perform (add, get, list, delete).
        project_id: The project ID.
        reference: Reference data for add action.
        
    Returns:
        Action result.
    """
    inspiration_collection = get_collection("inspiration_repository")
    
    if action == "add":
        if not reference:
            return {"error": "Reference data is required for add action"}
        
        # Create new reference
        reference_id = str(uuid.uuid4())
        
        inspiration_entry = {
            "reference_id": reference_id,
            "project_id": project_id,
            "title": reference.get("title", "Untitled Reference"),
            "source": reference.get("source", ""),
            "type": reference.get("type", "general"),
            "content": reference.get("content", ""),
            "url": reference.get("url"),
            "tags": reference.get("tags", []),
            "notes": reference.get("notes", ""),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        inspiration_collection.insert_one(inspiration_entry)
        
        return {
            "action": action,
            "status": "success",
            "reference_id": reference_id
        }
    
    elif action == "get":
        if not reference or "reference_id" not in reference:
            return {"error": "Reference ID is required for get action"}
        
        reference_id = reference["reference_id"]
        
        inspiration_entry = inspiration_collection.find_one(
            {"reference_id": reference_id, "project_id": project_id},
            {"_id": 0}
        )
        
        if not inspiration_entry:
            return {"error": f"Reference {reference_id} not found"}
        
        return {
            "action": action,
            "status": "success",
            "reference": inspiration_entry
        }
    
    elif action == "list":
        type_filter = reference.get("type") if reference else None
        tag_filter = reference.get("tag") if reference else None
        
        query = {"project_id": project_id}
        
        if type_filter:
            query["type"] = type_filter
        
        if tag_filter:
            query["tags"] = tag_filter
        
        references = list(inspiration_collection.find(query, {"_id": 0}))
        
        # Group references by type
        grouped_references = {}
        for ref in references:
            ref_type = ref.get("type", "general")
            if ref_type not in grouped_references:
                grouped_references[ref_type] = []
            grouped_references[ref_type].append(ref)
        
        return {
            "action": action,
            "status": "success",
            "references": references,
            "categories": grouped_references
        }
    
    elif action == "delete":
        if not reference or "reference_id" not in reference:
            return {"error": "Reference ID is required for delete action"}
        
        reference_id = reference["reference_id"]
        
        result = inspiration_collection.delete_one({
            "reference_id": reference_id,
            "project_id": project_id
        })
        
        if result.deleted_count == 0:
            return {"error": f"Reference {reference_id} not found"}
        
        return {
            "action": action,
            "status": "success",
            "reference_id": reference_id
        }
    
    else:
        return {"error": f"Unknown action: {action}"}

@tool
def concept_evaluation_matrix(concept: Dict[str, Any], project_id: str) -> Dict[str, Any]:
    """Assess creative ideas against vision.
    
    Args:
        concept: Concept to evaluate.
        project_id: The project ID.
        
    Returns:
        Evaluation results.
    """
    vision_collection = get_collection("creative_vision")
    
    # Get project vision elements
    vision_elements = list(vision_collection.find({"project_id": project_id}, {"_id": 0}))
    
    if not vision_elements:
        return {
            "error": "No vision elements found for this project. Use creative_vision_board to create them first."
        }
    
    # Simple evaluation logic - compare concept to vision elements
    evaluation = {}
    alignment_scores = []
    
    for element in vision_elements:
        element_type = element.get("type", "general")
        
        if element_type not in evaluation:
            evaluation[element_type] = {
                "alignment": 0.0,
                "notes": []
            }
        
        # Placeholder for actual evaluation logic
        # In a real implementation, this would use NLP or LLM to compare concept to vision
        alignment = 0.7  # Placeholder score
        alignment_scores.append(alignment)
        
        evaluation[element_type]["notes"].append(
            f"Alignment with '{element.get('title')}': {alignment:.2f}"
        )
        
        # Calculate average alignment for this element type
        evaluation[element_type]["alignment"] = sum(alignment_scores) / len(alignment_scores)
    
    # Calculate overall alignment score
    overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    
    # Generate recommendations based on alignment score
    recommendations = []
    
    if overall_alignment < 0.5:
        recommendations.append("This concept has significant deviation from the project vision.")
        recommendations.append("Consider revisiting the core creative direction.")
    elif overall_alignment < 0.7:
        recommendations.append("This concept has moderate alignment with the project vision.")
        recommendations.append("Some adjustments may be needed to better align with core elements.")
    else:
        recommendations.append("This concept aligns well with the project vision.")
        recommendations.append("Proceed with development while maintaining consistency.")
    
    return {
        "evaluation": evaluation,
        "alignment_score": overall_alignment,
        "recommendations": recommendations
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    creative_vision_board, 
    style_guide_creator,
    inspiration_repository, 
    concept_evaluation_matrix
]:
    tool_registry.register_tool(tool_func, "creative")
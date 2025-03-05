from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def revision_priority_matrix(content: str, project_id: str) -> Dict[str, Any]:
    """Identify most critical improvement areas.
    
    Args:
        content: The content to analyze.
        project_id: The project ID.
        
    Returns:
        Prioritized list of improvement areas.
    """
    revisions_collection = get_collection(COLLECTIONS["revisions"])
    
    # This would use NLP/LLM in a real implementation
    # Here we're using placeholder values
    priorities = {
        "pacing": 0.9,
        "character_consistency": 0.7,
        "dialogue_authenticity": 0.8,
        "description_vividness": 0.5,
        "plot_coherence": 0.6
    }
    
    # Store the analysis in MongoDB
    revision_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "content_type": "manuscript_section",
        "content_snippet": content[:200] + "..." if len(content) > 200 else content,
        "priorities": priorities,
        "recommendations": [
            "Focus on improving pacing in action sequences",
            "Check character dialogue for consistency with established voice",
            "Add more sensory details to key settings"
        ],
        "created_at": datetime.utcnow()
    }
    
    revisions_collection.insert_one(revision_analysis)
    
    return {
        "priorities": priorities,
        "recommendations": revision_analysis["recommendations"],
        "analysis_id": revision_analysis["analysis_id"]
    }

@tool
def style_enhancement_guide(text: str, project_id: str) -> Dict[str, Any]:
    """Access techniques for prose improvement.
    
    Args:
        text: The text to analyze for style enhancement.
        project_id: The project ID.
        
    Returns:
        Style enhancement suggestions.
    """
    style_collection = get_collection(COLLECTIONS["style_guides"])
    
    # This would use text analysis in a real implementation
    # Here we're using placeholder values
    suggestions = [
        "Replace passive voice with active voice",
        "Vary sentence length for better rhythm",
        "Add more sensory details"
    ]
    
    improvements = {
        "The man was seen by her.": "She saw the man.",
        "He walked to the store. He bought milk. He went home.": "He walked to the store, bought milk, and headed home.",
        "It was raining.": "Rain pattered against the windows, filling the room with its gentle rhythm."
    }
    
    # Store the analysis in MongoDB
    style_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "content_snippet": text[:200] + "..." if len(text) > 200 else text,
        "suggestions": suggestions,
        "improvements": improvements,
        "created_at": datetime.utcnow()
    }
    
    style_collection.insert_one(style_analysis)
    
    return {
        "suggestions": suggestions,
        "improvements": improvements,
        "analysis_id": style_analysis["analysis_id"]
    }

@tool
def character_arc_visualizer(character_id: str, project_id: str) -> Dict[str, Any]:
    """Map character changes over time.
    
    Args:
        character_id: ID of the character.
        project_id: The project ID.
        
    Returns:
        Visual representation of character development.
    """
    characters_collection = get_collection(COLLECTIONS["characters"])
    chapters_collection = get_collection(COLLECTIONS["chapters"])
    
    # Get character information
    character = characters_collection.find_one({"character_id": character_id, "project_id": project_id})
    
    if not character:
        return {"error": f"Character {character_id} not found"}
    
    # Get chapters where this character appears
    # In a real implementation, this would involve a more complex query
    chapters = list(chapters_collection.find({"project_id": project_id}))
    
    # Generate arc points - placeholder implementation
    arc_points = [
        {
            "chapter": 1,
            "emotional_state": "fearful",
            "key_relationship_status": {"character_id": "char123", "status": "hostile"},
            "growth_area": "confidence",
            "beliefs": ["World is threatening", "Cannot trust others"]
        },
        {
            "chapter": 5,
            "emotional_state": "cautious",
            "key_relationship_status": {"character_id": "char123", "status": "neutral"},
            "growth_area": "confidence",
            "beliefs": ["World has both threats and opportunities", "Some people can be trusted"]
        },
        {
            "chapter": 10,
            "emotional_state": "determined",
            "key_relationship_status": {"character_id": "char123", "status": "cooperative"},
            "growth_area": "confidence",
            "beliefs": ["Can overcome obstacles", "Building trust is worthwhile"]
        }
    ]
    
    # Store the character arc in MongoDB
    arc_document = {
        "arc_id": str(uuid.uuid4()),
        "character_id": character_id,
        "project_id": project_id,
        "character_name": character.get("name", "Unknown Character"),
        "arc_points": arc_points,
        "analysis": "Character shows clear growth from fearful to determined, with a key relationship changing from hostile to cooperative.",
        "created_at": datetime.utcnow()
    }
    
    character_arcs_collection = get_collection("character_arcs")
    character_arcs_collection.insert_one(arc_document)
    
    return {
        "character_name": character.get("name", "Unknown Character"),
        "arc_points": arc_points,
        "analysis": arc_document["analysis"],
        "arc_id": arc_document["arc_id"]
    }

@tool
def theme_mapping_system(project_id: str, themes: List[str] = None) -> Dict[str, Any]:
    """Visualize thematic elements throughout the narrative.
    
    Args:
        project_id: The project ID.
        themes: Optional list of specific themes to map.
        
    Returns:
        Map of themes across chapters.
    """
    themes_collection = get_collection("thematic_analysis")
    chapters_collection = get_collection(COLLECTIONS["chapters"])
    
    # Get project chapters
    chapters = list(chapters_collection.find({"project_id": project_id}, {"_id": 0, "chapter_id": 1, "title": 1, "number": 1}))
    chapters.sort(key=lambda x: x.get("number", 0))
    
    # Default themes if none provided
    if not themes:
        themes = ["redemption", "sacrifice", "identity", "power"]
    
    # Generate thematic occurrences - placeholder implementation
    occurrences = {}
    for theme in themes:
        occurrences[theme] = []
        for chapter in chapters:
            if chapter.get("number") % len(themes) == themes.index(theme) % len(themes):
                strength = 0.7  # This would be determined by analysis in a real implementation
            else:
                strength = 0.3
                
            occurrences[theme].append({
                "chapter_id": chapter.get("chapter_id"),
                "chapter_number": chapter.get("number"),
                "strength": strength
            })
    
    # Store the thematic analysis in MongoDB
    theme_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "themes": themes,
        "occurrences": occurrences,
        "insights": [
            f"'{themes[0]}' appears most prominently in the beginning and end, creating a thematic frame",
            f"'{themes[1]}' gradually increases in prominence, peaking in the climactic chapters"
        ],
        "created_at": datetime.utcnow()
    }
    
    themes_collection.insert_one(theme_analysis)
    
    return {
        "themes": themes,
        "occurrences": occurrences,
        "insights": theme_analysis["insights"],
        "analysis_id": theme_analysis["analysis_id"]
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    revision_priority_matrix,
    style_enhancement_guide,
    character_arc_visualizer,
    theme_mapping_system,
]:
    tool_registry.register_tool(tool_func, "refinement")
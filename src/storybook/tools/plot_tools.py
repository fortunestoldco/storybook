from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def plot_hole_detector(project_id: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
    """Identify logical inconsistencies in the plot.
    
    Args:
        project_id: The project ID.
        content: Optional content to analyze.
        
    Returns:
        Detected plot holes and recommendations.
    """
    plot_collection = get_collection(COLLECTIONS["plot_elements"])
    continuity_collection = get_collection(COLLECTIONS["continuity_facts"])
    
    # Get continuity facts for this project
    continuity_facts = list(continuity_collection.find({"project_id": project_id}, {"_id": 0}))
    
    # This would use NLP/LLM analysis in a real implementation
    # Here we're using placeholder values
    holes = [
        {
            "description": "Character knows information they couldn't have learned",
            "location": "Chapter 7",
            "severity": "major",
            "related_facts": [
                "Character A never met Character B in Chapter 3",
                "Yet Character A references specific details about Character B's past in Chapter 7"
            ]
        },
        {
            "description": "Object appears without being introduced",
            "location": "Chapter 12",
            "severity": "minor",
            "related_facts": [
                "The silver key is used to unlock the door",
                "But the key was never mentioned before this scene"
            ]
        }
    ]
    
    recommendations = [
        "Add a scene in Chapter 5 where Character A learns about Character B from a mutual acquaintance",
        "Introduce the silver key earlier, perhaps in Chapter 9 when the protagonist is gathering supplies"
    ]
    
    # Store the analysis in MongoDB
    plot_hole_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "holes": holes,
        "recommendations": recommendations,
        "created_at": datetime.utcnow()
    }
    
    plot_collection.insert_one(plot_hole_analysis)
    
    return {
        "holes": holes,
        "recommendations": recommendations,
        "analysis_id": plot_hole_analysis["analysis_id"]
    }

@tool
def plot_tension_graph(project_id: str, scene_id: str = None) -> Dict[str, Any]:
    """Visualize rising and falling action.
    
    Args:
        project_id: The project ID.
        scene_id: Optional specific scene to analyze.
        
    Returns:
        Tension graph and flow analysis.
    """
    tension_collection = get_collection("plot_tension")
    scenes_collection = get_collection(COLLECTIONS["scenes"])
    
    # Get scene data if scene_id provided
    scene = None
    if scene_id:
        scene = scenes_collection.find_one({"scene_id": scene_id, "project_id": project_id}, {"_id": 0})
        if not scene:
            return {"error": f"Scene {scene_id} not found"}
    
    # If no specific scene, analyze overall plot tension
    tension_points = []
    if scene:
        # Analyze single scene - would use text analysis in a real implementation
        tension_points = [
            {"position": 0.0, "tension_level": 0.2, "description": "Scene opening - low tension"},
            {"position": 0.3, "tension_level": 0.4, "description": "Complication introduced"},
            {"position": 0.7, "tension_level": 0.8, "description": "Confrontation"},
            {"position": 0.9, "tension_level": 0.6, "description": "Partial resolution"},
            {"position": 1.0, "tension_level": 0.7, "description": "Scene ends with lingering tension"}
        ]
    else:
        # Get all scenes from the project and analyze the overall plot
        scenes = list(scenes_collection.find({"project_id": project_id}, {"_id": 0}))
        
        # Would use more complex analysis in a real implementation
        chapter_count = len(set(s.get("chapter_id") for s in scenes))
        
        if chapter_count > 0:
            # Create points for a typical narrative arc
            tension_points = [
                {"position": 0.0, "chapter": 1, "tension_level": 0.2, "description": "Status quo"},
                {"position": 0.1, "chapter": 2, "tension_level": 0.4, "description": "Inciting incident"},
                {"position": 0.3, "chapter": int(chapter_count * 0.3), "tension_level": 0.5, "description": "End of Act I"},
                {"position": 0.5, "chapter": int(chapter_count * 0.5), "tension_level": 0.7, "description": "Midpoint"},
                {"position": 0.7, "chapter": int(chapter_count * 0.7), "tension_level": 0.6, "description": "Setback"},
                {"position": 0.8, "chapter": int(chapter_count * 0.8), "tension_level": 0.9, "description": "Climax builds"},
                {"position": 0.9, "chapter": int(chapter_count * 0.9), "tension_level": 1.0, "description": "Climax"},
                {"position": 1.0, "chapter": chapter_count, "tension_level": 0.3, "description": "Resolution"}
            ]
    
    # Generate flow analysis
    flow_analysis = {
        "overall_arc": "Classic three-act structure with clear rising action",
        "pacing_assessment": "Good build-up to climax, but midpoint could use stronger tension",
        "notable_features": [
            "Strong inciting incident with good tension spike",
            "Effective climax at 90% mark",
            "Suitable denouement length"
        ]
    }
    
    # Store the analysis in MongoDB
    tension_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "scene_id": scene_id,
        "tension_points": tension_points,
        "flow_analysis": flow_analysis,
        "created_at": datetime.utcnow()
    }
    
    tension_collection.insert_one(tension_analysis)
    
    return {
        "tension_points": tension_points,
        "flow_analysis": flow_analysis,
        "analysis_id": tension_analysis["analysis_id"]
    }

@tool
def subplot_integration_matrix(project_id: str, subplots: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map relationships between narrative threads.
    
    Args:
        project_id: The project ID.
        subplots: Optional list of subplots to analyze.
        
    Returns:
        Analysis of subplot integration.
    """
    subplots_collection = get_collection("subplot_analysis")
    
    # Get subplots if not provided
    if not subplots:
        plot_collection = get_collection(COLLECTIONS["plot_elements"])
        subplots = list(plot_collection.find(
            {"project_id": project_id, "element_type": "subplot"},
            {"_id": 0}
        ))
    
    # This would use actual story analysis in a real implementation
    # Here we're using placeholder values
    if not subplots:
        # Generate placeholder subplots
        subplots = [
            {"subplot_id": str(uuid.uuid4()), "name": "Romance subplot", "description": "Relationship between protagonist and love interest"},
            {"subplot_id": str(uuid.uuid4()), "name": "Family conflict", "description": "Protagonist's strained relationship with parent"},
            {"subplot_id": str(uuid.uuid4()), "name": "Career challenge", "description": "Protagonist's work-related obstacles"}
        ]
    
    connections = []
    for i, subplot1 in enumerate(subplots):
        for j, subplot2 in enumerate(subplots):
            if i != j:
                # Generate a connection strength between 0.1 and 0.9
                strength = 0.1 + ((hash(subplot1.get("name", "") + subplot2.get("name", "")) % 9) / 10)
                
                connections.append({
                    "subplot1_id": subplot1.get("subplot_id"),
                    "subplot1_name": subplot1.get("name"),
                    "subplot2_id": subplot2.get("subplot_id"),
                    "subplot2_name": subplot2.get("name"),
                    "strength": strength,
                    "connection_type": "thematic" if strength > 0.6 else "character-based",
                    "description": f"Connected through {'shared themes' if strength > 0.6 else 'character interactions'}"
                })
    
    # Calculate integration score - average of connection strengths
    integration_score = sum(c["strength"] for c in connections) / len(connections) if connections else 0
    
    # Store the analysis in MongoDB
    subplot_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "subplots": subplots,
        "connections": connections,
        "integration_score": integration_score,
        "recommendations": [
            "Strengthen connection between Family conflict and Career challenge",
            "Consider how the Romance subplot impacts the main plot"
        ],
        "created_at": datetime.utcnow()
    }
    
    subplots_collection.insert_one(subplot_analysis)
    
    return {
        "subplots": subplots,
        "connections": connections,
        "integration_score": integration_score,
        "recommendations": subplot_analysis["recommendations"],
        "analysis_id": subplot_analysis["analysis_id"]
    }

@tool
def scene_flow_analyzer(project_id: str, scenes: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate transitions and connections between scenes.
    
    Args:
        project_id: The project ID.
        scenes: Optional list of scenes to analyze.
        
    Returns:
        Scene flow analysis.
    """
    flow_collection = get_collection("scene_flow")
    
    # Get scenes if not provided
    if not scenes:
        scenes_collection = get_collection(COLLECTIONS["scenes"])
        scenes = list(scenes_collection.find(
            {"project_id": project_id},
            {"_id": 0, "scene_id": 1, "chapter_id": 1, "title": 1, "description": 1}
        ))
    
    # Would use text analysis in a real implementation
    # Here we're using placeholder values
    flow_quality = 0.75  # 0.0 to 1.0 scale
    
    transition_notes = []
    
    # Generate transition notes for sequential scenes
    if len(scenes) >= 2:
        scenes.sort(key=lambda s: (s.get("chapter_id", ""), s.get("scene_id", "")))
        
        for i in range(len(scenes) - 1):
            current_scene = scenes[i]
            next_scene = scenes[i+1]
            
            # Simple transition assessment
            transition_notes.append({
                "from_scene_id": current_scene.get("scene_id"),
                "from_scene_title": current_scene.get("title", f"Scene {i+1}"),
                "to_scene_id": next_scene.get("scene_id"),
                "to_scene_title": next_scene.get("title", f"Scene {i+2}"),
                "quality": 0.5 + (hash(str(i)) % 6) / 10,  # Random score between 0.5-1.0
                "new_chapter": current_scene.get("chapter_id") != next_scene.get("chapter_id"),
                "notes": "Good transition with clear cause-effect relationship" if (hash(str(i)) % 3 == 0) else 
                         "Transition lacks strong connection" if (hash(str(i)) % 3 == 1) else
                         "Abrupt transition that may confuse readers"
            })
    
    # Store the analysis in MongoDB
    flow_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "flow_quality": flow_quality,
        "transition_notes": transition_notes,
        "overall_assessment": "Generally smooth flow with some areas needing improvement",
        "recommendations": [
            "Add transitional phrase at the beginning of Scene 4",
            "Consider if Scene 7 would flow better earlier in the sequence"
        ],
        "created_at": datetime.utcnow()
    }
    
    flow_collection.insert_one(flow_analysis)
    
    return {
        "flow_quality": flow_quality,
        "transition_notes": transition_notes,
        "overall_assessment": flow_analysis["overall_assessment"],
        "recommendations": flow_analysis["recommendations"],
        "analysis_id": flow_analysis["analysis_id"]
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [plot_hole_detector, plot_tension_graph,
                 subplot_integration_matrix, scene_flow_analyzer]:
    tool_registry.register_tool(tool_func, "plot")
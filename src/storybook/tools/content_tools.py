from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def content_blueprint_generator(project_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create detailed development guidelines.
    
    Args:
        project_id: The project ID.
        parameters: Optional parameters for customizing the blueprint.
        
    Returns:
        Content development blueprint.
    """
    blueprint_collection = get_collection("content_blueprints")
    
    # Default parameters if none provided
    if not parameters:
        parameters = {
            "genre": ["fiction"],
            "target_length": "novel",
            "pov": "third_limited",
            "style": "contemporary"
        }
    
    # Generate blueprint based on parameters
    blueprint = {}
    guidelines = []
    milestones = []
    
    # Content structure guidelines
    if parameters.get("target_length") == "novel":
        blueprint["structure"] = {
            "target_word_count": 80000,
            "chapter_count": 25,
            "average_chapter_length": 3200,
            "recommended_scenes_per_chapter": 2.5
        }
    elif parameters.get("target_length") == "novella":
        blueprint["structure"] = {
            "target_word_count": 30000,
            "chapter_count": 12,
            "average_chapter_length": 2500,
            "recommended_scenes_per_chapter": 2
        }
    elif parameters.get("target_length") == "short_story":
        blueprint["structure"] = {
            "target_word_count": 7500,
            "chapter_count": 1,
            "average_chapter_length": 7500,
            "recommended_scenes_per_chapter": 3
        }
    
    # POV and style guidelines
    blueprint["narrative"] = {
        "pov": parameters.get("pov", "third_limited"),
        "style": parameters.get("style", "contemporary"),
        "tense": "past"
    }
    
    # Generate POV-specific guidelines
    if blueprint["narrative"]["pov"] == "first":
        guidelines.append("Maintain consistent narrator voice throughout")
        guidelines.append("Limit knowledge to what narrator character can observe or knows")
        guidelines.append("Consider reliability of narrator and how it affects story")
    elif blueprint["narrative"]["pov"] == "third_limited":
        guidelines.append("Focus on thoughts and perceptions of POV character in each scene")
        guidelines.append("Avoid head-hopping within scenes")
        guidelines.append("Use scene breaks for POV shifts")
    elif blueprint["narrative"]["pov"] == "third_omniscient":
        guidelines.append("Maintain consistent narrative voice while revealing multiple perspectives")
        guidelines.append("Clearly signal perspective shifts to avoid confusing readers")
        guidelines.append("Use omniscient perspective strategically for insights unavailable to characters")
    
    # Generate style-specific guidelines
    if blueprint["narrative"]["style"] == "contemporary":
        guidelines.append("Use clear, accessible language appropriate for modern readers")
        guidelines.append("Balance description with action and dialogue")
        guidelines.append("Aim for authenticity in dialogue and character interactions")
    elif blueprint["narrative"]["style"] == "literary":
        guidelines.append("Emphasize rich, evocative language and deeper thematic elements")
        guidelines.append("Develop complex character interiority")
        guidelines.append("Layer meaning through symbolism and subtext")
    elif blueprint["narrative"]["style"] == "minimalist":
        guidelines.append("Use sparse, economical prose with significant subtext")
        guidelines.append("Show rather than tell through concrete details")
        guidelines.append("Leave space for reader interpretation")
    
    # Generate genre-specific guidelines
    for genre in parameters.get("genre", []):
        if genre == "mystery":
            guidelines.append("Plant clues and red herrings throughout narrative")
            guidelines.append("Balance information revelation to maintain suspense")
            guidelines.append("Ensure fair-play with readers by making clues available")
        elif genre == "fantasy":
            guidelines.append("Establish consistent magic system rules early")
            guidelines.append("Integrate worldbuilding details organically within the narrative")
            guidelines.append("Balance exposition with action to avoid info-dumping")
        elif genre == "romance":
            guidelines.append("Develop compelling relationship chemistry through meaningful interactions")
            guidelines.append("Create credible emotional and external obstacles")
            guidelines.append("Ensure satisfying emotional payoff at relationship milestones")
    
    # Generate development milestones
    milestones = [
        {
            "name": "Initial concept development",
            "description": "Define core premise, main characters, and central conflict",
            "completion_criteria": ["Concept statement", "Character sketches", "Conflict description"]
        },
        {
            "name": "Structural planning",
            "description": "Develop plot outline, story beats, and character arcs",
            "completion_criteria": ["Chapter outline", "Story beat map", "Character arc descriptions"]
        },
        {
            "name": "Drafting phase",
            "description": f"Complete first draft of approximately {blueprint['structure']['target_word_count']} words",
            "completion_criteria": ["Full manuscript draft", "All major plot points addressed"]
        },
        {
            "name": "Revision phase",
            "description": "Address structural and content issues in the manuscript",
            "completion_criteria": ["Revised manuscript", "Feedback incorporation log"]
        },
        {
            "name": "Polishing phase",
            "description": "Refine prose, dialogue, and scene details",
            "completion_criteria": ["Final manuscript", "Style consistency check"]
        }
    ]
    
    # Store the blueprint in MongoDB
    blueprint_document = {
        "blueprint_id": str(uuid.uuid4()),
        "project_id": project_id,
        "parameters": parameters,
        "blueprint": blueprint,
        "guidelines": guidelines,
        "milestones": milestones,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    blueprint_collection.insert_one(blueprint_document)
    
    return {
        "blueprint": blueprint,
        "guidelines": guidelines,
        "milestones": milestones,
        "blueprint_id": blueprint_document["blueprint_id"]
    }

@tool
def component_integration_tracker(project_id: str, components: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Monitor how elements work together.
    
    Args:
        project_id: The project ID.
        components: Optional list of components to track.
        
    Returns:
        Component integration analysis.
    """
    integration_collection = get_collection("component_integration")
    
    # Get components if not provided
    if not components:
        # In a real implementation, this would collect various elements from the project
        # For now, we'll create placeholder components
        components = [
            {"type": "character_arc", "id": str(uuid.uuid4()), "name": "Protagonist's growth journey"},
            {"type": "plot_thread", "id": str(uuid.uuid4()), "name": "Main mystery investigation"},
            {"type": "theme", "id": str(uuid.uuid4()), "name": "Trust and betrayal"},
            {"type": "setting", "id": str(uuid.uuid4()), "name": "Coastal town environment"}
        ]
    
    # Analyze integration status
    integration_status = {}
    for comp in components:
        comp_type = comp.get("type")
        comp_id = comp.get("id")
        comp_name = comp.get("name")
        
        if not (comp_type and comp_id and comp_name):
            continue
            
        if comp_type not in integration_status:
            integration_status[comp_type] = {}
        
        # Generate integration metrics (this would be based on actual analysis in a real implementation)
        integration_status[comp_type][comp_id] = {
            "name": comp_name,
            "overall_integration": 0.5 + (hash(comp_id) % 50) / 100,  # Random score between 0.5-1.0
            "chapter_presence": [i+1 for i in range(20) if hash(comp_id + str(i)) % 3 == 0],  # Random chapters
            "integration_metrics": {
                "relevance": 0.5 + (hash(comp_id + "rel") % 50) / 100,
                "consistency": 0.5 + (hash(comp_id + "con") % 50) / 100,
                "impact": 0.5 + (hash(comp_id + "imp") % 50) / 100
            }
        }
    
    # Identify issues
    issues = []
    for comp_type, comps in integration_status.items():
        for comp_id, metrics in comps.items():
            if metrics["overall_integration"] < 0.7:
                issues.append({
                    "component_type": comp_type,
                    "component_id": comp_id,
                    "component_name": metrics["name"],
                    "issue": "Low overall integration",
                    "severity": "medium" if metrics["overall_integration"] < 0.6 else "low"
                })
            
            if metrics["integration_metrics"]["consistency"] < 0.7:
                issues.append({
                    "component_type": comp_type,
                    "component_id": comp_id,
                    "component_name": metrics["name"],
                    "issue": "Inconsistent presence throughout narrative",
                    "severity": "high" if metrics["integration_metrics"]["consistency"] < 0.6 else "medium"
                })
    
    # Generate recommendations
    recommendations = []
    for issue in issues:
        if issue["issue"] == "Low overall integration":
            recommendations.append(f"Strengthen {issue['component_name']} integration by connecting it more clearly to other elements")
        elif issue["issue"] == "Inconsistent presence throughout narrative":
            recommendations.append(f"Ensure {issue['component_name']} appears more consistently throughout the narrative")
    
    # Add general recommendations
    if len(components) > 1:
        recommendations.append("Consider how thematic elements connect to character development")
        recommendations.append("Ensure setting details reinforce the emotional tone of scenes")
    
    # Store the analysis in MongoDB
    integration_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "components": components,
        "integration_status": integration_status,
        "issues": issues,
        "recommendations": recommendations,
        "created_at": datetime.utcnow().isoformat()
    }
    
    integration_collection.insert_one(integration_analysis)
    
    return {
        "integration_status": integration_status,
        "issues": issues,
        "recommendations": recommendations,
        "analysis_id": integration_analysis["analysis_id"]
    }

@tool
def content_evaluation_matrix(project_id: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
    """Assess elements against development criteria.
    
    Args:
        project_id: The project ID.
        content: Optional content to evaluate.
        
    Returns:
        Content evaluation results.
    """
    evaluation_collection = get_collection("content_evaluations")
    
    # Define evaluation criteria
    criteria = {
        "character": [
            "depth", "authenticity", "arc_development", "distinctiveness"
        ],
        "plot": [
            "structure", "pacing", "tension", "resolution", "originality"
        ],
        "setting": [
            "vividness", "consistency", "integration", "uniqueness"
        ],
        "theme": [
            "clarity", "development", "relevance", "subtlety"
        ],
        "dialogue": [
            "authenticity", "purpose", "characterization", "flow"
        ],
        "prose": [
            "clarity", "style", "rhythm", "imagery"
        ]
    }
    
    # Generate evaluation (this would use actual analysis in a real implementation)
    evaluation = {}
    scores = {}
    improvements = []
    
    for category, criteria_list in criteria.items():
        evaluation[category] = {}
        scores[category] = 0
        
        for criterion in criteria_list:
            # Generate score between 0.5 and 0.95
            score = 0.5 + (hash(category + criterion + project_id) % 46) / 100
            
            evaluation[category][criterion] = {
                "score": score,
                "strengths": [f"Good {criterion} in {category}"],
                "weaknesses": [f"Could improve {criterion} in {category}"] if score < 0.7 else []
            }
            
            # Add to improvements if score is low
            if score < 0.7:
                improvements.append({
                    "category": category,
                    "criterion": criterion,
                    "current_score": score,
                    "suggestion": f"Strengthen {category} {criterion} by focusing on more detailed development"
                })
            
            scores[category] += score
        
        # Calculate average score for the category
        scores[category] /= len(criteria_list)
    
    # Store the evaluation in MongoDB
    evaluation_document = {
        "evaluation_id": str(uuid.uuid4()),
        "project_id": project_id,
        "evaluation": evaluation,
        "scores": scores,
        "improvements": improvements,
        "created_at": datetime.utcnow().isoformat()
    }
    
    evaluation_collection.insert_one(evaluation_document)
    
    return {
        "evaluation": evaluation,
        "scores": scores,
        "improvements": improvements,
        "evaluation_id": evaluation_document["evaluation_id"]
    }

@tool
def balance_analyzer(project_id: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
    """Ensure proper distribution of content types.
    
    Args:
        project_id: The project ID.
        content: Optional content to analyze.
        
    Returns:
        Content balance analysis.
    """
    balance_collection = get_collection("content_balance")
    
    # This would analyze actual content in a real implementation
    # Here we're creating placeholder analysis
    
    # Generate distribution analysis
    distribution = {
        "dialogue_vs_narrative": {
            "dialogue_percentage": 35,
            "narrative_percentage": 65,
            "target_range": {"dialogue": "30-40%", "narrative": "60-70%"},
            "assessment": "Within ideal range"
        },
        "action_vs_reflection": {
            "action_percentage": 45,
            "reflection_percentage": 55,
            "target_range": {"action": "40-60%", "reflection": "40-60%"},
            "assessment": "Well balanced"
        },
        "character_vs_plot": {
            "character_focus_percentage": 30,
            "plot_focus_percentage": 70,
            "target_range": {"character": "40-50%", "plot": "50-60%"},
            "assessment": "Plot-heavy; consider more character development"
        },
        "description_density": {
            "heavy_description_percentage": 25,
            "moderate_description_percentage": 50,
            "minimal_description_percentage": 25,
            "target_range": {"heavy": "20-30%", "moderate": "40-60%", "minimal": "20-30%"},
            "assessment": "Well balanced description density"
        }
    }
    
    # Identify imbalances
    imbalances = []
    
    if distribution["character_vs_plot"]["character_focus_percentage"] < 40:
        imbalances.append({
            "type": "character_vs_plot",
            "description": "Insufficient character development relative to plot",
            "severity": "medium",
            "current": f"{distribution['character_vs_plot']['character_focus_percentage']}% character focus",
            "target": distribution["character_vs_plot"]["target_range"]["character"]
        })
    
    # Generate suggestions
    suggestions = []
    
    for imbalance in imbalances:
        if imbalance["type"] == "character_vs_plot":
            suggestions.append("Add more scenes focused on character development and introspection")
            suggestions.append("Incorporate character reactions to plot events")
            suggestions.append("Develop secondary character arcs more fully")
    
    # Add general balance suggestions
    suggestions.append("Ensure each scene serves multiple purposes (character, plot, theme)")
    suggestions.append("Vary rhythm between dialogue-heavy and narrative-heavy sections")
    
    # Store the analysis in MongoDB
    balance_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "distribution": distribution,
        "imbalances": imbalances,
        "suggestions": suggestions,
        "created_at": datetime.utcnow().isoformat()
    }
    
    balance_collection.insert_one(balance_analysis)
    
    return {
        "distribution": distribution,
        "imbalances": imbalances,
        "suggestions": suggestions,
        "analysis_id": balance_analysis["analysis_id"]
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    content_blueprint_generator, component_integration_tracker,
    content_evaluation_matrix, balance_analyzer
]:
    tool_registry.register_tool(tool_func, "content")
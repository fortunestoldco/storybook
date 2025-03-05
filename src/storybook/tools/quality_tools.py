from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def quality_rubric_generator(project_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create customized evaluation metrics.
    
    Args:
        project_id: The project ID.
        parameters: Optional parameters for rubric customization.
        
    Returns:
        Generated quality rubric.
    """
    quality_collection = get_collection(COLLECTIONS["quality_metrics"])
    
    # Default parameters if none provided
    if not parameters:
        parameters = {
            "genre": ["fiction"],
            "audience": ["adult"],
            "focus_areas": ["plot", "character", "prose"]
        }
    
    # Generate rubric based on parameters
    rubric = {}
    metrics = []
    
    # Plot metrics
    if "plot" in parameters.get("focus_areas", []):
        rubric["plot"] = {
            "pacing": {"description": "Appropriate speed of story progression", "weight": 0.2},
            "coherence": {"description": "Logical flow of events", "weight": 0.3},
            "tension": {"description": "Effective building and release of tension", "weight": 0.2},
            "resolution": {"description": "Satisfying conclusion to plot threads", "weight": 0.3}
        }
        metrics.extend(["plot.pacing", "plot.coherence", "plot.tension", "plot.resolution"])
    
    # Character metrics
    if "character" in parameters.get("focus_areas", []):
        rubric["character"] = {
            "development": {"description": "Growth and change over time", "weight": 0.25},
            "consistency": {"description": "Believable and coherent behavior", "weight": 0.25},
            "uniqueness": {"description": "Distinctive and memorable qualities", "weight": 0.25},
            "motivation": {"description": "Clear and compelling desires/goals", "weight": 0.25}
        }
        metrics.extend(["character.development", "character.consistency", "character.uniqueness", "character.motivation"])
    
    # Prose metrics
    if "prose" in parameters.get("focus_areas", []):
        rubric["prose"] = {
            "clarity": {"description": "Clear and understandable writing", "weight": 0.3},
            "style": {"description": "Distinctive and appropriate voice", "weight": 0.2},
            "imagery": {"description": "Effective use of description and sensory detail", "weight": 0.25},
            "mechanics": {"description": "Correct grammar, spelling, and punctuation", "weight": 0.25}
        }
        metrics.extend(["prose.clarity", "prose.style", "prose.imagery", "prose.mechanics"])
    
    # Add genre-specific guidelines
    guidelines = []
    for genre in parameters.get("genre", []):
        if genre == "mystery":
            guidelines.append("Fair-play: All clues should be available to the reader")
            guidelines.append("Pacing: Reveals should be timed for maximum impact")
        elif genre == "romance":
            guidelines.append("Character chemistry: Connection should feel authentic")
            guidelines.append("Emotional arc: Clear development of the relationship")
        elif genre == "fantasy":
            guidelines.append("World-building: Consistent and compelling setting")
            guidelines.append("Magic systems: Clear rules and limitations")
    
    # Store the rubric in MongoDB
    rubric_document = {
        "rubric_id": str(uuid.uuid4()),
        "project_id": project_id,
        "parameters": parameters,
        "rubric": rubric,
        "metrics": metrics,
        "guidelines": guidelines,
        "created_at": datetime.utcnow()
    }
    
    quality_collection.insert_one(rubric_document)
    
    return {
        "rubric": rubric,
        "metrics": metrics,
        "guidelines": guidelines,
        "rubric_id": rubric_document["rubric_id"]
    }

@tool
def comparative_analysis_tool(project_id: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
    """Benchmark against published works.
    
    Args:
        project_id: The project ID.
        content: Optional content to benchmark.
        
    Returns:
        Comparison results against published works.
    """
    comparison_collection = get_collection("quality_comparisons")
    
    # Get comparison works - this would be based on genre/style in a real implementation
    # Here we're using placeholder values
    comparisons = [
        {
            "title": "Award-Winning Novel 1",
            "author": "Established Author",
            "areas": {
                "prose_quality": {"score": 9, "project_score": 7},
                "character_depth": {"score": 8, "project_score": 8},
                "plot_construction": {"score": 9, "project_score": 6}
            }
        },
        {
            "title": "Bestselling Novel 2",
            "author": "Popular Author",
            "areas": {
                "prose_quality": {"score": 7, "project_score": 7},
                "character_depth": {"score": 7, "project_score": 8},
                "plot_construction": {"score": 8, "project_score": 6}
            }
        }
    ]
    
    # Calculate strengths and weaknesses
    strengths = []
    areas_for_improvement = []
    
    for comparison in comparisons:
        for area, scores in comparison["areas"].items():
            if scores.get("project_score", 0) >= scores.get("score", 0):
                if {"area": area, "description": f"On par or better than {comparison['title']}"} not in strengths:
                    strengths.append({
                        "area": area,
                        "description": f"On par or better than {comparison['title']}"
                    })
            else:
                gap = scores.get("score", 0) - scores.get("project_score", 0)
                if gap >= 2 and {"area": area, "description": f"Significantly below {comparison['title']}"} not in areas_for_improvement:
                    areas_for_improvement.append({
                        "area": area,
                        "description": f"Significantly below {comparison['title']}"
                    })
    
    # Store the analysis in MongoDB
    comparison_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "comparisons": comparisons,
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "overall_assessment": "The manuscript shows strong character work but needs improvement in plot construction.",
        "created_at": datetime.utcnow()
    }
    
    comparison_collection.insert_one(comparison_analysis)
    
    return {
        "comparisons": comparisons,
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "overall_assessment": comparison_analysis["overall_assessment"],
        "analysis_id": comparison_analysis["analysis_id"]
    }

@tool
def quality_trend_tracker(project_id: str, revisions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Monitor quality improvements over revisions.
    
    Args:
        project_id: The project ID.
        revisions: Optional list of revision data.
        
    Returns:
        Quality trend analysis.
    """
    trends_collection = get_collection("quality_trends")
    
    # Get revision data if not provided
    if not revisions:
        revisions_collection = get_collection(COLLECTIONS["revisions"])
        revisions = list(revisions_collection.find(
            {"project_id": project_id},
            {"_id": 0, "revision_number": 1, "metrics": 1, "timestamp": 1}
        ).sort("revision_number", 1))
    
    # Generate trend data - this would use actual metrics in a real implementation
    trends = {}
    improvements = []
    concerns = []
    
    if revisions:
        # Plot the trends for each metric
        metrics = revisions[0].get("metrics", {}).keys()
        
        for metric in metrics:
            trends[metric] = [
                {
                    "revision": r.get("revision_number", i+1),
                    "score": r.get("metrics", {}).get(metric, 0),
                    "timestamp": r.get("timestamp", datetime.utcnow().isoformat())
                }
                for i, r in enumerate(revisions)
            ]
            
            # Determine if improving or declining
            if len(trends[metric]) >= 2:
                first_score = trends[metric][0]["score"]
                last_score = trends[metric][-1]["score"]
                
                if last_score > first_score:
                    improvements.append({
                        "metric": metric,
                        "change": last_score - first_score,
                        "percentage": ((last_score - first_score) / first_score) * 100 if first_score > 0 else 0
                    })
                elif last_score < first_score:
                    concerns.append({
                        "metric": metric,
                        "change": first_score - last_score,
                        "percentage": ((first_score - last_score) / first_score) * 100 if first_score > 0 else 0
                    })
    
    # Store the analysis in MongoDB
    trend_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "trends": trends,
        "improvements": improvements,
        "concerns": concerns,
        "summary": f"Overall quality is {len(improvements) > len(concerns) and len(improvements) > 0 and 'improving' or 'stable'} across {len(trends)} tracked metrics.",
        "created_at": datetime.utcnow()
    }
    
    trends_collection.insert_one(trend_analysis)
    
    return {
        "trends": trends,
        "improvements": improvements,
        "concerns": concerns,
        "summary": trend_analysis["summary"],
        "analysis_id": trend_analysis["analysis_id"]
    }

@tool
def assessment_matrix(project_id: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluate across different quality dimensions.
    
    Args:
        project_id: The project ID.
        content: Optional content to evaluate.
        
    Returns:
        Multi-dimensional quality assessment.
    """
    assessment_collection = get_collection("quality_assessments")
    
    # Define quality dimensions
    dimensions = [
        "plot_coherence",
        "character_development",
        "prose_quality",
        "pacing",
        "dialogue",
        "worldbuilding",
        "thematic_depth",
        "emotional_impact"
    ]
    
    # Generate assessments - this would use actual analysis in a real implementation
    assessments = {}
    scores = {}
    
    for dimension in dimensions:
        # Simulate different scoring for each dimension
        score = 0.5 + (hash(dimension + project_id) % 50) / 100  # Generate pseudo-random scores between 0.5-1.0
        
        assessments[dimension] = {
            "strengths": [f"Strong {dimension.replace('_', ' ')} in chapters 3-5"],
            "weaknesses": [f"Inconsistent {dimension.replace('_', ' ')} in final chapter"]
        }
        
        scores[dimension] = score
    
    # Generate recommendations
    recommendations = []
    for dimension, score in scores.items():
        if score < 0.7:
            recommendations.append(f"Focus on improving {dimension.replace('_', ' ')}")
    
    # Store the assessment in MongoDB
    assessment_document = {
        "assessment_id": str(uuid.uuid4()),
        "project_id": project_id,
        "dimensions": dimensions,
        "assessments": assessments,
        "scores": scores,
        "recommendations": recommendations,
        "created_at": datetime.utcnow()
    }
    
    assessment_collection.insert_one(assessment_document)
    
    return {
        "assessments": assessments,
        "scores": scores,
        "recommendations": recommendations,
        "assessment_id": assessment_document["assessment_id"]
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    quality_rubric_generator, comparative_analysis_tool,
    quality_trend_tracker, assessment_matrix
]:
    tool_registry.register_tool(tool_func, "quality")
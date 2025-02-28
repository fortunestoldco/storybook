from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any, Optional, Union, Literal, TypedDict
import logging

# Third-party imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Local imports
from storybook.agents.character_developer import CharacterDeveloper
from storybook.agents.dialogue_enhancer import DialogueEnhancer
from storybook.agents.world_builder import WorldBuilder
from storybook.agents.subplot_weaver import SubplotWeaver
from storybook.agents.story_arc_analyst import StoryArcAnalyst
from storybook.agents.continuity_editor import ContinuityEditor
from storybook.agents.language_polisher import LanguagePolisher
from storybook.agents.quality_reviewer import QualityReviewer
from storybook.agents.market_researcher import MarketResearcher
from storybook.agents.content_analyzer import ContentAnalyzer
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class NovelGraphState(TypedDict):
    manuscript_id: str
    title: str
    current_state: str
    characters: List[Dict[str, Any]]
    settings: List[Dict[str, Any]]
    subplots: List[Dict[str, Any]]
    story_analysis: Dict[str, Any]
    continuity_issues: List[Dict[str, Any]]
    style_analysis: Dict[str, Any]
    final_review: Dict[str, Any]
    research_insights: Dict[str, Any]
    analysis_results: Dict[str, Any]
    target_audience: Dict[str, Any]
    message: str
    stage_progress: Dict[str, float]


def start_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Start the workflow with a new manuscript."""
    logger.info("Starting workflow")

    # Extract input parameters
    manuscript_id = state.get("manuscript_id")
    title = state.get("title")

    if not manuscript_id:
        return {
            "current_state": END,
            "message": "Error: manuscript_id is required to start the transformation process.",
        }

    # Initialize the document store
    document_store = DocumentStore()

    # Check if manuscript exists
    manuscript = document_store.get_manuscript(manuscript_id)
    if not manuscript:
        return {
            "current_state": END,
            "message": f"Error: Manuscript with ID {manuscript_id} not found.",
        }

    # Get title if not provided
    if not title:
        title = manuscript.get("title", "Untitled")

    # Initialize state with manuscript ID
    new_state = NovelGraphState(
        manuscript_id=manuscript_id,
        title=title,
        current_state="research",
        characters=[],
        settings=[],
        subplots=[],
        story_analysis={},
        continuity_issues=[],
        style_analysis={},
        final_review={},
        research_insights={},
        analysis_results={},
        target_audience={},
        message=f"Starting transformation process for manuscript: {title}",
        stage_progress={},
    )

    return new_state


def conduct_market_research(state: Dict[str, Any]) -> Dict[str, Any]:
    """Conduct market research on publishing trends and target audience."""
    logger.info(f"Conducting market research for manuscript {state['manuscript_id']}")

    agent = MarketResearcher()
    result = agent.research_market(state["manuscript_id"], state["title"])

    # Update state with research insights
    new_state = state.copy()
    new_state["research_insights"] = result.get("research_insights", {})
    new_state["target_audience"] = result.get("target_audience", {})
    new_state["current_state"] = "analysis"
    new_state["message"] = (
        "Completed market research on publishing trends and target audience."
    )
    new_state["stage_progress"] = {**state.get("stage_progress", {}), "research": 1.0}

    # Store the research insights in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=str(result.get("research_insights", {})),
        metadata={
            "type": "market_research",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("research", [doc])

    return new_state


def analyze_manuscript(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze manuscript content using NLP techniques."""
    logger.info(f"Analyzing manuscript {state['manuscript_id']}")

    agent = ContentAnalyzer()
    result = agent.analyze_content(state["manuscript_id"])

    # Update state with analysis results
    new_state = state.copy()
    new_state["analysis_results"] = result.get("analysis", {})
    new_state["current_state"] = "initialize"
    new_state["message"] = "Completed NLP analysis of manuscript content."
    new_state["stage_progress"] = {**state.get("stage_progress", {}), "analysis": 1.0}

    # Store the analysis results in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=str(result.get("analysis", {})),
        metadata={
            "type": "content_analysis",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    return new_state


def initialize_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the graph with a new manuscript."""
    logger.info("Initializing graph")

    # At this point, we already have research and analysis
    # Update state to move to next phase
    new_state = state.copy()
    new_state["current_state"] = "character_development"
    new_state["message"] = (
        f"Research and analysis complete. Starting character development for {state['title']}."
    )
    new_state["stage_progress"] = {**state.get("stage_progress", {}), "initialize": 1.0}

    return new_state


def develop_characters(state: Dict[str, Any]) -> Dict[str, Any]:
    """Develop and enhance characters in the manuscript."""
    logger.info(f"Developing characters for manuscript {state['manuscript_id']}")

    agent = CharacterDeveloper()
    result = agent.enhance_characters(
        state["manuscript_id"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state with character information
    new_state = state.copy()
    new_state["characters"] = result.get("characters", [])
    new_state["current_state"] = "dialogue_enhancement"
    new_state["message"] = (
        f"Enhanced {len(new_state['characters'])} characters based on target audience research."
    )
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "character_development": 1.0,
    }

    # Store the character development results in MongoDB Atlas Vector
    document_store = DocumentStore()
    for character in result.get("characters", []):
        doc = Document(
            page_content=str(character),
            metadata={
                "type": "character_development",
                "character_name": character.get("name", "Unknown"),
                "manuscript_id": state["manuscript_id"],
                "title": state["title"],
            },
        )
        document_store.db.store_documents_with_embeddings("characters", [doc])

    return new_state


def enhance_dialogue(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance dialogue in the manuscript."""
    logger.info(f"Enhancing dialogue for manuscript {state['manuscript_id']}")

    agent = DialogueEnhancer()
    result = agent.enhance_dialogue(
        state["manuscript_id"],
        state["characters"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state
    new_state = state.copy()
    new_state["current_state"] = "world_building"
    new_state["message"] = result.get("message", "Dialogue enhancement completed.")
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "dialogue_enhancement": 1.0,
    }

    # Store the dialogue enhancement results in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=str(result),
        metadata={
            "type": "dialogue_enhancement",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    return new_state


def build_world(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build and enhance the world and settings."""
    logger.info(f"Building world for manuscript {state['manuscript_id']}")

    agent = WorldBuilder()
    result = agent.build_world(
        state["manuscript_id"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state
    new_state = state.copy()
    new_state["settings"] = result.get("settings", [])
    new_state["current_state"] = "subplot_integration"
    new_state["message"] = (
        f"Enhanced {len(new_state['settings'])} settings based on research."
    )
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "world_building": 1.0,
    }

    # Store the world building results in MongoDB Atlas Vector
    document_store = DocumentStore()
    for setting in result.get("settings", []):
        doc = Document(
            page_content=str(setting),
            metadata={
                "type": "world_building",
                "setting_name": setting.get("name", "Unknown"),
                "manuscript_id": state["manuscript_id"],
                "title": state["title"],
            },
        )
        document_store.db.store_documents_with_embeddings("worlds", [doc])

    return new_state


def integrate_subplots(state: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate subplots into the manuscript."""
    logger.info(f"Integrating subplots for manuscript {state['manuscript_id']}")

    agent = SubplotWeaver()
    result = agent.weave_subplots(
        state["manuscript_id"],
        state["characters"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state
    new_state = state.copy()
    new_state["subplots"] = result.get("developed_subplots", [])
    new_state["current_state"] = "story_arc_evaluation"
    new_state["message"] = result.get("message", "Subplot integration completed.")
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "subplot_integration": 1.0,
    }

    # Store the subplot integration results in MongoDB Atlas Vector
    document_store = DocumentStore()
    for subplot in result.get("developed_subplots", []):
        doc = Document(
            page_content=str(subplot),
            metadata={
                "type": "subplot_integration",
                "subplot_title": subplot.get("title", "Unknown"),
                "manuscript_id": state["manuscript_id"],
                "title": state["title"],
            },
        )
        document_store.db.store_documents_with_embeddings("subplots", [doc])

    return new_state


def evaluate_story_arcs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze and refine story arcs."""
    logger.info(f"Evaluating story arcs for manuscript {state['manuscript_id']}")

    agent = StoryArcAnalyst()
    result = agent.refine_story_arcs(
        state["manuscript_id"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state
    new_state = state.copy()
    new_state["story_analysis"] = result.get("analysis", {})
    new_state["current_state"] = "continuity_check"
    new_state["message"] = result.get(
        "message", "Story arc analysis and refinement completed."
    )
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "story_arc_evaluation": 1.0,
    }

    # Update the analysis results with progress tracking
    analysis_agent = ContentAnalyzer()
    progress_analysis = analysis_agent.analyze_progress(
        state["manuscript_id"], state.get("analysis_results", {}), "story_arc"
    )
    new_state["analysis_results"] = {
        **new_state.get("analysis_results", {}),
        **progress_analysis,
    }

    # Store the story arc analysis results in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=str(result.get("analysis", {})),
        metadata={
            "type": "story_arc_evaluation",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    # Also store the progress analysis
    doc = Document(
        page_content=str(progress_analysis),
        metadata={
            "type": "progress_analysis",
            "stage": "story_arc",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    return new_state


def check_continuity(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check and fix continuity issues."""
    logger.info(f"Checking continuity for manuscript {state['manuscript_id']}")

    agent = ContinuityEditor()
    result = agent.check_and_fix_continuity(state["manuscript_id"])

    # Update state
    new_state = state.copy()
    new_state["continuity_issues"] = result.get("issues", [])
    new_state["current_state"] = "language_polishing"
    new_state["message"] = result.get("message", "Continuity check completed.")
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "continuity_check": 1.0,
    }

    # Store the continuity check results in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=str(result),
        metadata={
            "type": "continuity_check",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
            "issues_fixed": len(result.get("issues", [])),
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    return new_state


def polish_language(state: Dict[str, Any]) -> Dict[str, Any]:
    """Polish language and style."""
    logger.info(f"Polishing language for manuscript {state['manuscript_id']}")

    agent = LanguagePolisher()
    result = agent.polish_language(
        state["manuscript_id"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state
    new_state = state.copy()
    new_state["style_analysis"] = result.get("style_analysis", {})
    new_state["current_state"] = "quality_review"
    new_state["message"] = result.get("message", "Language polishing completed.")
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "language_polishing": 1.0,
    }

    # Update the analysis results with progress tracking
    analysis_agent = ContentAnalyzer()
    progress_analysis = analysis_agent.analyze_progress(
        state["manuscript_id"], state.get("analysis_results", {}), "language"
    )
    new_state["analysis_results"] = {
        **new_state.get("analysis_results", {}),
        **progress_analysis,
    }

    # Store the language polishing results in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=str(result),
        metadata={
            "type": "language_polishing",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    # Also store the progress analysis
    doc = Document(
        page_content=str(progress_analysis),
        metadata={
            "type": "progress_analysis",
            "stage": "language",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    return new_state


def review_quality(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform final quality review."""
    logger.info(f"Reviewing quality for manuscript {state['manuscript_id']}")

    agent = QualityReviewer()
    result = agent.finalize_manuscript(
        state["manuscript_id"],
        target_audience=state.get("target_audience", {}),
        research_insights=state.get("research_insights", {}),
    )

    # Update state
    new_state = state.copy()
    new_state["final_review"] = {
        "review": result.get("review", ""),
        "improvements": result.get("improvements", []),
        "final_report": result.get("final_report", ""),
    }
    new_state["current_state"] = "finalize"
    new_state["message"] = result.get("message", "Quality review completed.")
    new_state["stage_progress"] = {
        **state.get("stage_progress", {}),
        "quality_review": 1.0,
    }

    # Final analysis to evaluate overall improvements
    analysis_agent = ContentAnalyzer()
    final_analysis = analysis_agent.analyze_progress(
        state["manuscript_id"], state.get("analysis_results", {}), "complete"
    )
    new_state["analysis_results"] = {
        **new_state.get("analysis_results", {}),
        **final_analysis,
    }

    # Store the quality review results in MongoDB Atlas Vector
    document_store = DocumentStore()
    doc = Document(
        page_content=result.get("final_report", ""),
        metadata={
            "type": "quality_review",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    # Also store the final analysis
    doc = Document(
        page_content=str(final_analysis),
        metadata={
            "type": "progress_analysis",
            "stage": "complete",
            "manuscript_id": state["manuscript_id"],
            "title": state["title"],
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    return new_state


def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize the transformation process."""
    logger.info(f"Finalizing transformation for manuscript {state['manuscript_id']}")

    # Create a summary of the transformation process
    manuscript_id = state["manuscript_id"]
    document_store = DocumentStore()
    manuscript = document_store.get_manuscript(manuscript_id)

    # Prepare final state
    new_state = state.copy()
    new_state["current_state"] = END
    new_state["message"] = (
        f"Transformation complete for manuscript: {state.get('title', 'Untitled')}. "
        f"Enhanced {len(state['characters'])} characters, {len(state['settings'])} settings, "
        f"and integrated {len(state['subplots'])} subplots. "
        f"Fixed {len(state['continuity_issues'])} continuity issues. "
        f"Final report available in the final_review field."
    )
    new_state["stage_progress"] = {**state.get("stage_progress", {}), "finalize": 1.0}

    # Create a final summary document and store it in MongoDB Atlas Vector
    final_report = state.get("final_review", {}).get("final_report", "")

    improvement_metrics = {
        "characters_enhanced": len(state.get("characters", [])),
        "settings_developed": len(state.get("settings", [])),
        "subplots_integrated": len(state.get("subplots", [])),
        "continuity_issues_fixed": len(state.get("continuity_issues", [])),
        "language_improvements": state.get("style_analysis", {}).get(
            "improvements_count", 0
        ),
    }

    # Store the comprehensive final report in MongoDB Atlas Vector
    doc = Document(
        page_content=final_report,
        metadata={
            "type": "final_report",
            "manuscript_id": manuscript_id,
            "title": state.get("title", "Untitled"),
            "improvement_metrics": str(improvement_metrics),
            "completion_status": "complete",
        },
    )
    document_store.db.store_documents_with_embeddings("analysis", [doc])

    # Create the output
    return {
        "manuscript_id": manuscript_id,
        "status": "complete",
        "final_report": final_report,
        "research_insights": state.get("research_insights", {}),
        "analysis_results": state.get("analysis_results", {}),
        "improvement_metrics": improvement_metrics,
    }


def should_retry_character_development(state: Dict[str, Any]) -> str:
    """Determine if character development needs to be retried."""
    if not state.get("characters"):
        return "character_development"
    return "dialogue_enhancement"


def route_after_quality_review(state: Dict[str, Any]) -> str:
    """Route after quality review depending on final review content."""
    # If there's a comprehensive final review, proceed to finalize
    if state.get("final_review") and "review" in state["final_review"]:
        return "finalize"
    # Otherwise retry quality review
    return "quality_review"


def check_analysis_progress(
    state: Dict[str, Any],
) -> Union[Literal["initialize"], Literal["analysis"]]:
    """Check if the analysis is complete and comprehensive enough."""
    analysis_results = state.get("analysis_results", {})

    # Check if we have comprehensive analysis data
    required_analysis_types = [
        "sentiment",
        "readability",
        "content_structure",
        "genre_match",
    ]

    for analysis_type in required_analysis_types:
        if (analysis_type not in analysis_results) or (not analysis_results[analysis_type]):
            return "analysis"  # Retry analysis

    # If all required data is present, move to initialization
    return "initialize"


def build_storybook() -> StateGraph:
    workflow = StateGraph(NovelGraphState)

    # Add nodes
    workflow.add_node("START", start_workflow)
    workflow.add_node("research", conduct_market_research)
    workflow.add_node("analysis", analyze_manuscript)
    workflow.add_node("initialize", initialize_graph)
    workflow.add_node("character_development", develop_characters)
    workflow.add_node("dialogue_enhancement", enhance_dialogue)
    workflow.add_node("world_building", build_world)
    workflow.add_node("subplot_integration", integrate_subplots)
    workflow.add_node("story_arc_evaluation", evaluate_story_arcs)
    workflow.add_node("continuity_check", check_continuity)
    workflow.add_node("language_polishing", polish_language)
    workflow.add_node("quality_review", review_quality)
    workflow.add_node("finalize", finalize)
    workflow.add_node("END")

    # Add edges
    workflow.set_entry_point("START")
    workflow.add_edge("START", "research")
    workflow.add_edge("research", "analysis")
    workflow.add_conditional_edges(
        "analysis",
        check_analysis_progress,
        {"analysis": "analysis", "initialize": "initialize"},
    )
    workflow.add_edge("initialize", "character_development")
    workflow.add_conditional_edges(
        "character_development",
        should_retry_character_development,
        {
            "character_development": "character_development",
            "dialogue_enhancement": "dialogue_enhancement",
        },
    )
    workflow.add_edge("dialogue_enhancement", "world_building")
    workflow.add_edge("world_building", "subplot_integration")
    workflow.add_edge("subplot_integration", "story_arc_evaluation")
    workflow.add_edge("story_arc_evaluation", "continuity_check")
    workflow.add_edge("continuity_check", "language_polishing")
    workflow.add_edge("language_polishing", "quality_review")
    workflow.add_conditional_edges(
        "quality_review",
        route_after_quality_review,
        {"quality_review": "quality_review", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", "END")

    return workflow


# Create the novel transformation graph
storybook = build_storybook().compile()

from typing import Dict, List, Callable, Optional, Any, Annotated, TypedDict, cast
from langchain_core.runnables.config import RunnableConfig
import json
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver

from config import MONGODB_CONFIG, QUALITY_GATES
from state import NovelSystemState
from mongodb import MongoDBManager
from agents import AgentFactory
from utils import check_quality_gate


# Define routing functions outside of the graph creation functions
def route_after_executive_director_initialization(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "human_feedback" in task.lower():
        return "human_feedback_manager"
    elif "quality" in task.lower() or "assessment" in task.lower():
        return "quality_assessment_director"
    elif "timeline" in task.lower() or "schedule" in task.lower():
        return "project_timeline_manager"
    elif "market" in task.lower() or "trend" in task.lower():
        return "market_alignment_director"
    else:
        metrics = state["project"].quality_assessment
        gate_result = check_quality_gate("initialization_to_development", metrics)
        return END if gate_result["passed"] else "executive_director"


def create_initialization_graph(config: RunnableConfig) -> StateGraph:
    """Create the workflow graph for the initialization phase."""
    metadata = config.get("metadata", {})
    project_id = metadata.get("project_id")
    agent_factory = metadata.get("agent_factory")

    if not project_id or not agent_factory:
        raise ValueError("Missing required metadata: project_id and agent_factory are required")

    # Create agent nodes
    executive_director = agent_factory.create_agent("executive_director", project_id)
    human_feedback_manager = agent_factory.create_agent("human_feedback_manager", project_id)
    quality_assessment_director = agent_factory.create_agent("quality_assessment_director", project_id)
    project_timeline_manager = agent_factory.create_agent("project_timeline_manager", project_id)
    market_alignment_director = agent_factory.create_agent("market_alignment_director", project_id)

    # Define the state graph
    workflow = StateGraph(NovelSystemState)

    # Add nodes
    workflow.add_node("executive_director", executive_director)
    workflow.add_node("human_feedback_manager", human_feedback_manager)
    workflow.add_node("quality_assessment_director", quality_assessment_director)
    workflow.add_node("project_timeline_manager", project_timeline_manager)
    workflow.add_node("market_alignment_director", market_alignment_director)

    # Set up the edges - IMPORTANT: Use route_fn=... parameter instead of directly passing the function
    workflow.add_conditional_edges(
        "executive_director",
        route_after_executive_director_initialization
    )
    workflow.add_edge("human_feedback_manager", "executive_director")
    workflow.add_edge("quality_assessment_director", "executive_director")
    workflow.add_edge("project_timeline_manager", "executive_director")
    workflow.add_edge("market_alignment_director", "executive_director")

    # Set the entry point
    workflow.add_edge("__start__", "executive_director")

    # Get MongoDB config from environment or defaults
    mongodb_uri = os.getenv("MONGODB_URI", MONGODB_CONFIG["connection_string"])
    db_name = os.getenv("MONGODB_DB", MONGODB_CONFIG["database_name"])

    # Set up checkpointing with MongoDB
    checkpointer = MongoDBSaver.from_conn_string(
        mongodb_uri,
        database_name=db_name,
        collection_name=f"checkpoint_initialization_{project_id}"
    )

    # Compile the graph with persistent checkpoint saver
    graph = workflow.compile(checkpointer=checkpointer)
    graph.name = config.get("configurable", {}).get("graph_name", "Initialization Graph")

    return graph


# Define routing functions for other phases as global functions
def route_after_executive_director_development(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "creative" in task.lower() or "story" in task.lower():
        return "creative_director"
    elif "market" in task.lower() or "trend" in task.lower():
        return "market_alignment_director"
    elif "research" in task.lower() or "knowledge" in task.lower():
        return "domain_knowledge_specialist"
    else:
        metrics = state["project"].quality_assessment
        gate_result = check_quality_gate("development_to_creation", metrics)
        return END if gate_result["passed"] else "creative_director"


def route_after_creative_director_development(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "structure" in task.lower() or "plot" in task.lower():
        return "structure_architect"
    elif "character" in task.lower() and "psychology" in task.lower():
        return "character_psychology_specialist"
    elif "character" in task.lower() and "voice" in task.lower():
        return "character_voice_designer"
    elif "character" in task.lower() and "relationship" in task.lower():
        return "character_relationship_mapper"
    elif "world" in task.lower() or "setting" in task.lower():
        return "world_building_expert"
    else:
        return "executive_director"


def create_development_graph(config: RunnableConfig) -> StateGraph:
    """Create the workflow graph for the development phase."""
    metadata = config.get("metadata", {})
    project_id = metadata.get("project_id")
    agent_factory = metadata.get("agent_factory")

    if not project_id or not agent_factory:
        raise ValueError("Missing required metadata: project_id and agent_factory are required")

    # Create agent nodes for the development phase
    executive_director = agent_factory.create_agent("executive_director", project_id)
    creative_director = agent_factory.create_agent("creative_director", project_id)
    structure_architect = agent_factory.create_agent("structure_architect", project_id)
    plot_development_specialist = agent_factory.create_agent("plot_development_specialist", project_id)
    world_building_expert = agent_factory.create_agent("world_building_expert", project_id)
    character_psychology_specialist = agent_factory.create_agent("character_psychology_specialist", project_id)
    character_voice_designer = agent_factory.create_agent("character_voice_designer", project_id)
    character_relationship_mapper = agent_factory.create_agent("character_relationship_mapper", project_id)
    domain_knowledge_specialist = agent_factory.create_agent("domain_knowledge_specialist", project_id)
    cultural_authenticity_expert = agent_factory.create_agent("cultural_authenticity_expert", project_id)
    market_alignment_director = agent_factory.create_agent("market_alignment_director", project_id)

    # Define the state graph
    workflow = StateGraph(NovelSystemState)

    # Add nodes
    workflow.add_node("executive_director", executive_director)
    workflow.add_node("creative_director", creative_director)
    workflow.add_node("structure_architect", structure_architect)
    workflow.add_node("plot_development_specialist", plot_development_specialist)
    workflow.add_node("world_building_expert", world_building_expert)
    workflow.add_node("character_psychology_specialist", character_psychology_specialist)
    workflow.add_node("character_voice_designer", character_voice_designer)
    workflow.add_node("character_relationship_mapper", character_relationship_mapper)
    workflow.add_node("domain_knowledge_specialist", domain_knowledge_specialist)
    workflow.add_node("cultural_authenticity_expert", cultural_authenticity_expert)
    workflow.add_node("market_alignment_director", market_alignment_director)

    # Set up the edges - use the globally defined routing functions with add_conditional_edges
    workflow.add_conditional_edges(
        "executive_director",
        route_after_executive_director_development
    )
    workflow.add_conditional_edges(
        "creative_director",
        route_after_creative_director_development
    )
    workflow.add_edge("structure_architect", "creative_director")
    workflow.add_edge("plot_development_specialist", "creative_director")
    workflow.add_edge("world_building_expert", "creative_director")
    workflow.add_edge("character_psychology_specialist", "creative_director")
    workflow.add_edge("character_voice_designer", "creative_director")
    workflow.add_edge("character_relationship_mapper", "creative_director")
    workflow.add_edge("domain_knowledge_specialist", "executive_director")
    workflow.add_edge("cultural_authenticity_expert", "executive_director")
    workflow.add_edge("market_alignment_director", "executive_director")

    # Set the entry point
    workflow.add_edge("__start__", "executive_director")

    # Get MongoDB config from environment or defaults
    mongodb_uri = os.getenv("MONGODB_URI", MONGODB_CONFIG["connection_string"])
    db_name = os.getenv("MONGODB_DB", MONGODB_CONFIG["database_name"])

    # Set up checkpointing with MongoDB
    checkpointer = MongoDBSaver.from_conn_string(
        mongodb_uri,
        database_name=db_name,
        collection_name=f"checkpoint_development_{project_id}"
    )

    # Compile the graph with persistent checkpoint saver
    graph = workflow.compile(checkpointer=checkpointer)
    graph.name = config.get("configurable", {}).get("graph_name", "Development Graph")

    return graph


# Define routing functions for creation phase
def route_after_executive_director_creation(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "content" in task.lower() or "draft" in task.lower():
        return "content_development_director"
    elif "creative" in task.lower() or "emotion" in task.lower():
        return "creative_director"
    else:
        metrics = state["project"].quality_assessment
        gate_result = check_quality_gate("creation_to_refinement", metrics)
        return END if gate_result["passed"] else "content_development_director"


def route_after_content_director_creation(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "chapter" in task.lower():
        return "chapter_drafters"
    elif "scene" in task.lower():
        return "scene_construction_specialists"
    elif "dialogue" in task.lower():
        return "dialogue_crafters"
    elif "continuity" in task.lower():
        return "continuity_manager"
    elif "voice" in task.lower() or "consistency" in task.lower():
        return "voice_consistency_monitor"
    elif "research" in task.lower() or "knowledge" in task.lower():
        return "domain_knowledge_specialist"
    else:
        return "executive_director"


def route_after_creative_director_creation(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "emotion" in task.lower() or "arc" in task.lower():
        return "emotional_arc_designer"
    else:
        return "content_development_director"


def create_creation_graph(config: RunnableConfig) -> StateGraph:
    """Create the workflow graph for the creation phase."""
    metadata = config.get("metadata", {})
    project_id = metadata.get("project_id")
    agent_factory = metadata.get("agent_factory")

    if not project_id or not agent_factory:
        raise ValueError("Missing required metadata: project_id and agent_factory are required")

    # Create agent nodes for the creation phase
    executive_director = agent_factory.create_agent("executive_director", project_id)
    content_development_director = agent_factory.create_agent("content_development_director", project_id)
    creative_director = agent_factory.create_agent("creative_director", project_id)
    chapter_drafters = agent_factory.create_agent("chapter_drafters", project_id)
    scene_construction_specialists = agent_factory.create_agent("scene_construction_specialists", project_id)
    dialogue_crafters = agent_factory.create_agent("dialogue_crafters", project_id)
    continuity_manager = agent_factory.create_agent("continuity_manager", project_id)
    voice_consistency_monitor = agent_factory.create_agent("voice_consistency_monitor", project_id)
    emotional_arc_designer = agent_factory.create_agent("emotional_arc_designer", project_id)
    domain_knowledge_specialist = agent_factory.create_agent("domain_knowledge_specialist", project_id)

    # Define the state graph
    workflow = StateGraph(NovelSystemState)

    # Add nodes
    workflow.add_node("executive_director", executive_director)
    workflow.add_node("content_development_director", content_development_director)
    workflow.add_node("creative_director", creative_director)
    workflow.add_node("chapter_drafters", chapter_drafters)
    workflow.add_node("scene_construction_specialists", scene_construction_specialists)
    workflow.add_node("dialogue_crafters", dialogue_crafters)
    workflow.add_node("continuity_manager", continuity_manager)
    workflow.add_node("voice_consistency_monitor", voice_consistency_monitor)
    workflow.add_node("emotional_arc_designer", emotional_arc_designer)
    workflow.add_node("domain_knowledge_specialist", domain_knowledge_specialist)

    # Set up the edges - use the globally defined routing functions with add_conditional_edges
    workflow.add_conditional_edges(
        "executive_director",
        route_after_executive_director_creation
    )
    workflow.add_conditional_edges(
        "content_development_director",
        route_after_content_director_creation
    )
    workflow.add_conditional_edges(
        "creative_director",
        route_after_creative_director_creation
    )
    workflow.add_edge("chapter_drafters", "content_development_director")
    workflow.add_edge("scene_construction_specialists", "content_development_director")
    workflow.add_edge("dialogue_crafters", "content_development_director")
    workflow.add_edge("continuity_manager", "content_development_director")
    workflow.add_edge("voice_consistency_monitor", "content_development_director")
    workflow.add_edge("emotional_arc_designer", "creative_director")
    workflow.add_edge("domain_knowledge_specialist", "content_development_director")

    # Set the entry point
    workflow.add_edge("__start__", "executive_director")

    # Get MongoDB config from environment or defaults
    mongodb_uri = os.getenv("MONGODB_URI", MONGODB_CONFIG["connection_string"])
    db_name = os.getenv("MONGODB_DB", MONGODB_CONFIG["database_name"])

    # Set up checkpointing with MongoDB
    checkpointer = MongoDBSaver.from_conn_string(
        mongodb_uri,
        database_name=db_name,
        collection_name=f"checkpoint_creation_{project_id}"
    )

    # Compile the graph with persistent checkpoint saver
    graph = workflow.compile(checkpointer=checkpointer)
    graph.name = config.get("configurable", {}).get("graph_name", "Creation Graph")

    return graph


# Define routing functions for refinement phase
def route_after_executive_director_refinement(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "edit" in task.lower() or "revise" in task.lower():
        return "editorial_director"
    elif "creative" in task.lower():
        return "creative_director"
    elif "market" in task.lower():
        return "market_alignment_director"
    else:
        metrics = state["project"].quality_assessment
        gate_result = check_quality_gate("refinement_to_finalization", metrics)
        return END if gate_result["passed"] else "editorial_director"


def route_after_editorial_director_refinement(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")
    editing_type = state["current_input"].get("editing_type", "").lower()

    if editing_type == "developmental" or "structure" in task.lower():
        return "structural_editor"
    elif editing_type == "developmental" or "character" in task.lower():
        return "character_arc_evaluator"
    elif editing_type == "developmental" or "theme" in task.lower():
        return "thematic_coherence_analyst"
    elif editing_type == "line" or "prose" in task.lower():
        return "prose_enhancement_specialist"
    elif editing_type == "line" or "dialogue" in task.lower():
        return "dialogue_refinement_expert"
    elif editing_type == "line" or "rhythm" in task.lower() or "flow" in task.lower():
        return "rhythm_cadence_optimizer"
    elif editing_type == "technical" or "grammar" in task.lower():
        return "grammar_consistency_checker"
    elif editing_type == "technical" or "fact" in task.lower():
        return "fact_verification_specialist"
    else:
        return "executive_director"


def create_refinement_graph(config: RunnableConfig) -> StateGraph:
    """Create the workflow graph for the refinement phase."""
    metadata = config.get("metadata", {})
    project_id = metadata.get("project_id")
    agent_factory = metadata.get("agent_factory")

    if not project_id or not agent_factory:
        raise ValueError("Missing required metadata: project_id and agent_factory are required")

    # Create agent nodes for the refinement phase
    executive_director = agent_factory.create_agent("executive_director", project_id)
    editorial_director = agent_factory.create_agent("editorial_director", project_id)
    creative_director = agent_factory.create_agent("creative_director", project_id)
    market_alignment_director = agent_factory.create_agent("market_alignment_director", project_id)
    structural_editor = agent_factory.create_agent("structural_editor", project_id)
    character_arc_evaluator = agent_factory.create_agent("character_arc_evaluator", project_id)
    thematic_coherence_analyst = agent_factory.create_agent("thematic_coherence_analyst", project_id)
    prose_enhancement_specialist = agent_factory.create_agent("prose_enhancement_specialist", project_id)
    dialogue_refinement_expert = agent_factory.create_agent("dialogue_refinement_expert", project_id)
    rhythm_cadence_optimizer = agent_factory.create_agent("rhythm_cadence_optimizer", project_id)
    grammar_consistency_checker = agent_factory.create_agent("grammar_consistency_checker", project_id)
    fact_verification_specialist = agent_factory.create_agent("fact_verification_specialist", project_id)

    # Define the state graph
    workflow = StateGraph(NovelSystemState)

    # Add nodes
    workflow.add_node("executive_director", executive_director)
    workflow.add_node("editorial_director", editorial_director)
    workflow.add_node("creative_director", creative_director)
    workflow.add_node("market_alignment_director", market_alignment_director)
    workflow.add_node("structural_editor", structural_editor)
    workflow.add_node("character_arc_evaluator", character_arc_evaluator)
    workflow.add_node("thematic_coherence_analyst", thematic_coherence_analyst)
    workflow.add_node("prose_enhancement_specialist", prose_enhancement_specialist)
    workflow.add_node("dialogue_refinement_expert", dialogue_refinement_expert)
    workflow.add_node("rhythm_cadence_optimizer", rhythm_cadence_optimizer)
    workflow.add_node("grammar_consistency_checker", grammar_consistency_checker)
    workflow.add_node("fact_verification_specialist", fact_verification_specialist)

    # Set up the edges - use the globally defined routing functions with add_conditional_edges
    workflow.add_conditional_edges(
        "executive_director",
        route_after_executive_director_refinement
    )
    workflow.add_conditional_edges(
        "editorial_director",
        route_after_editorial_director_refinement
    )
    workflow.add_edge("creative_director", "executive_director")
    workflow.add_edge("market_alignment_director", "executive_director")
    workflow.add_edge("structural_editor", "editorial_director")
    workflow.add_edge("character_arc_evaluator", "editorial_director")
    workflow.add_edge("thematic_coherence_analyst", "editorial_director")
    workflow.add_edge("prose_enhancement_specialist", "editorial_director")
    workflow.add_edge("dialogue_refinement_expert", "editorial_director")
    workflow.add_edge("rhythm_cadence_optimizer", "editorial_director")
    workflow.add_edge("grammar_consistency_checker", "editorial_director")
    workflow.add_edge("fact_verification_specialist", "editorial_director")

    # Set the entry point
    workflow.add_edge("__start__", "executive_director")

    # Get MongoDB config from environment or defaults
    mongodb_uri = os.getenv("MONGODB_URI", MONGODB_CONFIG["connection_string"])
    db_name = os.getenv("MONGODB_DB", MONGODB_CONFIG["database_name"])

    # Set up checkpointing with MongoDB
    checkpointer = MongoDBSaver.from_conn_string(
        mongodb_uri,
        database_name=db_name,
        collection_name=f"checkpoint_refinement_{project_id}"
    )

    # Compile the graph with persistent checkpoint saver
    graph = workflow.compile(checkpointer=checkpointer)
    graph.name = config.get("configurable", {}).get("graph_name", "Refinement Graph")

    return graph


# Define routing functions for finalization phase
def route_after_executive_director_finalization(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "market" in task.lower() or "position" in task.lower():
        return "market_alignment_director"
    elif "edit" in task.lower() or "format" in task.lower():
        return "editorial_director"
    else:
        metrics = state["project"].quality_assessment
        gate_result = check_quality_gate("finalization_to_complete", metrics)
        return END if gate_result["passed"] else "market_alignment_director"


def route_after_market_director_finalization(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "position" in task.lower() or "strategy" in task.lower():
        return "positioning_specialist"
    elif "title" in task.lower() or "blurb" in task.lower():
        return "title_blurb_optimizer"
    elif "different" in task.lower() or "unique" in task.lower():
        return "differentiation_strategist"
    else:
        return "executive_director"


def route_after_editorial_director_finalization(state: NovelSystemState) -> str:
    task = state["current_input"].get("task", "")

    if "format" in task.lower() or "standard" in task.lower():
        return "formatting_standards_expert"
    else:
        return "executive_director"


def create_finalization_graph(config: RunnableConfig) -> StateGraph:
    """Create the workflow graph for the finalization phase."""
    metadata = config.get("metadata", {})
    project_id = metadata.get("project_id")
    agent_factory = metadata.get("agent_factory")

    if not project_id or not agent_factory:
        raise ValueError("Missing required metadata: project_id and agent_factory are required")

    # Create agent nodes for the finalization phase
    executive_director = agent_factory.create_agent("executive_director", project_id)
    editorial_director = agent_factory.create_agent("editorial_director", project_id)
    market_alignment_director = agent_factory.create_agent("market_alignment_director", project_id)
    positioning_specialist = agent_factory.create_agent("positioning_specialist", project_id)
    title_blurb_optimizer = agent_factory.create_agent("title_blurb_optimizer", project_id)
    differentiation_strategist = agent_factory.create_agent("differentiation_strategist", project_id)
    formatting_standards_expert = agent_factory.create_agent("formatting_standards_expert", project_id)

    # Define the state graph
    workflow = StateGraph(NovelSystemState)

    # Add nodes
    workflow.add_node("executive_director", executive_director)
    workflow.add_node("editorial_director", editorial_director)
    workflow.add_node("market_alignment_director", market_alignment_director)
    workflow.add_node("positioning_specialist", positioning_specialist)
    workflow.add_node("title_blurb_optimizer", title_blurb_optimizer)
    workflow.add_node("differentiation_strategist", differentiation_strategist)
    workflow.add_node("formatting_standards_expert", formatting_standards_expert)

    # Set up the edges - use the globally defined routing functions with add_conditional_edges
    workflow.add_conditional_edges(
        "executive_director",
        route_after_executive_director_finalization
    )
    workflow.add_conditional_edges(
        "market_alignment_director",
        route_after_market_director_finalization
    )
    workflow.add_conditional_edges(
        "editorial_director",
        route_after_editorial_director_finalization
    )
    workflow.add_edge("positioning_specialist", "market_alignment_director")
    workflow.add_edge("title_blurb_optimizer", "market_alignment_director")
    workflow.add_edge("differentiation_strategist", "market_alignment_director")
    workflow.add_edge("formatting_standards_expert", "editorial_director")

    # Set the entry point
    workflow.add_edge("__start__", "executive_director")

    # Get MongoDB config from environment or defaults
    mongodb_uri = os.getenv("MONGODB_URI", MONGODB_CONFIG["connection_string"])
    db_name = os.getenv("MONGODB_DB", MONGODB_CONFIG["database_name"])

    # Set up checkpointing with MongoDB
    checkpointer = MongoDBSaver.from_conn_string(
        mongodb_uri,
        database_name=db_name,
        collection_name=f"checkpoint_finalization_{project_id}"
    )

    # Compile the graph with persistent checkpoint saver
    graph = workflow.compile(checkpointer=checkpointer)
    graph.name = config.get("configurable", {}).get("graph_name", "Finalization Graph")

    return graph


def get_phase_workflow(config: RunnableConfig) -> StateGraph:
    """Get the workflow graph for a specific phase."""
    metadata = config.get("metadata", {})
    phase = metadata.get("phase")
    project_id = metadata.get("project_id")

    # Create a default project_id for schema retrieval if not provided
    if not project_id:
        project_id = "default_project"
        metadata["project_id"] = project_id

    # Create agent factory if not provided
    if "agent_factory" not in metadata:
        from agents import AgentFactory  # Import here to avoid circular imports
        metadata["agent_factory"] = AgentFactory()

    # Update the config with our modified metadata
    config_with_metadata = dict(config)
    config_with_metadata["metadata"] = metadata

    # If phase is not specified, default to "initialization" to avoid errors
    # This allows the API to get schemas without requiring phase parameter
    if not phase:
        phase = "initialization"  # Default phase when not specified

    # Create unique graph name for this project and phase
    graph_name = f"storybook_{project_id}_{phase}"

    # Set graph name in config before creating workflow
    config_with_name = dict(config_with_metadata)
    if "configurable" not in config_with_name:
        config_with_name["configurable"] = {}
    config_with_name["configurable"]["graph_name"] = graph_name

    # Map phases to their workflow creation functions
    workflow_map = {
        "initialization": create_initialization_graph,
        "development": create_development_graph,
        "creation": create_creation_graph,
        "refinement": create_refinement_graph,
        "finalization": create_finalization_graph,
    }

    if phase not in workflow_map:
        raise ValueError(f"Unknown phase: {phase}")

    # Create and return the workflow graph
    return workflow_map[phase](config_with_name)
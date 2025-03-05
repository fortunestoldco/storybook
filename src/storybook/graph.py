from typing import Dict, Any, List, Literal, cast
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBCheckpointHandler

from storybook.configuration import (
    StoryBookConfig, 
    Configuration, 
    ProjectType,
    NewProjectInput,
    ExistingProjectInput
)
from storybook.state import NovelSystemState, InputState
from storybook.agents import AgentFactory
from storybook.utils import check_quality_gate


def create_phase_graph(phase: str, project_id: str, config: Configuration) -> StateGraph:
    """Create a workflow graph for a specific phase.

    Args:
        phase: The phase name.
        project_id: ID of the project.
        config: System configuration.

    Returns:
        A StateGraph for the specified phase.
    """
    # Create agent factory
    agent_factory = AgentFactory(config)

    # Define phase-specific agents
    agents = {
        "initialization": [
            "executive_director",
            "human_feedback_manager",
            "quality_assessment_director",
            "project_timeline_manager",
            "market_alignment_director"
        ],
        "development": [
            "executive_director",
            "creative_director",
            "structure_architect",
            "plot_development_specialist",
            "world_building_expert",
            "character_psychology_specialist",
            "character_voice_designer",
            "character_relationship_mapper",
            "domain_knowledge_specialist",
            "cultural_authenticity_expert",
            "market_alignment_director"
        ],
        "creation": [
            "executive_director",
            "content_development_director",
            "creative_director",
            "chapter_drafters",
            "scene_construction_specialists",
            "dialogue_crafters",
            "continuity_manager",
            "voice_consistency_monitor",
            "emotional_arc_designer",
            "domain_knowledge_specialist"
        ],
        "refinement": [
            "executive_director",
            "editorial_director",
            "creative_director",
            "market_alignment_director",
            "structural_editor",
            "character_arc_evaluator",
            "thematic_coherence_analyst",
            "prose_enhancement_specialist",
            "dialogue_refinement_expert",
            "rhythm_cadence_optimizer",
            "grammar_consistency_checker",
            "fact_verification_specialist"
        ],
        "finalization": [
            "executive_director",
            "editorial_director",
            "market_alignment_director",
            "positioning_specialist",
            "title_blurb_optimizer",
            "differentiation_strategist",
            "formatting_standards_expert"
        ]
    }

    if phase not in agents:
        raise ValueError(f"Unknown phase: {phase}")

    # Create the graph builder
    builder = StateGraph(NovelSystemState, input=InputState, config_schema=Configuration)

    # Create and add all agents for this phase
    for agent_name in agents[phase]:
        agent_function = agent_factory.create_agent(agent_name, project_id)
        builder.add_node(agent_name, agent_function)

    # Set executive_director as the entry point
    builder.add_edge("__start__", "executive_director")

    # Define the routing functions based on the phase
    if phase == "initialization":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in initialization phase."""
            task = state.current_input.get("task", "").lower()

            if "human_feedback" in task:
                return "human_feedback_manager"
            elif "quality" in task or "assessment" in task:
                return "quality_assessment_director"
            elif "timeline" in task or "schedule" in task:
                return "project_timeline_manager"
            elif "market" in task or "trend" in task:
                return "market_alignment_director"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "initialization_to_development",
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )

                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )

        # All specialized agents return to executive director
        for agent in ["human_feedback_manager", "quality_assessment_director",
                     "project_timeline_manager", "market_alignment_director"]:
            builder.add_edge(agent, "executive_director")

    elif phase == "development":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in development phase."""
            task = state.current_input.get("task", "").lower()

            if "creative" in task or "story" in task:
                return "creative_director"
            elif "market" in task or "trend" in task:
                return "market_alignment_director"
            elif "research" in task or "knowledge" in task:
                return "domain_knowledge_specialist"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "development_to_creation",
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )

                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "creative_director"

        def route_after_creative_director(state: NovelSystemState) -> str:
            """Route after the creative director node."""
            task = state.current_input.get("task", "").lower()

            if "structure" in task or "plot" in task:
                return "structure_architect"
            elif "character" in task and "psychology" in task:
                return "character_psychology_specialist"
            elif "character" in task and "voice" in task:
                return "character_voice_designer"
            elif "character" in task and "relationship" in task:
                return "character_relationship_mapper"
            elif "world" in task or "setting" in task:
                return "world_building_expert"
            else:
                return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )

        builder.add_conditional_edges(
            "creative_director",
            route_after_creative_director
        )

        # Connect specialized agents to their supervisors
        for agent in ["structure_architect", "plot_development_specialist",
                     "world_building_expert", "character_psychology_specialist",
                     "character_voice_designer", "character_relationship_mapper"]:
            builder.add_edge(agent, "creative_director")

        for agent in ["domain_knowledge_specialist", "cultural_authenticity_expert",
                     "market_alignment_director"]:
            builder.add_edge(agent, "executive_director")

    elif phase == "creation":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in creation phase."""
            task = state.current_input.get("task", "").lower()

            if "content" in task or "development" in task:
                return "content_development_director"
            elif "creative" in task:
                return "creative_director"
            elif "domain" in task or "knowledge" in task:
                return "domain_knowledge_specialist"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "creation_to_refinement",
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )

                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "content_development_director"

        def route_after_content_director(state: NovelSystemState) -> str:
            """Route after the content development director node."""
            task = state.current_input.get("task", "").lower()

            if "chapter" in task:
                return "chapter_drafters"
            elif "scene" in task:
                return "scene_construction_specialists"
            elif "dialogue" in task:
                return "dialogue_crafters"
            elif "continuity" in task:
                return "continuity_manager"
            elif "voice" in task or "tone" in task:
                return "voice_consistency_monitor"
            elif "emotion" in task or "arc" in task:
                return "emotional_arc_designer"
            else:
                return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )

        builder.add_conditional_edges(
            "content_development_director",
            route_after_content_director
        )

        # Connect specialized agents to their supervisors
        for agent in ["chapter_drafters", "scene_construction_specialists",
                     "dialogue_crafters", "continuity_manager",
                     "voice_consistency_monitor", "emotional_arc_designer"]:
            builder.add_edge(agent, "content_development_director")

        builder.add_edge("content_development_director", "executive_director")
        builder.add_edge("creative_director", "executive_director")
        builder.add_edge("domain_knowledge_specialist", "executive_director")

    elif phase == "refinement":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in refinement phase."""
            task = state.current_input.get("task", "").lower()

            if "editorial" in task or "edit" in task:
                return "editorial_director"
            elif "creative" in task:
                return "creative_director"
            elif "market" in task:
                return "market_alignment_director"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "refinement_to_finalization",
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )

                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "editorial_director"

        def route_after_editorial_director(state: NovelSystemState) -> str:
            """Route after the editorial director node."""
            task = state.current_input.get("task", "").lower()

            if "structure" in task:
                return "structural_editor"
            elif "character" in task and "arc" in task:
                return "character_arc_evaluator"
            elif "theme" in task:
                return "thematic_coherence_analyst"
            elif "prose" in task:
                return "prose_enhancement_specialist"
            elif "dialogue" in task:
                return "dialogue_refinement_expert"
            elif "rhythm" in task or "cadence" in task:
                return "rhythm_cadence_optimizer"
            elif "grammar" in task:
                return "grammar_consistency_checker"
            elif "fact" in task or "verify" in task:
                return "fact_verification_specialist"
            else:
                return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )

        builder.add_conditional_edges(
            "editorial_director",
            route_after_editorial_director
        )

        # Connect specialized agents to their supervisors
        for agent in ["structural_editor", "character_arc_evaluator",
                     "thematic_coherence_analyst", "prose_enhancement_specialist",
                     "dialogue_refinement_expert", "rhythm_cadence_optimizer",
                     "grammar_consistency_checker", "fact_verification_specialist"]:
            builder.add_edge(agent, "editorial_director")

        builder.add_edge("editorial_director", "executive_director")
        builder.add_edge("creative_director", "executive_director")
        builder.add_edge("market_alignment_director", "executive_director")

    elif phase == "finalization":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in finalization phase."""
            task = state.current_input.get("task", "").lower()

            if "editorial" in task:
                return "editorial_director"
            elif "market" in task:
                return "market_alignment_director"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "finalization_to_complete",
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )

                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "editorial_director"

        def route_after_market_director(state: NovelSystemState) -> str:
            """Route after the market alignment director node."""
            task = state.current_input.get("task", "").lower()

            if "position" in task:
                return "positioning_specialist"
            elif "title" in task or "blurb" in task:
                return "title_blurb_optimizer"
            elif "different" in task or "unique" in task:
                return "differentiation_strategist"
            else:
                return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )

        builder.add_conditional_edges(
            "market_alignment_director",
            route_after_market_director
        )

        # Connect specialized agents to their supervisors
        builder.add_edge("editorial_director", "executive_director")

        for agent in ["positioning_specialist", "title_blurb_optimizer",
                     "differentiation_strategist"]:
            builder.add_edge(agent, "market_alignment_director")

        builder.add_edge("market_alignment_director", "executive_director")
        builder.add_edge("formatting_standards_expert", "editorial_director")

    # Set up MongoDB checkpointing if configured
    if config.mongodb_connection_string and config.mongodb_database_name:
        checkpointer = MongoDBCheckpointHandler(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database_name,
            collection_name=f"checkpoint_{phase}_{project_id}"
        )
        graph = builder.compile(checkpointer=checkpointer)
    else:
        # Fallback to memory checkpointing
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

    return graph


def create_supervisor_graph(config: Configuration) -> StateGraph:
    """Create the supervisor graph that manages phase transitions."""
    builder = StateGraph(NovelSystemState, input=InputState, config_schema=Configuration)

    # Phase transition node
    async def phase_manager(state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Manage phase transitions."""
        configuration = Configuration.from_runnable_config(config)
        current_phase = state.phase

        # Check if we should transition to the next phase
        next_phase_map = {
            "initialization": "development",
            "development": "creation",
            "creation": "refinement",
            "refinement": "finalization",
            "finalization": "complete"
        }

        if current_phase not in next_phase_map:
            return {"phase": current_phase}

        # Check quality gate
        gate_name = f"{current_phase}_to_{next_phase_map[current_phase]}"
        gate_result = check_quality_gate(
            gate_name,
            state.project.quality_assessment,
            {"quality_gates": configuration.quality_gates}
        )

        if gate_result["passed"]:
            next_phase = next_phase_map[current_phase]
            # Record phase transition in history
            phase_history = state.phase_history.copy()
            if current_phase not in phase_history:
                phase_history[current_phase] = []

            phase_history[current_phase].append({
                "end_time": datetime.now().isoformat(),
                "transition_to": next_phase,
                "quality_assessment": state.project.quality_assessment
            })

            return {
                "phase": next_phase,
                "phase_history": phase_history,
                "messages": [
                    AIMessage(content=f"The project has successfully passed the quality gate from {current_phase} to {next_phase}. Phase transition successful.")
                ]
            }
        else:
            # Stay in current phase
            return {"phase": current_phase}

    builder.add_node("phase_manager", phase_manager)
    builder.set_entry_point("phase_manager")
    builder.add_edge("phase_manager", "__end__")

    # Set up MongoDB checkpointing if configured
    if config.mongodb_connection_string and config.mongodb_database_name:
        checkpointer = MongoDBCheckpointHandler(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database_name,
            collection_name="checkpoint_supervisor"
        )
        graph = builder.compile(checkpointer=checkpointer)
    else:
        # Fallback to memory checkpointing
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

    return graph


def create_storybook_graph(runnable_config: RunnableConfig) -> StateGraph:
    """Create the main storybook graph."""
    config = Configuration.from_runnable_config(runnable_config)
    
    # Create graph with runtime config schema
    builder = StateGraph(NovelSystemState, input_schema=InputState)
    
    async def initialize_project(state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Initialize new project or load existing one.""" 
        input_state = cast(InputState, state.input)
        
        if input_state.project_type == ProjectType.NEW:
            project_data = cast(NewProjectInput, input_state.project_data)
            project_id = str(uuid4())
            
            return {
                "project": {
                    "id": project_id,
                    "title": project_data.title,
                    "synopsis": project_data.synopsis,
                    "manuscript": project_data.manuscript,
                    "notes": project_data.notes,
                    "type": "revision" if project_data.manuscript else "creation",
                    "quality_assessment": {},
                    "created_at": datetime.now().isoformat()
                },
                "phase": "initialization",
                "current_input": {
                    "task": "Initial project assessment and planning",
                    "phase": "initialization"
                }
            }
        else:
            # Load existing project state from checkpointer
            project_data = cast(ExistingProjectInput, input_state.project_data)
            checkpoint = config.get("configurable", {}).get("checkpointer").get_tuple({
                "configurable": {"thread_id": project_data.project_id}
            })
            return checkpoint.checkpoint["channel_values"]

    # Add initialization node
    builder.add_node("initialize", initialize_project)
    builder.set_entry_point("initialize")
    
    # Add phase graphs
    phase_graph = create_phase_graph("initialization", "", config)
    builder.add_subgraph(phase_graph, include_edges=True)
    
    # Add supervisor
    supervisor = create_supervisor_graph(config)
    builder.add_subgraph(supervisor, include_edges=True)
    
    # Connect initialization to phase start
    builder.add_edge("initialize", "executive_director")
    
    # Set up checkpointing
    if config.mongodb_connection_string:
        checkpointer = MongoDBCheckpointHandler(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database_name,
            collection_name="storybook_projects"
        )
    else:
        checkpointer = MemorySaver()
    
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# Exporting individual phase graphs (for backward compatibility)
def get_storybook_supervisor(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the supervisor graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The supervisor graph for managing phase transitions.
    """
    config = Configuration.from_runnable_config(runnable_config)
    return create_supervisor_graph(config)


def get_storybook_initialization(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the initialization phase graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The initialization phase graph.
    """
    config = Configuration.from_runnable_config(runnable_config)
    project_id = runnable_config.get("configurable", {}).get("project_id")
    if not project_id:
        project_id = runnable_config.get("metadata", {}).get("project_id", "default_project")
    return create_phase_graph("initialization", project_id, config)


def get_storybook_development(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the development phase graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The development phase graph.
    """
    config = Configuration.from_runnable_config(runnable_config)
    project_id = runnable_config.get("configurable", {}).get("project_id")
    if not project_id:
        project_id = runnable_config.get("metadata", {}).get("project_id", "default_project")
    return create_phase_graph("development", project_id, config)


def get_storybook_creation(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the creation phase graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The creation phase graph.
    """
    config = Configuration.from_runnable_config(runnable_config)
    project_id = runnable_config.get("configurable", {}).get("project_id")
    if not project_id:
        project_id = runnable_config.get("metadata", {}).get("project_id", "default_project")
    return create_phase_graph("creation", project_id, config)


def get_storybook_refinement(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the refinement phase graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The refinement phase graph.
    """
    config = Configuration.from_runnable_config(runnable_config)
    project_id = runnable_config.get("configurable", {}).get("project_id")
    if not project_id:
        project_id = runnable_config.get("metadata", {}).get("project_id", "default_project")
    return create_phase_graph("refinement", project_id, config)


def get_storybook_finalization(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the finalization phase graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The finalization phase graph.
    """
    config = Configuration.from_runnable_config(runnable_config)
    project_id = runnable_config.get("configurable", {}).get("project_id")
    if not project_id:
        project_id = runnable_config.get("metadata", {}).get("project_id", "default_project")
    return create_phase_graph("finalization", project_id, config)


def get_storybook(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the main storybook graph.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The main storybook graph that integrates all phases.
    """
    return create_storybook_graph(runnable_config)

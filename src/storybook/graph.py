from typing import Dict, Any, List, Literal, cast
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from storybook.configuration import Configuration
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

    # Set up memory checkpointing
    checkpointer = MemorySaver()

    # Name the graph
    builder.name = f"storybook - {phase.capitalize()} Phase"

    # Create and compile the graph with the checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    return graph


def create_supervisor_graph(config: Configuration) -> StateGraph:
    """Create the supervisor graph that manages phase transitions.

    Args:
        config: System configuration.

    Returns:
        A supervisor StateGraph.
    """
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

    # Set up memory checkpointing for the supervisor
    checkpointer = MemorySaver()

    # Name the graph
    builder.name = "storybook - Phase Supervisor"

    # Compile the supervisor graph with the checkpointer
    supervisor = builder.compile(checkpointer=checkpointer)

    return supervisor


def create_storybook_graph(runnable_config: RunnableConfig) -> StateGraph:
    """Create and return the main storybook graph that integrates all phases.

    Args:
        runnable_config: The runnable configuration.

    Returns:
        The main storybook graph incorporating all phases.
    """
    config = Configuration.from_runnable_config(runnable_config)
    project_id = runnable_config.get("configurable", {}).get("project_id")
    if not project_id:
        project_id = runnable_config.get("metadata", {}).get("project_id", "default_project")
    
    # Create the main graph
    builder = StateGraph(NovelSystemState, input=InputState, config_schema=Configuration)
    
    # Create subgraphs for each phase
    initialization_subgraph = create_phase_graph("initialization", project_id, config)
    development_subgraph = create_phase_graph("development", project_id, config)
    creation_subgraph = create_phase_graph("creation", project_id, config)
    refinement_subgraph = create_phase_graph("refinement", project_id, config)
    finalization_subgraph = create_phase_graph("finalization", project_id, config)
    supervisor_subgraph = create_supervisor_graph(config)
    
    # Add all subgraphs to the main graph
    builder.add_node("initialization_phase", initialization_subgraph)
    builder.add_node("development_phase", development_subgraph)
    builder.add_node("creation_phase", creation_subgraph)
    builder.add_node("refinement_phase", refinement_subgraph)
    builder.add_node("finalization_phase", finalization_subgraph)
    builder.add_node("phase_supervisor", supervisor_subgraph)
    
    # Create a router node to direct to the appropriate phase
    def phase_router(state: NovelSystemState) -> str:
        """Route to the appropriate phase based on the current phase in state."""
        current_phase = state.phase
        if current_phase == "initialization":
            return "initialization_phase"
        elif current_phase == "development":
            return "development_phase"
        elif current_phase == "creation":
            return "creation_phase"
        elif current_phase == "refinement":
            return "refinement_phase"
        elif current_phase == "finalization":
            return "finalization_phase"
        elif current_phase == "complete":
            return "__end__"
        else:
            return "phase_supervisor"
    
    # Add the router node
    builder.add_node("router", lambda state: state)
    
    # Set the entry point
    builder.set_entry_point("router")
    
    # Add conditional edges from router to phase subgraphs
    builder.add_conditional_edges("router", phase_router)
    
    # After each phase completes, go back to the supervisor
    builder.add_edge("initialization_phase", "phase_supervisor")
    builder.add_edge("development_phase", "phase_supervisor")
    builder.add_edge("creation_phase", "phase_supervisor")
    builder.add_edge("refinement_phase", "phase_supervisor")
    builder.add_edge("finalization_phase", "phase_supervisor")
    
    # From supervisor back to router
    builder.add_edge("phase_supervisor", "router")
    
    # Set up memory checkpointing
    checkpointer = MemorySaver()
    
    # Name the graph
    builder.name = "storybook - Novel Writing System"
    
    # Compile the main graph with the checkpointer
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

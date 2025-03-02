def get_phase_workflow(config: RunnableConfig) -> StateGraph:
    """Get the workflow graph for a specific phase.

    Args:
        config: A RunnableConfig containing phase, project_id, and agent_factory in its metadata.

    Returns:
        A StateGraph for the specified phase.
    """
    # Extract necessary information from the config's metadata
    metadata = config.get("metadata", {})
    phase = metadata.get("phase")
    project_id = metadata.get("project_id")
    agent_factory = metadata.get("agent_factory")
    
    if not phase or not project_id or not agent_factory:
        raise ValueError("Missing required metadata: phase, project_id, and agent_factory are required")
    
    workflow_map = {
        "initialization": create_initialization_graph,
        "development": create_development_graph,
        "creation": create_creation_graph,
        "refinement": create_refinement_graph,
        "finalization": create_finalization_graph,
    }

    if phase not in workflow_map:
        raise ValueError(f"Unknown phase: {phase}")

    # Pass the config directly to the workflow creation function
    return workflow_map[phase](config)
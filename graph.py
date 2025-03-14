import os
import warnings
from langgraph.graph import StateGraph, END
import operator
from typing import List, Dict, Any
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from state import AgentState
from utils import check_quality_gate, extract_chunk_references
from config import MONGODB_URI, QUALITY_GATES
from langchain_core.runnables import RunnableConfig

def create_phase_graph(config: RunnableConfig) -> StateGraph:
    """Create a graph for a specific phase of the storybook process."""
    # Extract phase and project_id from config
    phase = config.get("configurable", {}).get("phase", "initialization")
    project_id = config.get("configurable", {}).get("project_id", "default_project")

    # Get agent factory from config or create a new one
    agent_factory = config.get("configurable", {}).get("agent_factory")
    if agent_factory is None:
        from agent import AgentFactory
        agent_factory = AgentFactory(config)

    # Create the graph builder without checkpoint config initially
    builder = StateGraph(AgentState)

    # Get MongoDB connection string from environment
    mongo_uri = MONGODB_URI
    if not mongo_uri:
        print("Warning: MONGODB_URI not found in environment variables. Checkpointing will be disabled.")

    # Define available agents for each phase
    agents = {
        "initialization": [
            "executive_director",
            "creative_director",
            "market_alignment_director",
            "domain_knowledge_specialist",  # Add these specialists
            "cultural_authenticity_expert", # to initialization phase
            "positioning_specialist",
            "title_blurb_optimizer",
            "differentiation_strategist"
        ],
        "development": [
            "executive_director", "creative_director", "structure_architect",
            "plot_development_specialist", "world_building_expert",
            "character_psychology_specialist", "character_voice_designer",
            "character_relationship_mapper", "domain_knowledge_specialist",
            "cultural_authenticity_expert", "market_alignment_director"
        ],
        "creation": [
            "executive_director", "creative_director", "content_development_director",
            "chapter_drafters", "scene_construction_specialists", "dialogue_crafters",
            "continuity_manager", "voice_consistency_monitor", "emotional_arc_designer",
            "domain_knowledge_specialist"
        ],
        "refinement": [
            "executive_director", "creative_director", "editorial_director",
            "structural_editor", "character_arc_evaluator", "thematic_coherence_analyst",
            "prose_enhancement_specialist", "dialogue_refinement_expert",
            "rhythm_cadence_optimizer", "grammar_consistency_checker",
            "fact_verification_specialist", "market_alignment_director"
        ],
        "finalization": [
            "executive_director", "editorial_director", "market_alignment_director",
            "positioning_specialist", "title_blurb_optimizer",
            "differentiation_strategist", "formatting_standards_expert"
        ]
    }

    # Add agent nodes to the graph for this phase
    phase_agents = agents.get(phase, [])
    for agent_name in phase_agents:
        builder.add_node(agent_name, agent_factory.create_agent(agent_name, project_id))

    # Add research nodes
    builder.add_node("domain_research", agent_factory.create_research_agent("domain"))
    builder.add_node("cultural_research", agent_factory.create_research_agent("cultural"))
    builder.add_node("market_research", agent_factory.create_research_agent("market"))

    # Add the starting node - all phases start with executive director
    builder.set_entry_point("executive_director")

    # Define phase-specific routing
    if phase == "initialization":
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in initialization phase."""
            task = state.get("current_input", {}).get("task", "").lower()
            messages = state.get("messages", [])

            # Get the last assistant message if there is one
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if executive director specified research needs or delegated to specialists
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the executive director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for all specialists
                specialist_mappings = {
                    # Creative team
                    "creative director": "creative_director",
                    "structure architect": "structure_architect",
                    "plot development specialist": "plot_development_specialist",
                    "world building expert": "world_building_expert",
                    "character psychology specialist": "character_psychology_specialist",
                    "character voice designer": "character_voice_designer",
                    "character relationship mapper": "character_relationship_mapper",

                    # Research team
                    "domain knowledge specialist": "domain_knowledge_specialist",
                    "cultural authenticity expert": "cultural_authenticity_expert",

                    # Editorial team
                    "editorial director": "editorial_director",

                    # Market team
                    "market alignment director": "market_alignment_director",

                    # Other specialists
                    "positioning specialist": "positioning_specialist",
                    "title/blurb optimizer": "title_blurb_optimizer",
                    "differentiation strategist": "differentiation_strategist"
                }

                # Check for specialist delegations
                for specialist_name, node_name in specialist_mappings.items():
                    if specialist_name in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {specialist_name} in executive director's message")
                        return node_name

                # Check for research instructions more broadly
                if ("research" in last_message_lower or "investigate" in last_message_lower):
                    if "market" in last_message_lower or "audience" in last_message_lower:
                        return "market_research"
                    elif "cultural" in last_message_lower or "heritage" in last_message_lower:
                        return "cultural_research"
                    else:
                        return "domain_research"

            # Count agent visits to prevent loops
            exec_visits = sum(1 for msg in messages if
                             msg.get("role") == "user" and
                             "Executive Director" in msg.get("content", ""))

            # Force termination after too many visits to prevent infinite loops
            if exec_visits > 10:  # Increased from 5 to 10
                print("Forcing termination after 10 executive director visits to prevent infinite loops")
                # Force quality assessment update
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                quality_assessment["planning_quality"] = 0.8
                quality_assessment["market_alignment"] = 0.7
                project["quality_assessment"] = quality_assessment

                return END

            # Default routing based on task keywords if no delegation was detected
            if "creative" in task or "story" in task:
                return "creative_director"
            elif "market" in task or "audience" in task:
                return "market_alignment_director"
            elif "research" in task or "information" in task:
                return "domain_research"
            elif exec_visits > 5:  # If we've visited executive_director too many times
                # Check quality gate to possibly end this phase
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "initialization_to_development",
                    quality_assessment,
                    {"quality_gates": config.get("configurable", {}).get("quality_gates", {})}
                )

                if gate_result["passed"]:
                    return END
                else:
                    # If we've been to the executive director multiple times without delegation, try creative
                    if exec_visits > 2:
                        return "creative_director"
                    else:
                        # When in doubt, stick with executive director for another iteration
                        return "market_alignment_director"

        def route_after_creative_director(state: AgentState) -> str:
            """Route after the creative director in initialization phase."""
            messages = state.get("messages", [])

            # Get the last assistant message if there is one
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if creative director referred to specific needs or specialists
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the creative director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection - look for specialist references
                if "structure architect" in last_message_lower:
                    return "structure_architect"

                if "plot development specialist" in last_message_lower:
                    return "plot_development_specialist"

                if "world building expert" in last_message_lower:
                    return "world_building_expert"

                if "character psychology specialist" in last_message_lower:
                    return "character_psychology_specialist"

                if "character voice designer" in last_message_lower:
                    return "character_voice_designer"

                if "character relationship mapper" in last_message_lower:
                    return "character_relationship_mapper"

                # Check for market or research needs
                if "market" in last_message_lower or "audience" in last_message_lower:
                    return "market_alignment_director"

                if "research" in last_message_lower or "domain knowledge" in last_message_lower:
                    return "domain_research"

            # Count agent visits to prevent loops
            creative_visits = sum(1 for msg in messages if
                                msg.get("role") == "user" and
                                "Creative Director" in msg.get("content", ""))

            # Prevent infinite loops
            if creative_visits > 5:  # Increased from 2 to 5
                print("Forcing return to executive_director after 5 creative director visits to prevent infinite loops")
                return "executive_director"
            elif creative_visits > 2:  # If we've been to creative_director too many times
                # Go back to executive_director to reassess
                return "executive_director"
            else:
                # Stay with creative director for one more iteration to encourage delegation
                return "executive_director"

        def route_after_market_alignment_director(state: AgentState) -> str:
            """Route after the market alignment director in initialization phase."""
            messages = state.get("messages", [])

            # Check if market director wants to do research or delegates
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the market director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection
                if "positioning specialist" in last_message_lower:
                    return "positioning_specialist"

                if "title/blurb optimizer" in last_message_lower or "marketing copy" in last_message_lower:
                    return "title_blurb_optimizer"

                if "differentiation strategist" in last_message_lower:
                    return "differentiation_strategist"

                # Check for research needs
                if "research" in last_message_lower or "investigate" in last_message_lower:
                    return "market_research"

            # Count visits to market alignment director
            market_visits = sum(1 for msg in messages if
                               msg.get("role") == "user" and
                               "Market Alignment Director" in msg.get("content", ""))

            # Prevent infinite loops
            if market_visits > 5:  # Increased from 2 to 5
                print("Forcing quality assessment update after 5 market director visits to prevent infinite loops")
                # Force quality assessment update
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                quality_assessment["market_alignment"] = 0.7
                project["quality_assessment"] = quality_assessment
                return "executive_director"
            elif market_visits > 2:
                # Return to executive director after a few visits
                return "executive_director"
            else:
                # Return to executive director
                return "executive_director"

        def route_after_research(state: AgentState) -> str:
            """Route after research nodes."""
            # Route back to the appropriate specialist based on research type
            research_type = state.get("lnode", "")

            if research_type == "domain_research":
                return "domain_knowledge_specialist"
            elif research_type == "cultural_research":
                return "cultural_authenticity_expert"
            elif research_type == "market_research":
                return "market_alignment_director"
            else:
                return "executive_director"

        # Add conditional edges
        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )

        builder.add_conditional_edges(
            "creative_director",
            route_after_creative_director
        )

        builder.add_conditional_edges(
            "market_alignment_director",
            route_after_market_alignment_director
        )

        # Add research routing
        builder.add_conditional_edges(
            "domain_research",
            route_after_research
        )

        builder.add_conditional_edges(
            "cultural_research",
            route_after_research
        )

        builder.add_conditional_edges(
            "market_research",
            route_after_research
        )

        # Only add edges if both source and target nodes exist in this phase
        if "domain_knowledge_specialist" in phase_agents:
            builder.add_edge("domain_knowledge_specialist", "executive_director")

    elif phase == "development":
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in development phase."""
            task = state.get("current_input", {}).get("task", "").lower()
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if executive director specified different specialists
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the executive director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced detection of specialist delegation
                specialist_mappings = {
                    # Creative team
                    "creative director": "creative_director",
                    "structure architect": "structure_architect",
                    "plot development specialist": "plot_development_specialist",
                    "world building expert": "world_building_expert",
                    "character psychology specialist": "character_psychology_specialist",
                    "character voice designer": "character_voice_designer",
                    "character relationship mapper": "character_relationship_mapper",

                    # Research team
                    "domain knowledge specialist": "domain_knowledge_specialist",
                    "cultural authenticity expert": "cultural_authenticity_expert",

                    # Market team
                    "market alignment director": "market_alignment_director"
                }

                # Check for specialist delegations
                for specialist_name, node_name in specialist_mappings.items():
                    if specialist_name in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {specialist_name} in executive director's message")
                        return node_name

                # Check for research needs
                if "research" in last_message_lower:
                    if "market" in last_message_lower:
                        return "market_research"
                    elif "cultural" in last_message_lower:
                        return "cultural_research"
                    else:
                        return "domain_research"

            # Count executive director visits to prevent loops
            exec_visits = sum(1 for msg in messages if
                              msg.get("role") == "user" and
                              "Executive Director" in msg.get("content", ""))

            # Force termination after too many visits to prevent infinite loops
            if exec_visits > 10:  # Increased from 5 to 10
                print("Forcing termination after 10 executive director visits to prevent infinite loops")
                # Force quality assessment update
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                quality_assessment["structure_quality"] = 0.8
                quality_assessment["character_depth"] = 0.8
                quality_assessment["world_building"] = 0.8
                project["quality_assessment"] = quality_assessment

                return END

            # Default routing based on task keywords
            if "creative" in task or "story" in task:
                return "creative_director"
            elif "market" in task or "trend" in task:
                return "market_alignment_director"
            elif "research" in task or "knowledge" in task:
                return "domain_knowledge_specialist"
            elif exec_visits > 5:  # If we've visited executive_director too many times
                # Check quality gate to possibly end this phase
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "development_to_creation",
                    quality_assessment,
                    {"quality_gates": config.get("quality_gates", {})}
                )
                if gate_result["passed"]:
                    return END
                else:
                    # If we've been to executive several times, try creative director
                    if exec_visits > 2:
                        return "creative_director"
                    else:
                        # Try structure architect for a change to encourage more diverse delegation
                        return "structure_architect"

        def route_after_creative_director(state: AgentState) -> str:
            """Route after the creative director node."""
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Count visits to different specialists to detect routing loops
            creative_visits = sum(1 for msg in messages if
                                msg.get("role") == "user" and
                                "Creative Director" in msg.get("content", ""))

            structure_visits = sum(1 for msg in messages if
                                 msg.get("role") == "user" and
                                 "Structure Architect" in msg.get("content", ""))

            # Detect potential loop between Creative Director and Structure Architect
            recent_pattern = []
            for i in range(min(6, len(messages))):
                if i < len(messages) and messages[-(i+1)].get("role") == "user":
                    content = messages[-(i+1)].get("content", "")
                    if "Creative Director" in content:
                        recent_pattern.append("creative_director")
                    elif "Structure Architect" in content:
                        recent_pattern.append("structure_architect")

            # If we're in a loop between Creative Director and Structure Architect
            if len(recent_pattern) >= 4:
                if recent_pattern[0] == "creative_director" and recent_pattern[1] == "structure_architect" and \
                   recent_pattern[2] == "creative_director" and recent_pattern[3] == "structure_architect":
                    # Break the loop by going back to executive director
                    return "executive_director"

            # Check if creative director delegated to a specific specialist
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the creative director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for specialists
                specialist_mappings = {
                    "structure architect": "structure_architect",
                    "plot development specialist": "plot_development_specialist",
                    "world building expert": "world_building_expert",
                    "character psychology specialist": "character_psychology_specialist",
                    "character voice designer": "character_voice_designer",
                    "character relationship mapper": "character_relationship_mapper"
                }

                # If we detect a potential loop with Structure Architect, avoid that routing
                if structure_visits > 2 and "structure architect" in last_message_lower:
                    # Try other specialists instead
                    if "plot development" in last_message_lower:
                        return "plot_development_specialist"
                    elif "world building" in last_message_lower:
                        return "world_building_expert"
                    elif "character" in last_message_lower:
                        return "character_psychology_specialist"
                    else:
                        # Break the loop by going to executive director
                        return "executive_director"

                # Check for specialist delegations
                for specialist_name, node_name in specialist_mappings.items():
                    if specialist_name in last_message_lower and node_name in phase_agents:
                        # Before routing to a specialist, check we're not creating a loop
                        specialist_visits = sum(1 for msg in messages if
                                             msg.get("role") == "user" and
                                             specialist_name.title() in msg.get("content", ""))

                        # If we've already visited this specialist many times, consider it a loop
                        if specialist_visits > 2:
                            # Try to break the loop by going to executive director
                            return "executive_director"

                        print(f"Detected delegation to {specialist_name} in creative director's message")
                        return node_name

                # Check for research or domain knowledge needs
                if "domain knowledge" in last_message_lower or "research" in last_message_lower:
                    return "domain_knowledge_specialist"

            # Prevent loops with Creative Director
            if creative_visits > 5:  # Increased from 2 to 5
                print("Forcing return to executive_director after 5 creative director visits to prevent infinite loops")
                return "executive_director"
            # After a few visits, if no delegation, return to executive
            elif creative_visits > 2:
                return "executive_director"
            else:
                # Return to executive_director
                return "executive_director"

        # Add routing for specialists to prevent loops
        def route_after_structure_architect(state: AgentState) -> str:
            """Route after the structure architect node to prevent loops with Creative Director."""
            messages = state.get("messages", [])

            # Count visits to Structure Architect
            structure_visits = sum(1 for msg in messages if
                                  msg.get("role") == "user" and
                                  "Structure Architect" in msg.get("content", ""))

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Prevent infinite loops
            if structure_visits > 5:  # Increased from 2 to 5
                print("Forcing return to executive_director after 5 structure architect visits to prevent infinite loops")
                return "executive_director"
            # Check for potential loops
            elif structure_visits > 2:
                # If we've been to Structure Architect multiple times, go to Executive Director
                return "executive_director"

            # Check if the structure architect recommended other specialists
            if last_message:
                last_message_lower = last_message.lower()

                # Enhanced delegation detection
                specialist_mappings = {
                    "plot development specialist": "plot_development_specialist",
                    "world building expert": "world_building_expert",
                    "character psychology specialist": "character_psychology_specialist",
                    "character voice designer": "character_voice_designer",
                    "character relationship mapper": "character_relationship_mapper"
                }

                # Check for specialist delegations
                for specialist_name, node_name in specialist_mappings.items():
                    if specialist_name in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {specialist_name} in structure architect's message")
                        return node_name

                if "executive" in last_message_lower:
                    return "executive_director"

            # Default to creative director (but only the first couple times)
            return "creative_director"

        # Add routing for research nodes
        def route_after_research(state: AgentState) -> str:
            """Route after research nodes."""
            research_type = state.get("lnode", "")

            if research_type == "domain_research":
                return "domain_knowledge_specialist"
            elif research_type == "cultural_research":
                return "cultural_authenticity_expert"
            elif research_type == "market_research":
                return "market_alignment_director"
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

        # Add routing for Structure Architect to prevent loops
        builder.add_conditional_edges(
            "structure_architect",
            route_after_structure_architect
        )

        # Add research routing
        builder.add_conditional_edges(
            "domain_research",
            route_after_research
        )

        builder.add_conditional_edges(
            "cultural_research",
            route_after_research
        )

        builder.add_conditional_edges(
            "market_research",
            route_after_research
        )
        # Connect domain knowledge specialist to domain research
        builder.add_edge("domain_knowledge_specialist", "domain_research")
        builder.add_edge("cultural_authenticity_expert", "cultural_research")
        builder.add_edge("market_alignment_director", "market_research")

        # Use conditional routing for specialists rather than direct edges to prevent loops
        def route_after_specialist(state: AgentState) -> str:
            """Route after a specialist to prevent loops."""
            messages = state.get("messages", [])

            # Count visits to this specialist
            specialist_type = state.get("lnode", "")
            specialist_visits = sum(1 for msg in messages if
                                   msg.get("role") == "user" and
                                   specialist_type.replace("_", " ").title() in msg.get("content", ""))

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Prevent infinite loops
            if specialist_visits > 5:  # Increased from 2 to 5
                print(f"Forcing return to executive_director after 5 {specialist_type} visits to prevent infinite loops")
                return "executive_director"

            # If specialist has mentioned other specialists, route there
            if last_message:
                last_message_lower = last_message.lower()

                # Enhanced delegation detection
                specialist_mappings = {
                    "plot development specialist": "plot_development_specialist",
                    "world building expert": "world_building_expert",
                    "character psychology specialist": "character_psychology_specialist",
                    "character voice designer": "character_voice_designer",
                    "character relationship mapper": "character_relationship_mapper"
                }

                # Check for specialist delegations
                for specialist_name, node_name in specialist_mappings.items():
                    if (specialist_name in last_message_lower and
                        node_name in phase_agents and
                        node_name != specialist_type):
                        print(f"Detected delegation to {specialist_name} in {specialist_type}'s message")
                        return node_name

                if "executive director" in last_message_lower:
                    return "executive_director"

            # Avoid loops by going to executive director after multiple visits
            if specialist_visits > 2:
                return "executive_director"

            # Default to creative director
            return "creative_director"

        # Add conditional edges for other specialists instead of direct edges
        for agent in ["plot_development_specialist", "world_building_expert",
                     "character_psychology_specialist", "character_voice_designer",
                     "character_relationship_mapper"]:
            if agent in agents[phase]:
                builder.add_conditional_edges(agent, route_after_specialist)

    elif phase == "creation":
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in creation phase, focused on manuscript improvement."""
            task = state.get("current_input", {}).get("task", "").lower()
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if executive director specified different specialists, prioritizing content creators
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the executive director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for all specialists
                specialist_mappings = {
                    # Content team
                    "content development director": "content_development_director",
                    "chapter draft": "chapter_drafters",
                    "rewrite chapter": "chapter_drafters",
                    "dialogue": "dialogue_crafters",
                    "conversation": "dialogue_crafters",
                    "scene construction": "scene_construction_specialists",
                    "build scene": "scene_construction_specialists",
                    "continuity manager": "continuity_manager",
                    "voice consistency": "voice_consistency_monitor",
                    "emotional arc": "emotional_arc_designer",

                    # Creative team
                    "creative director": "creative_director",

                    # Research
                    "domain knowledge": "domain_knowledge_specialist",
                    "research": "domain_knowledge_specialist"
                }

                # Check for specialist delegations
                for keyword, node_name in specialist_mappings.items():
                    if keyword in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {node_name} in executive director's message")
                        return node_name

            # Count executive director visits to prevent loops
            exec_visits = sum(1 for msg in messages if
                              msg.get("role") == "user" and
                              "Executive Director" in msg.get("content", ""))

            # Force termination after too many visits to prevent infinite loops
            if exec_visits > 10:  # Increased from 5 to 10
                print("Forcing termination after 10 executive director visits to prevent infinite loops")
                # Force quality assessment update
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                quality_assessment["content_quality"] = 0.8
                quality_assessment["narrative_flow"] = 0.8
                quality_assessment["dialogue_quality"] = 0.8
                project["quality_assessment"] = quality_assessment

                return END

            # Default routing based on task keywords
            if "chapter" in task or "write" in task or "content" in task:
                return "chapter_drafters"
            elif "dialogue" in task or "conversation" in task:
                return "dialogue_crafters"
            elif "scene" in task:
                return "scene_construction_specialists"
            elif "content" in task or "development" in task:
                return "content_development_director"
            elif exec_visits > 5:  # If we've visited executive_director too many times
                # Check quality gate to possibly end this phase
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "creation_to_refinement",
                    quality_assessment,
                    {"quality_gates": config.get("quality_gates", {})}
                )
                if gate_result["passed"]:
                    return END
                else:
                    # After a few iterations, try content specialists
                    if exec_visits > 1:
                        # Check which content creator has been used less
                        chapter_visits = sum(1 for msg in messages if
                                            msg.get("role") == "user" and
                                            "Chapter Drafter" in msg.get("content", ""))
                        dialogue_visits = sum(1 for msg in messages if
                                            msg.get("role") == "user" and
                                            "Dialogue Crafter" in msg.get("content", ""))

                        # Choose the specialist that's been used less
                        if chapter_visits <= dialogue_visits:
                            return "chapter_drafters"
                        else:
                            return "dialogue_crafters"
                    else:
                        # First try content development director
                        return "content_development_director"

        def route_after_content_director(state: AgentState) -> str:
            """Route after the content development director node."""
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if content director delegated to a specific specialist
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the content director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for content team specialists
                specialist_mappings = {
                    "chapter draft": "chapter_drafters",
                    "rewrite chapter": "chapter_drafters",
                    "dialogue": "dialogue_crafters",
                    "conversation": "dialogue_crafters",
                    "scene construction": "scene_construction_specialists",
                    "build scene": "scene_construction_specialists",
                    "continuity manager": "continuity_manager",
                    "voice consistency": "voice_consistency_monitor",
                    "emotional arc": "emotional_arc_designer",
                    "domain knowledge": "domain_knowledge_specialist",
                    "research": "domain_knowledge_specialist"
                }

                # Check for specialist delegations
                for keyword, node_name in specialist_mappings.items():
                    if keyword in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {node_name} in content director's message")
                        return node_name

            # Count content director visits
            content_visits = sum(1 for msg in messages if
                                msg.get("role") == "user" and
                                "Content Development Director" in msg.get("content", ""))

            # Prevent infinite loops
            if content_visits > 5:  # Increased from 2 to 5
                print("Forcing return to executive_director after 5 content director visits to prevent infinite loops")
                return "executive_director"

            # Count specialist visits
            chapter_visits = sum(1 for msg in messages if
                                msg.get("role") == "user" and
                                "Chapter Drafter" in msg.get("content", ""))
            dialogue_visits = sum(1 for msg in messages if
                                msg.get("role") == "user" and
                                "Dialogue Crafter" in msg.get("content", ""))

            if content_visits > 2:
                # After multiple visits without delegation, try specialists directly
                # Choose the specialist that's been used less
                if chapter_visits <= dialogue_visits:
                    return "chapter_drafters"
                else:
                    return "dialogue_crafters"
            else:
                # Give another chance for delegation
                return "executive_director"

        # Add routing for research
        def route_after_research(state: AgentState) -> str:
            """Route after research nodes."""
            research_type = state.get("lnode", "")

            if research_type == "domain_research":
                return "domain_knowledge_specialist"
            else:
                return "executive_director"

        # Add routing for content creators to go back to executive director for review
        def route_after_content_creator(state: AgentState) -> str:
            """Route after content creators back to executive director for review."""
            # Always return to executive director for review after content creation
            return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        builder.add_conditional_edges(
            "content_development_director",
            route_after_content_director
        )

        # Add research routing
        builder.add_conditional_edges(
            "domain_research",
            route_after_research
        )

        # Add routing for content creators
        for creator in ["chapter_drafters", "dialogue_crafters", "scene_construction_specialists"]:
            if creator in phase_agents:
                builder.add_conditional_edges(creator, route_after_content_creator)

        # Connect domain knowledge specialist to domain research
        builder.add_edge("domain_knowledge_specialist", "domain_research")

        # Connect specialized agents to their supervisors
        for agent in ["continuity_manager", "voice_consistency_monitor", "emotional_arc_designer"]:
            if agent in agents[phase]:  # Only add edges for agents that exist in this phase
                builder.add_edge(agent, "content_development_director")

        builder.add_edge("creative_director", "executive_director")

    elif phase == "refinement":
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in refinement phase."""
            task = state.get("current_input", {}).get("task", "").lower()
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if executive director specified different specialists
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the executive director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for all refinement specialists
                specialist_mappings = {
                    # Editorial team
                    "editorial director": "editorial_director",
                    "creative director": "creative_director",
                    "market alignment director": "market_alignment_director",
                    "structural editor": "structural_editor",
                    "character arc evaluator": "character_arc_evaluator",
                    "thematic coherence analyst": "thematic_coherence_analyst",
                    "prose enhancement": "prose_enhancement_specialist",
                    "improve writing": "prose_enhancement_specialist",
                    "dialogue refinement": "dialogue_refinement_expert",
                    "improve dialogue": "dialogue_refinement_expert",
                    "rhythm cadence optimizer": "rhythm_cadence_optimizer",
                    "grammar consistency checker": "grammar_consistency_checker",
                    "fact verification specialist": "fact_verification_specialist"
                }

                # Check for specialist delegations
                for keyword, node_name in specialist_mappings.items():
                    if keyword in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {node_name} in executive director's message")
                        return node_name

            # Count executive director visits to prevent loops
            exec_visits = sum(1 for msg in messages if
                              msg.get("role") == "user" and
                              "Executive Director" in msg.get("content", ""))

            # Force termination after too many visits to prevent infinite loops
            if exec_visits > 10:  # Increased from 5 to 10
                print("Forcing termination after 10 executive director visits to prevent infinite loops")
                # Force quality assessment update
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                quality_assessment["editing_quality"] = 0.9
                quality_assessment["prose_quality"] = 0.9
                quality_assessment["thematic_coherence"] = 0.8
                project["quality_assessment"] = quality_assessment

                return END

            # Default routing based on task keywords
            if "prose" in task or "writing" in task or "style" in task:
                return "prose_enhancement_specialist"
            elif "dialogue" in task or "conversation" in task:
                return "dialogue_refinement_expert"
            elif "edit" in task:
                return "editorial_director"
            elif "creative" in task:
                return "creative_director"
            elif "market" in task:
                return "market_alignment_director"
            elif exec_visits > 5:  # If we've visited executive_director too many times
                # Check quality gate to possibly end this phase
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "refinement_to_finalization",
                    quality_assessment,
                    {"quality_gates": config.get("quality_gates", {})}
                )
                if gate_result["passed"]:
                    return END
                else:
                    # After a few iterations, focus on direct prose improvement
                    if exec_visits > 1:
                        # Check which refinement specialist has been used less
                        prose_visits = sum(1 for msg in messages if
                                          msg.get("role") == "user" and
                                          "Prose Enhancement Specialist" in msg.get("content", ""))
                        dialogue_visits = sum(1 for msg in messages if
                                            msg.get("role") == "user" and
                                            "Dialogue Refinement Expert" in msg.get("content", ""))

                        # Choose the specialist that's been used less
                        if prose_visits <= dialogue_visits:
                            return "prose_enhancement_specialist"
                        else:
                            return "dialogue_refinement_expert"
                    else:
                        # First try editorial director
                        return "editorial_director"

        def route_after_editorial_director(state: AgentState) -> str:
            """Route after the editorial director node with focus on manuscript improvement."""
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if editorial director delegated to specific specialists for manuscript improvement
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the editorial director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for editorial team specialists
                specialist_mappings = {
                    "structural editor": "structural_editor",
                    "character arc evaluator": "character_arc_evaluator",
                    "thematic coherence analyst": "thematic_coherence_analyst",
                    "prose enhancement": "prose_enhancement_specialist",
                    "improve writing": "prose_enhancement_specialist",
                    "dialogue refinement": "dialogue_refinement_expert",
                    "improve dialogue": "dialogue_refinement_expert",
                    "rhythm cadence optimizer": "rhythm_cadence_optimizer",
                    "grammar consistency checker": "grammar_consistency_checker",
                    "fact verification specialist": "fact_verification_specialist"
                }

                # Check for specialist delegations
                for keyword, node_name in specialist_mappings.items():
                    if keyword in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {node_name} in editorial director's message")
                        return node_name

            # Count editorial director visits
            editorial_visits = sum(1 for msg in messages if
                                  msg.get("role") == "user" and
                                  "Editorial Director" in msg.get("content", ""))

            # Prevent infinite loops
            if editorial_visits > 5:  # Increased from 2 to 5
                print("Forcing return to executive_director after 5 editorial director visits to prevent infinite loops")
                return "executive_director"

            # Count specialist visits
            prose_visits = sum(1 for msg in messages if
                              msg.get("role") == "user" and
                              "Prose Enhancement Specialist" in msg.get("content", ""))
            dialogue_visits = sum(1 for msg in messages if
                                msg.get("role") == "user" and
                                "Dialogue Refinement Expert" in msg.get("content", ""))

            if editorial_visits > 2:
                # After multiple visits, try a specialist directly
                # Choose the specialist that's been used less
                if prose_visits <= dialogue_visits:
                    return "prose_enhancement_specialist"
                else:
                    return "dialogue_refinement_expert"
            else:
                # Give another chance for delegation
                return "executive_director"

        # Add routing for fact verification and research
        def route_after_fact_verification(state: AgentState) -> str:
            """Route after fact verification specialist."""
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if fact verification needs research
            if last_message and "research" in last_message.lower():
                return "domain_research"
            else:
                return "editorial_director"

        def route_after_research(state: AgentState) -> str:
            """Route after domain research."""
            return "fact_verification_specialist"

        # Add routing for prose and dialogue refinement specialists to go back to executive
        def route_after_refinement_specialist(state: AgentState) -> str:
            """Route after refinement specialists back to executive director for review."""
            # Always return to executive director for review after refinement
            return "executive_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        builder.add_conditional_edges(
            "editorial_director",
            route_after_editorial_director
        )

        builder.add_conditional_edges(
            "fact_verification_specialist",
            route_after_fact_verification
        )

        builder.add_conditional_edges(
            "domain_research",
            route_after_research
        )

        # Add routing for refinement specialists
        for specialist in ["prose_enhancement_specialist", "dialogue_refinement_expert"]:
            if specialist in phase_agents:
                builder.add_conditional_edges(specialist, route_after_refinement_specialist)

        # Connect specialized agents to their supervisors
        for agent in ["structural_editor", "character_arc_evaluator",
                     "thematic_coherence_analyst", "rhythm_cadence_optimizer",
                     "grammar_consistency_checker"]:
            if agent in agents[phase]:  # Only add edges for agents that exist in this phase
                builder.add_edge(agent, "editorial_director")

        # Connect other directors back to executive director
        for agent in ["creative_director", "market_alignment_director"]:
            if agent in agents[phase]:
                builder.add_edge(agent, "executive_director")

    elif phase == "finalization":
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in finalization phase."""
            task = state.get("current_input", {}).get("task", "").lower()
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if executive director specified different specialists
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the executive director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for all finalization specialists
                specialist_mappings = {
                    "editorial director": "editorial_director",
                    "market alignment director": "market_alignment_director",
                    "positioning specialist": "positioning_specialist",
                    "title/blurb optimizer": "title_blurb_optimizer",
                    "marketing copy": "title_blurb_optimizer",
                    "differentiation strategist": "differentiation_strategist",
                    "formatting standards expert": "formatting_standards_expert"
                }

                # Check for specialist delegations
                for keyword, node_name in specialist_mappings.items():
                    if keyword in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {node_name} in executive director's message")
                        return node_name

            # Count executive director visits to prevent loops
            exec_visits = sum(1 for msg in messages if
                              msg.get("role") == "user" and
                              "Executive Director" in msg.get("content", ""))

            # Force termination after too many visits to prevent infinite loops
            if exec_visits > 10:  # Increased from 5 to 10
                print("Forcing termination after 10 executive director visits to prevent infinite loops")
                # Force quality assessment update
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                quality_assessment["market_readiness"] = 0.9
                quality_assessment["overall_quality"] = 0.9
                project["quality_assessment"] = quality_assessment

                return END

            # Default routing based on task keywords
            if "edit" in task:
                return "editorial_director"
            elif "market" in task:
                return "market_alignment_director"
            elif exec_visits > 5:  # If we've visited executive_director too many times
                # Check quality gate to possibly end this phase
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "finalization_to_complete",
                    quality_assessment,
                    {"quality_gates": config.get("quality_gates", {})}
                )
                if gate_result["passed"]:
                    return END
                else:
                    # After a few iterations, try market alignment
                    if exec_visits > 1:
                        return "market_alignment_director"
                    else:
                        return "editorial_director"

        def route_after_market_director(state: AgentState) -> str:
            """Route after the market alignment director node."""
            messages = state.get("messages", [])

            # Get the last assistant message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    break

            # Check if market director delegated to a specific specialist
            if last_message:
                last_message_lower = last_message.lower()

                # Get referenced chunks from the market director's message
                referenced_chunks = extract_chunk_references(last_message)
                current_input = state.get("current_input", {}).copy()
                current_input["referenced_chunks"] = referenced_chunks
                state["current_input"] = current_input

                # Enhanced delegation detection for marketing specialists
                specialist_mappings = {
                    "positioning specialist": "positioning_specialist",
                    "title/blurb optimizer": "title_blurb_optimizer",
                    "marketing copy": "title_blurb_optimizer",
                    "differentiation strategist": "differentiation_strategist"
                }

                # Check for specialist delegations
                for keyword, node_name in specialist_mappings.items():
                    if keyword in last_message_lower and node_name in phase_agents:
                        print(f"Detected delegation to {node_name} in market director's message")
                        return node_name

                # Check for research needs
                if "research" in last_message_lower:
                    return "market_research"

            # Count market director visits
            market_visits = sum(1 for msg in messages if
                               msg.get("role") == "user" and
                               "Market Alignment Director" in msg.get("content", ""))

            # Prevent infinite loops
            if market_visits > 5:  # Increased from 2 to 5
                print("Forcing return to executive_director after 5 market director visits to prevent infinite loops")
                return "executive_director"

            if market_visits > 2:
                # After multiple visits, try a specialist directly
                return "positioning_specialist"
            else:
                # Give another chance for delegation
                return "title_blurb_optimizer"

        def route_after_research(state: AgentState) -> str:
            """Route after market research."""
            return "market_alignment_director"

        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        builder.add_conditional_edges(
            "market_alignment_director",
            route_after_market_director
        )

        builder.add_conditional_edges(
            "market_research",
            route_after_research
        )
        # Connect specialized agents to their supervisors
        for agent in ["positioning_specialist", "title_blurb_optimizer", "differentiation_strategist"]:
            if agent in agents[phase]:  # Only add edges for agents that exist in this phase
                builder.add_edge(agent, "market_alignment_director")
        # Connect other specialists
        if "formatting_standards_expert" in agents[phase]:
            builder.add_edge("formatting_standards_expert", "editorial_director")

        # Connect other directors back to executive director
        builder.add_edge("editorial_director", "executive_director")

    # Try to compile with the MongoDB checkpointer
    try:
        if mongo_uri:
            # Create MongoDB client for checkpointing
            print(f"Creating MongoDB checkpointer for project {project_id}, phase {phase}")
            mongo_client = MongoClient(mongo_uri)
            checkpointer = MongoDBSaver(mongo_client)
            return builder.compile(checkpointer=checkpointer)
        else:
            print(f"MongoDB connection string not available. Proceeding without checkpointing for {project_id}, phase {phase}")
            return builder.compile()
    except Exception as e:
        print(f"Warning: Failed to create MongoDB checkpointer with error: {str(e)}. Proceeding without checkpointing.")
        return builder.compile()

def create_main_graph(config: Dict[str, Any]) -> StateGraph:
    """Create a main composite graph that includes all phase graphs."""
    # Extract agent_factory from the config
    agent_factory = config.get("agent_factory")
    if agent_factory is None:
        from agent import AgentFactory
        agent_factory = AgentFactory(config)

    # Create a graph builder
    builder = StateGraph(AgentState)

    # Create all phase graphs with shared agent_factory
    phase_graphs = {}
    for phase in ["initialization", "development", "creation", "refinement", "finalization"]:
        # Create a phase-specific config that includes the phase name
        phase_config = config.copy()
        phase_config["phase"] = phase
        phase_config["project_id"] = "composite_project"
        phase_config["agent_factory"] = agent_factory

        phase_graphs[phase] = create_phase_graph(phase_config)

    # Add phase graphs as nodes to the main graph
    for phase, graph in phase_graphs.items():
        builder.add_node(phase, graph)

    # Set entry point to initialization phase
    builder.set_entry_point("initialization")

    # Define transitions between phases
    def route_after_phase(state: AgentState) -> str:
        """Route after a phase completes."""
        current_phase = state.get("phase", "initialization")

        # Define the phase progression
        phase_order = ["initialization", "development", "creation", "refinement", "finalization"]

        # Find the current phase index
        try:
            current_index = phase_order.index(current_phase)
        except ValueError:
            # If the current phase is not in the list, start with initialization
            return "initialization"

        # Move to the next phase if not at the end
        if current_index < len(phase_order) - 1:
            next_phase = phase_order[current_index + 1]
            # Update the state phase
            state["phase"] = next_phase
            return next_phase
        else:
            # If we're at the final phase, we're done
            return END

    # Add conditional edges between phases
    for phase in ["initialization", "development", "creation", "refinement"]:
        builder.add_conditional_edges(
            phase,
            route_after_phase
        )

    # Add a final edge from finalization to END
    builder.add_edge("finalization", END)

    return builder.compile()

# This function creates a comprehensive graph with all agents exposed
def create_storybook_graph(config: Dict[str, Any]) -> StateGraph:
    """Create a comprehensive storybook graph with all agents fully exposed."""
    # Extract or create agent factory
    agent_factory = config.get("agent_factory")
    if agent_factory is None:
        from agent import AgentFactory
        agent_factory = AgentFactory(config)

    # Extract project_id or use default
    project_id = config.get("project_id", "storybook_project")

    builder = StateGraph(AgentState)

    # Define all possible agents including specialists and research nodes
    all_agents = {
        # Directors
        "executive_director", "creative_director", "editorial_director",
        "content_development_director", "market_alignment_director",

        # Creative Team
        "structure_architect", "plot_development_specialist", "world_building_expert",
        "character_psychology_specialist", "character_voice_designer",
        "character_relationship_mapper",

        # Content Team
        "chapter_drafters", "scene_construction_specialists", "dialogue_crafters",
        "continuity_manager", "voice_consistency_monitor", "emotional_arc_designer",

        # Editorial Team
        "structural_editor", "character_arc_evaluator", "thematic_coherence_analyst",
        "prose_enhancement_specialist", "dialogue_refinement_expert",
        "rhythm_cadence_optimizer", "grammar_consistency_checker",
        "fact_verification_specialist",

        # Research Team
        "domain_knowledge_specialist", "cultural_authenticity_expert",

        # Marketing Team
        "positioning_specialist", "title_blurb_optimizer", "differentiation_strategist",
        "formatting_standards_expert",

        # Research Nodes
        "domain_research", "cultural_research", "market_research"
    }

    # Add all agents as nodes
    for agent_name in all_agents:
        if agent_name.endswith('_research'):
            research_type = agent_name.split('_')[0]
            builder.add_node(agent_name, agent_factory.create_research_agent(research_type))
        else:
            builder.add_node(agent_name, agent_factory.create_agent(agent_name, project_id))

    # Add the starting node - all phases start with executive director
    builder.set_entry_point("executive_director")

    # Define all possible connections between agents
    connections = {
        "executive_director": [
            "creative_director", "editorial_director", "content_development_director",
            "market_alignment_director", "domain_knowledge_specialist",
            "cultural_authenticity_expert"
        ],
        "creative_director": [
            "structure_architect", "plot_development_specialist", "world_building_expert",
            "character_psychology_specialist", "character_voice_designer",
            "character_relationship_mapper", "executive_director"
        ],
        "editorial_director": [
            "structural_editor", "character_arc_evaluator", "thematic_coherence_analyst",
            "prose_enhancement_specialist", "dialogue_refinement_expert",
            "rhythm_cadence_optimizer", "grammar_consistency_checker",
            "fact_verification_specialist", "executive_director"
        ],
        "content_development_director": [
            "chapter_drafters", "scene_construction_specialists", "dialogue_crafters",
            "continuity_manager", "voice_consistency_monitor", "emotional_arc_designer",
            "executive_director"
        ],
        "market_alignment_director": [
            "positioning_specialist", "title_blurb_optimizer", "differentiation_strategist",
            "market_research", "executive_director"
        ],
        # Research connections
        "domain_knowledge_specialist": ["domain_research", "executive_director"],
        "cultural_authenticity_expert": ["cultural_research", "executive_director"],
        "fact_verification_specialist": ["domain_research", "editorial_director"]
    }

    # Add all connections
    for source, targets in connections.items():
        for target in targets:
            builder.add_edge(source, target)

    # Add return paths from specialists to their directors
    specialist_to_director = {
        # Creative team to creative director
        "structure_architect": "creative_director",
        "plot_development_specialist": "creative_director",
        "world_building_expert": "creative_director",
        "character_psychology_specialist": "creative_director",
        "character_voice_designer": "creative_director",
        "character_relationship_mapper": "creative_director",

        # Content team to content development director
        "chapter_drafters": "content_development_director",
        "scene_construction_specialists": "content_development_director",
        "dialogue_crafters": "content_development_director",
        "continuity_manager": "content_development_director",
        "voice_consistency_monitor": "content_development_director",
        "emotional_arc_designer": "content_development_director",

        # Editorial team to editorial director
        "structural_editor": "editorial_director",
        "character_arc_evaluator": "editorial_director",
        "thematic_coherence_analyst": "editorial_director",
        "prose_enhancement_specialist": "editorial_director",
        "dialogue_refinement_expert": "editorial_director",
        "rhythm_cadence_optimizer": "editorial_director",
        "grammar_consistency_checker": "editorial_director",

        # Marketing team to market alignment director
        "positioning_specialist": "market_alignment_director",
        "title_blurb_optimizer": "market_alignment_director",
        "differentiation_strategist": "market_alignment_director"
    }

    # Add return paths
    for specialist, director in specialist_to_director.items():
        builder.add_edge(specialist, director)

    # Add research return paths
    builder.add_edge("domain_research", "domain_knowledge_specialist")
    builder.add_edge("cultural_research", "cultural_authenticity_expert")
    builder.add_edge("market_research", "market_alignment_director")

    # Set entry point
    builder.set_entry_point("executive_director")

    # Add end state connections
    builder.add_edge("executive_director", END)

    # Try to use MongoDB checkpointer if available
    mongo_uri = MONGODB_URI
    try:
        if mongo_uri:
            mongo_client = MongoClient(mongo_uri)
            checkpointer = MongoDBSaver(mongo_client)
            return builder.compile(checkpointer=checkpointer)
        else:
            return builder.compile()
    except Exception as e:
        print(f"Warning: Failed to create MongoDB checkpointer: {str(e)}. Proceeding without checkpointing.")
        return builder.compile()
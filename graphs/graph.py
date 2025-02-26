# graphs/main_graph.py
from typing import Dict, List, Any, Annotated, TypedDict, Literal
from langgraph.graph import StateGraph
import uuid
from datetime import datetime

from ..utils.state import NovelState, ProjectStatus
from ..config import NovelGenConfig
from .subgraphs.initialization import create_initialization_graph
from .subgraphs.research import create_research_graph
from .subgraphs.character_development import create_character_development_graph
from .subgraphs.drafting import create_drafting_graph
from .subgraphs.revision import create_revision_graph
from .subgraphs.reader_optimization import create_reader_optimization_graph
from .subgraphs.publication import create_publication_graph

class NovelGenerationState(TypedDict):
    """State for the novel generation workflow."""
    novel_state: NovelState
    current_phase: str
    phase_states: Dict[str, Dict[str, Any]]
    phase_outputs: Dict[str, Dict[str, Any]]
    publication_package: Dict[str, Any]

def create_main_graph(config: NovelGenConfig = None):
    """Create the main workflow graph for novel generation."""
    if config is None:
        config = NovelGenConfig()
    
    # Create subgraphs
    initialization_graph = create_initialization_graph(config)
    research_graph = create_research_graph(config)
    character_development_graph = create_character_development_graph(config)
    drafting_graph = create_drafting_graph(config)
    revision_graph = create_revision_graph(config)
    reader_optimization_graph = create_reader_optimization_graph(config)
    publication_graph = create_publication_graph(config)
    
    # Define state for the main graph
    workflow = StateGraph(NovelGenerationState)
    
    # Define nodes
    
    # Start: Initialize empty state
    def initialize_state() -> NovelGenerationState:
        """Initialize the workflow state."""
        return {
            "novel_state": None,
            "current_phase": ProjectStatus.INITIALIZED.value,
            "phase_states": {},
            "phase_outputs": {},
            "publication_package": None
        }
    
    # 1. Initialization Phase
    def run_initialization_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the initialization phase."""
        # Prepare input state for the subgraph
        init_input = {}
        if state["novel_state"] is not None:
            init_input["novel_state"] = state["novel_state"]
        
        # Retrieve previous state if available
        if "initialization" in state["phase_states"]:
            init_input.update(state["phase_states"]["initialization"])
        
        # Run the subgraph
        subgraph_result = initialization_graph.invoke(init_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "initialization": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "initialization": {
                    "concept": subgraph_result.get("refined_concept", {}),
                    "evaluation": subgraph_result.get("evaluation", {})
                }
            }
        }
    
    # 2. Research Phase
    def run_research_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the research phase."""
        # Prepare input state for the subgraph
        research_input = {"novel_state": state["novel_state"]}
        
        # Retrieve previous state if available
        if "research" in state["phase_states"]:
            research_input.update(state["phase_states"]["research"])
        
        # Run the subgraph
        subgraph_result = research_graph.invoke(research_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "research": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "research": {
                    "research_compilation": subgraph_result.get("research_compilation", {}),
                    "story_structure": subgraph_result.get("story_structure", {}),
                    "plot_points": subgraph_result.get("plot_points", []),
                    "setting_bible": subgraph_result.get("setting_bible", {})
                }
            }
        }
    
    # 3. Character Development Phase
    def run_character_development_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the character development phase."""
        # Prepare input state for the subgraph
        char_dev_input = {"novel_state": state["novel_state"]}
        
        # Retrieve previous state if available
        if "character_development" in state["phase_states"]:
            char_dev_input.update(state["phase_states"]["character_development"])
        
        # Run the subgraph
        subgraph_result = character_development_graph.invoke(char_dev_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "character_development": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "character_development": {
                    "character_profiles": subgraph_result.get("character_profiles", {}),
                    "character_arcs": subgraph_result.get("character_arcs", {}),
                    "relationship_map": subgraph_result.get("relationship_map", {}),
                    "dialogue_patterns": subgraph_result.get("dialogue_patterns", {})
                }
            }
        }
    
    # 4. Drafting Phase
    def run_drafting_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the drafting phase."""
        # Prepare input state for the subgraph
        drafting_input = {"novel_state": state["novel_state"]}
        
        # Retrieve previous state if available
        if "drafting" in state["phase_states"]:
            drafting_input.update(state["phase_states"]["drafting"])
        
        # Run the subgraph
        subgraph_result = drafting_graph.invoke(drafting_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "drafting": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "drafting": {
                    "writing_plan": subgraph_result.get("writing_plan", {}),
                    "chapter_outlines": subgraph_result.get("chapter_outlines", []),
                    "drafted_chapters": subgraph_result.get("drafted_chapters", {}),
                    "narrative_tracking": subgraph_result.get("narrative_tracking", {})
                }
            }
        }
    
    # 5. Revision Phase
    def run_revision_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the revision phase."""
        # Prepare input state for the subgraph
        revision_input = {"novel_state": state["novel_state"]}
        
        # Retrieve previous state if available
        if "revision" in state["phase_states"]:
            revision_input.update(state["phase_states"]["revision"])
        
        # Run the subgraph
        subgraph_result = revision_graph.invoke(revision_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "revision": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "revision": {
                    "structural_analysis": subgraph_result.get("structural_analysis", {}),
                    "revision_priorities": subgraph_result.get("revision_priorities", {}),
                    "tension_map": subgraph_result.get("tension_map", {})
                }
            }
        }
    
    # 6. Reader Optimization Phase
    def run_reader_optimization_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the reader optimization phase."""
        # Prepare input state for the subgraph
        optimization_input = {"novel_state": state["novel_state"]}
        
        # Retrieve previous state if available
        if "reader_optimization" in state["phase_states"]:
            optimization_input.update(state["phase_states"]["reader_optimization"])
        
        # Run the subgraph
        subgraph_result = reader_optimization_graph.invoke(optimization_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "reader_optimization": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "reader_optimization": {
                    "emotional_analysis": subgraph_result.get("emotional_analysis", {}),
                    "readability_metrics": subgraph_result.get("readability_metrics", {})
                }
            }
        }
    
    # 7. Publication Preparation Phase
    def run_publication_phase(state: NovelGenerationState) -> NovelGenerationState:
        """Run the publication preparation phase."""
        # Prepare input state for the subgraph
        publication_input = {"novel_state": state["novel_state"]}
        
        # Retrieve previous state if available
        if "publication" in state["phase_states"]:
            publication_input.update(state["phase_states"]["publication"])
        
        # Run the subgraph
        subgraph_result = publication_graph.invoke(publication_input)
        
        # Update the main state
        novel_state = subgraph_result["novel_state"]
        publication_package = subgraph_result.get("publication_package", {})
        
        return {
            **state,
            "novel_state": novel_state,
            "current_phase": novel_state.status.value,
            "phase_states": {
                **state["phase_states"],
                "publication": subgraph_result
            },
            "phase_outputs": {
                **state["phase_outputs"],
                "publication": {
                    "blurbs": subgraph_result.get("blurbs", {}),
                    "title_options": subgraph_result.get("title_options", []),
                    "comp_titles": subgraph_result.get("comp_titles", []),
                    "positioning_statement": subgraph_result.get("positioning_statement", {})
                }
            },
            "publication_package": publication_package
        }
    
    # Add nodes to graph
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("initialization_phase", run_initialization_phase)
    workflow.add_node("research_phase", run_research_phase)
    workflow.add_node("character_development_phase", run_character_development_phase)
    workflow.add_node("drafting_phase", run_drafting_phase)
    workflow.add_node("revision_phase", run_revision_phase)
    workflow.add_node("reader_optimization_phase", run_reader_optimization_phase)
    workflow.add_node("publication_phase", run_publication_phase)
    
    # Define conditional routing based on current phase
    def route_by_phase(state: NovelGenerationState) -> Literal["initialization_phase", "research_phase", "character_development_phase", "drafting_phase", "revision_phase", "reader_optimization_phase", "publication_phase", "end"]:
        """Route to the appropriate phase based on the current state."""
        current_phase = state["current_phase"]
        
        # Map phase to the corresponding node
        phase_mapping = {
            ProjectStatus.INITIALIZED.value: "initialization_phase",
            ProjectStatus.RESEARCHING.value: "research_phase",
            ProjectStatus.CHARACTER_DEVELOPMENT.value: "character_development_phase",
            ProjectStatus.DRAFTING.value: "drafting_phase",
            ProjectStatus.REVISING.value: "revision_phase",
            ProjectStatus.OPTIMIZING.value: "reader_optimization_phase",
            ProjectStatus.PREPARING_PUBLICATION.value: "publication_phase",
            ProjectStatus.COMPLETED.value: "end"
        }
        
        return phase_mapping.get(current_phase, "initialization_phase")
    
    # Define edges
    workflow.add_edge("initialize", "initialization_phase")
    workflow.add_conditional_edges(
        "initialization_phase",
        route_by_phase
    )
    workflow.add_conditional_edges(
        "research_phase",
        route_by_phase
    )
    workflow.add_conditional_edges(
        "character_development_phase",
        route_by_phase
    )
    workflow.add_conditional_edges(
        "drafting_phase",
        route_by_phase
    )
    workflow.add_conditional_edges(
        "revision_phase",
        route_by_phase
    )
    workflow.add_conditional_edges(
        "reader_optimization_phase",
        route_by_phase
    )
    workflow.add_conditional_edges(
        "publication_phase",
        route_by_phase
    )
    
    # Define the end state
    workflow.add_node("end", lambda x: x)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    return workflow

from typing import Dict, List, Callable, Optional, Any, Annotated, TypedDict, cast, Union
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langchain.schema.runnable import RunnableConfig
from langsmith.run_helpers import traceable
from langsmith import RunTree

from state import NovelSystemState
from agents import AgentFactory

# Define input, output, and overall state schemas
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"

class StoryInput(TypedDict):
    title: str
    manuscript: str
    model_provider: ModelProvider
    model_name: str  # e.g. "gpt-4", "claude-3", "mistralai/Mistral-7B-Instruct-v0.2"

class StoryOutput(TypedDict):
    title: str
    manuscript: str
    feedback: List[str]
    model_provider: ModelProvider  # Keep track of which model was used
    model_name: str

class StoryState(StoryInput, StoryOutput):
    pass

# Add tracing decorator to agent functions
@traceable(name="Executive Director Agent")
def executive_director_agent(state: StoryState) -> Dict:
    return {
        "title": state["title"], 
        "manuscript": state["manuscript"],
        "model_provider": state["model_provider"],
        "model_name": state["model_name"],
        "feedback": [],
        "agent_type": "executive_director",
        "agent_model": state["model_name"]
    }

@traceable(name="Human Feedback Manager Agent")
def human_feedback_manager_agent(state: StoryState) -> Dict:
    return {
        "feedback": ["Initial review completed"],
        "agent_type": "human_feedback_manager",
        "agent_model": state["model_name"]
    }

def create_initialization_graph(config: RunnableConfig) -> StateGraph:
    """Creates the initialization phase workflow graph."""
    workflow = StateGraph(StoryState)
    
    # Add nodes with traced agent functions and metadata
    workflow.add_node(
        "executive_director", 
        executive_director_agent,
        metadata={
            "description": "Executive oversight and initial planning",
            "agent_type": "executive_director",
            "team": "management"
        }
    )
    workflow.add_node(
        "human_feedback_manager", 
        human_feedback_manager_agent,
        metadata={
            "description": "Manages human feedback integration",
            "agent_type": "human_feedback_manager",
            "team": "feedback"
        }
    )
    workflow.add_node(
        "quality_assessment_director",
        quality_assessment_director_agent,
        metadata={
            "description": "Evaluates story quality and provides feedback",
            "agent_type": "quality_assessment_director",
            "team": "quality"
        }
    )
    workflow.add_node(
        "project_timeline_manager",
        project_timeline_manager_agent,
        metadata={
            "description": "Manages project timeline and milestones",
            "agent_type": "project_timeline_manager",
            "team": "management"
        }
    )
    workflow.add_node(
        "market_alignment_director",
        market_alignment_director_agent,
        metadata={
            "description": "Analyzes market fit and positioning",
            "agent_type": "market_alignment_director",
            "team": "market"
        }
    )
    
    # Add edges without metadata
    workflow.add_edge(START, "executive_director")
    workflow.add_edge("executive_director", "human_feedback_manager")
    workflow.add_edge("human_feedback_manager", "quality_assessment_director")
    workflow.add_edge("quality_assessment_director", "project_timeline_manager")
    workflow.add_edge("project_timeline_manager", "market_alignment_director")
    workflow.add_edge("market_alignment_director", END)
    
    return workflow.compile()

def create_development_graph(config: RunnableConfig) -> StateGraph:
    """Creates the development phase workflow graph."""
    workflow = StateGraph(StoryState)
    
    workflow.add_node(
        "plot_developer",
        plot_development_agent,
        metadata={
            "description": "Develops plot structure and story arcs",
            "agent_type": "plot_developer",
            "team": "creative"
        }
    )
    
    workflow.add_node(
        "character_developer",
        character_development_agent,
        metadata={
            "description": "Develops character profiles and arcs",
            "agent_type": "character_developer",
            "team": "creative"
        }
    )
    
    workflow.add_edge(START, "plot_developer")
    workflow.add_edge("plot_developer", "character_developer")
    workflow.add_edge("character_developer", END)
    
    return workflow.compile()

def create_creation_graph(config: RunnableConfig) -> StateGraph:
    """Creates the creation phase workflow graph."""
    workflow = StateGraph(StoryState)
    
    workflow.add_node(
        "content_creator",
        content_creator_agent,
        metadata={
            "description": "Generates and manages story content",
            "agent_type": "content_creator",
            "team": "writing"
        }
    )
    
    workflow.add_node(
        "draft_reviewer",
        draft_reviewer_agent,
        metadata={
            "description": "Reviews and assesses content quality",
            "agent_type": "draft_reviewer",
            "team": "quality"
        }
    )
    
    workflow.add_edge(START, "content_creator")
    workflow.add_edge("content_creator", "draft_reviewer")
    workflow.add_edge("draft_reviewer", END)
    
    return workflow.compile()

def create_refinement_graph(config: RunnableConfig) -> StateGraph:
    """Creates the refinement phase workflow graph."""
    workflow = StateGraph(StoryState)
    
    workflow.add_node(
        "editor",
        editor_agent,
        metadata={
            "description": "Edits and refines story content",
            "agent_type": "editor",
            "team": "editing"
        }
    )
    
    workflow.add_node(
        "proofreader",
        proofreader_agent,
        metadata={
            "description": "Performs final proofreading",
            "agent_type": "proofreader",
            "team": "editing"
        }
    )
    
    workflow.add_edge(START, "editor")
    workflow.add_edge("editor", "proofreader")
    workflow.add_edge("proofreader", END)
    
    return workflow.compile()

def create_finalization_graph(config: RunnableConfig) -> StateGraph:
    """Creates the finalization phase workflow graph."""
    workflow = StateGraph(StoryState)
    
    workflow.add_node(
        "finalizer",
        finalizer_agent,
        metadata={
            "description": "Prepares story for publication",
            "agent_type": "finalizer",
            "team": "market"
        }
    )
    
    workflow.add_node(
        "quality_checker",
        quality_checker_agent,
        metadata={
            "description": "Performs final quality assurance",
            "agent_type": "quality_checker",
            "team": "quality"
        }
    )
    
    workflow.add_edge(START, "finalizer")
    workflow.add_edge("finalizer", "quality_checker")
    workflow.add_edge("quality_checker", END)
    
    return workflow.compile()

def get_phase_workflow(phase: str, project_id: str, agent_factory: AgentFactory) -> StateGraph:
    """Get the workflow graph for a specific phase."""
    workflow_map = {
        "initialization": create_initialization_graph,
        "development": create_development_graph,
        "creation": create_creation_graph,
        "refinement": create_refinement_graph,
        "finalization": create_finalization_graph,
    }
    
    if phase not in workflow_map:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Configure tracing with RunTree
    config = RunnableConfig(
        run_name=f"{phase}_phase",
        callbacks=None,  # LangSmith tracing is handled via env vars
        tags=[f"project_{project_id}", f"phase_{phase}"],
        metadata={
            "project_id": project_id,
            "phase": phase,
            "teams": ["management", "feedback", "quality", "timeline", "market"]
        }
    )
    
    return workflow_map[phase](config)

# Add this after your existing graph functions

def create_novel_writing_workflow(config: RunnableConfig) -> StateGraph:
    """Creates a complete novel writing workflow combining all phases."""
    workflow = StateGraph(StoryState)
    
    # Add each phase as a node with detailed metadata for LangSmith visualization
    phases = {
        "initialization": {
            "description": "Initial project setup and planning",
            "team": "management"
        },
        "development": {
            "description": "Story and character development",
            "team": "creative"
        },
        "creation": {
            "description": "Content creation and drafting",
            "team": "writing"
        },
        "refinement": {
            "description": "Editing and refinement",
            "team": "editing"
        },
        "finalization": {
            "description": "Final review and polishing",
            "team": "quality"
        }
    }
    
    # Add each phase with its metadata
    for phase_name, phase_info in phases.items():
        workflow.add_node(
            f"{phase_name}_phase",
            get_phase_workflow(
                phase_name, 
                config.metadata.get("project_id", "default"),
                config.metadata.get("agent_factory")
            ),
            metadata={
                "description": phase_info["description"],
                "team": phase_info["team"],
                "phase": phase_name,
                "project_id": config.metadata.get("project_id", "default")
            }
        )
    
    # Connect phases sequentially
    workflow.add_edge(START, "initialization_phase")
    workflow.add_edge("initialization_phase", "development_phase")
    workflow.add_edge("development_phase", "creation_phase")
    workflow.add_edge("creation_phase", "refinement_phase")
    workflow.add_edge("refinement_phase", "finalization_phase")
    workflow.add_edge("finalization_phase", END)
    
    return workflow.compile()

def create_storybook_workflow(config: RunnableConfig) -> StateGraph:
    """Creates a complete storybook workflow combining all phases."""
    workflow = StateGraph(StoryState)
    
    # Define phases with their teams and descriptions
    phases = {
        "initialization": {
            "description": "Project setup and planning",
            "team": "management",
            "agents": ["executive_director", "human_feedback_manager"]
        },
        "development": {
            "description": "Story and character development",
            "team": "creative",
            "agents": ["plot_developer", "character_developer"]
        },
        "creation": {
            "description": "Content creation and drafting",
            "team": "writing",
            "agents": ["content_creator", "draft_reviewer"]
        },
        "refinement": {
            "description": "Editing and refinement",
            "team": "editing",
            "agents": ["editor", "proofreader"]
        },
        "finalization": {
            "description": "Final review and publication prep",
            "team": "quality",
            "agents": ["finalizer", "quality_checker"]
        }
    }
    
    # Add each phase with its metadata and agents
    for phase_name, phase_info in phases.items():
        workflow.add_node(
            f"{phase_name}_phase",
            get_phase_workflow(
                phase_name, 
                config.metadata.get("project_id", "default"),
                config.metadata.get("agent_factory")
            ),
            metadata={
                "description": phase_info["description"],
                "team": phase_info["team"],
                "phase": phase_name,
                "agents": phase_info["agents"],
                "project_id": config.metadata.get("project_id", "default")
            }
        )
    
    # Create sequential flow through phases
    workflow.add_edge(START, "initialization_phase")
    workflow.add_edge("initialization_phase", "development_phase")
    workflow.add_edge("development_phase", "creation_phase")
    workflow.add_edge("creation_phase", "refinement_phase")
    workflow.add_edge("refinement_phase", "finalization_phase")
    workflow.add_edge("finalization_phase", END)
    
    return workflow.compile()

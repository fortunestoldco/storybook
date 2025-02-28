from __future__ import annotations

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import threading
import functools

from storybook.config import get_llm
from storybook.agents import (
    CharacterDeveloper, DialogueEnhancer, WorldBuilder, SubplotWeaver,
    StoryArcAnalyst, ContinuityEditor, LanguagePolisher, QualityReviewer,
    MarketResearcher, ContentAnalyzer
)

# Schema Definitions
class InputState(TypedDict):
    """Input state schema definition."""
    manuscript_text: str

class OutputState(TypedDict):
    """Output state schema definition."""
    market_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    characters: List[Dict[str, Any]]
    dialogue: List[Dict[str, Any]]
    world_building: Dict[str, Any]
    subplots: List[Dict[str, Any]]
    story_arc: Dict[str, Any]
    language: Dict[str, Any]
    quality_review: Dict[str, Any]

class GraphState(InputState, OutputState):
    """Overall graph state schema."""
    state: str

# Thread lock for initialization
_init_lock = threading.Lock()

# Cache for initialization status
_is_initialized = False

# The graph instance
graph = None

# Lazily initialized agent instances
_agents = None

def initialize_once(func):
    """Decorator to ensure a function runs only once."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _is_initialized
        with _init_lock:
            if not _is_initialized:
                result = func(*args, **kwargs)
                _is_initialized = True
                return result
        return None
    return wrapper

@initialize_once
def initialize_agents(config):
    """Initialize agents only once."""
    global _agents
    
    # Get LLM configuration
    llm_config = get_llm(config.get("metadata", {}))
    
    # Initialize agents
    _agents = {
        "market_researcher": MarketResearcher(llm_config),
        "content_analyzer": ContentAnalyzer(llm_config),
        "character_developer": CharacterDeveloper(llm_config),
        "dialogue_enhancer": DialogueEnhancer(llm_config),
        "world_builder": WorldBuilder(llm_config),
        "subplot_weaver": SubplotWeaver(llm_config),
        "story_arc_analyst": StoryArcAnalyst(llm_config),
        "language_polisher": LanguagePolisher(llm_config),
        "quality_reviewer": QualityReviewer(llm_config)
    }
    
    return _agents

def get_agents(config=None):
    """Get agents instance, initializing if necessary."""
    global _agents
    if _agents is None:
        initialize_agents(config or {})
    return _agents

def build_storybook_nodes(agent_dict):
    """Build the node functions using the provided agent dictionary."""
    
    async def market_research(state: GraphState):
        """Analyze market positioning and trends."""
        results = await agent_dict["market_researcher"].process_manuscript(state)
        return {
            "market_analysis": results,
            "state": "market_analyzed"
        }

    async def content_analysis(state: GraphState):
        """Analyze content and themes."""
        results = await agent_dict["content_analyzer"].process_manuscript(state)
        return {
            "content_analysis": results,
            "state": "content_analyzed"
        }

    async def creative_development(state: GraphState):
        """Develop creative elements."""
        character_results = await agent_dict["character_developer"].process_manuscript(state)
        dialogue_results = await agent_dict["dialogue_enhancer"].process_manuscript(
            state,
            characters=character_results
        )
        world_results = await agent_dict["world_builder"].process_manuscript(state)
        subplot_results = await agent_dict["subplot_weaver"].process_manuscript(
            state,
            characters=character_results
        )
        
        return {
            "characters": character_results,
            "dialogue": dialogue_results,
            "world_building": world_results,
            "subplots": subplot_results,
            "state": "creative_complete"
        }

    async def story_development(state: GraphState):
        """Develop story structure and language."""
        arc_results = await agent_dict["story_arc_analyst"].process_manuscript(
            state,
            characters=state["characters"],
            subplots=state["subplots"]
        )
        language_results = await agent_dict["language_polisher"].process_manuscript(state)
        
        return {
            "story_arc": arc_results,
            "language": language_results,
            "state": "story_complete"
        }

    async def quality_review(state: GraphState):
        """Review and validate story elements."""
        review_results = await agent_dict["quality_reviewer"].process_manuscript(
            state,
            context=state
        )
        return {
            "quality_review": review_results,
            "state": "complete"
        }
        
    return {
        "market_research": market_research,
        "content_analysis": content_analysis,
        "creative_development": creative_development,
        "story_development": story_development,
        "quality_review": quality_review
    }

@initialize_once
def create_graph():
    """Create the graph (guaranteed to run only once)."""
    global graph
    
    # Initialize workflow with state schemas and initial state
    initial_state = {
        "manuscript_text": "",  # Will be set by input
        "market_analysis": {},
        "content_analysis": {},
        "characters": [],
        "dialogue": [],
        "world_building": {},
        "subplots": [],
        "story_arc": {},
        "language": {},
        "quality_review": {},
        "state": "start"
    }
    
    # Create the workflow - use the correct parameter names for your version
    workflow = StateGraph(
        GraphState,  # Use the class as the schema
        initial_state,
        input=InputState,    # Using input instead of input_type
        output=OutputState   # Using output instead of output_type
    )
    
    # Get agent functions
    agents = get_agents()
    nodes = build_storybook_nodes(agents)
    
    # Add nodes to workflow
    workflow.add_node("analyze_market", nodes["market_research"])
    workflow.add_node("analyze_content", nodes["content_analysis"])
    workflow.add_node("develop_creative", nodes["creative_development"])
    workflow.add_node("develop_story", nodes["story_development"])
    workflow.add_node("review_quality", nodes["quality_review"])
    
    # Configure workflow routing
    workflow.add_edge(START, "analyze_market")
    workflow.add_edge("analyze_market", "analyze_content")
    workflow.add_edge("analyze_content", "develop_creative")
    workflow.add_edge("develop_creative", "develop_story")
    workflow.add_edge("develop_story", "review_quality")
    workflow.add_edge("review_quality", END)
    
    # Compile the graph
    graph = workflow.compile()
    return graph

# Initialize the graph at module level - this is crucial for LangGraph API
graph = create_graph()

# This is what gets called from outside to get the graph
def build_storybook(config: RunnableConfig = None) -> StateGraph:
    """Get the storybook graph instance."""
    global graph
    # Graph was already created at module level initialization
    return graph

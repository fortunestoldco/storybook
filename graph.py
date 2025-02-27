from typing import Dict, List, Any, Annotated, TypedDict
import os
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongo import MongoCheckpointManager
from agents import (
    character_development_agent, dialogue_enhancer_agent, worldbuilding_agent,
    subplot_weaver_agent, story_arc_analyst_agent, continuity_editor_agent,
    language_style_polisher_agent, quality_assurance_agent
)
from models import Novel, AgentInput, AgentOutput
from db_utils import MongoDBManager


# Define the graph state
class NovelState(TypedDict):
    novel: Novel
    current_agent: str
    feedback: str
    error: str
    completed_agents: List[str]
    iterations: int
    max_iterations: int
    

# Initialize MongoDB for checkpointing
mongo_uri = os.getenv("MONGODB_URI")
mongo_client = MongoDBManager()

checkpoint_manager = MongoCheckpointManager(
    mongo_uri=mongo_uri,
    db_name=os.getenv("MONGODB_DATABASE_NAME"),
    collection_name="graph_checkpoints"
)

# Define the Storybook graph
def build_novel_graph():
    """Build the Storybook graph with all agents"""
    # Create the graph
    graph = StateGraph(NovelState)
    
    # Define nodes for each agent
    graph.add_node("character_development", character_development_node)
    graph.add_node("dialogue_enhancement", dialogue_enhancement_node)
    graph.add_node("world_building", world_building_node)
    graph.add_node("subplot_weaving", subplot_weaving_node)
    graph.add_node("story_arc_analysis", story_arc_analysis_node)
    graph.add_node("continuity_editing", continuity_editing_node)
    graph.add_node("language_style", language_style_node)
    graph.add_node("quality_assurance", quality_assurance_node)
    
    # Define the edges between nodes
    graph.add_edge("character_development", "dialogue_enhancement")
    graph.add_edge("dialogue_enhancement", "world_building")
    graph.add_edge("world_building", "subplot_weaving")
    graph.add_edge("subplot_weaving", "story_arc_analysis")
    graph.add_edge("story_arc_analysis", "continuity_editing")
    graph.add_edge("continuity_editing", "language_style")
    graph.add_edge("language_style", "quality_assurance")
    
    # Define conditional edge from quality assurance
    graph.add_conditional_edges(
        "quality_assurance",
        decide_next_agent,
        {
            "character_development": "character_development",
            "dialogue_enhancement": "dialogue_enhancement",
            "world_building": "world_building",
            "subplot_weaving": "subplot_weaving",
            "story_arc_analysis": "story_arc_analysis",
            "continuity_editing": "continuity_editing",
            "language_style": "language_style",
            "END": END
        }
    )
    
    # Set the entry point
    graph.set_entry_point("character_development")
    
    # Compile the graph
    return graph.compile()


# Define the agent nodes
def character_development_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for character development agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "character_development"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Develop fully detailed characters")
        result = character_development_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "character_development"})],
            {"novel_id": novel.title, "agent": "character_development"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "character_development" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("character_development")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in character_development_node: {str(e)}"
        return state

def dialogue_enhancement_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for dialogue enhancement agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "dialogue_enhancement"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Enhance dialogue to be more authentic and character-driven")
        result = dialogue_enhancer_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "dialogue_enhancement"})],
            {"novel_id": novel.title, "agent": "dialogue_enhancement"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "dialogue_enhancement" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("dialogue_enhancement")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in dialogue_enhancement_node: {str(e)}"
        return state

def world_building_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for world building agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "world_building"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Develop detailed settings and world elements")
        result = worldbuilding_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "world_building"})],
            {"novel_id": novel.title, "agent": "world_building"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "world_building" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("world_building")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in world_building_node: {str(e)}"
        return state

def subplot_weaving_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for subplot weaving agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "subplot_weaving"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Create and integrate compelling subplots")
        result = subplot_weaver_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "subplot_weaving"})],
            {"novel_id": novel.title, "agent": "subplot_weaving"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "subplot_weaving" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("subplot_weaving")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in subplot_weaving_node: {str(e)}"
        return state

def story_arc_analysis_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for story arc analysis agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "story_arc_analysis"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Analyze and refine plot structure and arcs")
        result = story_arc_analyst_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "story_arc_analysis"})],
            {"novel_id": novel.title, "agent": "story_arc_analysis"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "story_arc_analysis" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("story_arc_analysis")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in story_arc_analysis_node: {str(e)}"
        return state

def continuity_editing_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for continuity editing agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "continuity_editing"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Check and fix continuity issues")
        result = continuity_editor_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "continuity_editing"})],
            {"novel_id": novel.title, "agent": "continuity_editing"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "continuity_editing" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("continuity_editing")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in continuity_editing_node: {str(e)}"
        return state

def language_style_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for language and style polishing agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "language_style"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Polish language and writing style")
        result = language_style_polisher_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "language_style"})],
            {"novel_id": novel.title, "agent": "language_style"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "language_style" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("language_style")
        updated_state["feedback"] = "\n".join(result.notes.split("\n")[:5])
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in language_style_node: {str(e)}"
        return state

def quality_assurance_node(state: NovelState, config: RunnableConfig = None) -> Dict:
    """Node for quality assurance agent"""
    try:
        novel = state["novel"]
        
        # Update state
        updated_state = state.copy()
        updated_state["current_agent"] = "quality_assurance"
        
        # Run the agent
        input_data = AgentInput(novel=novel, instructions="Perform a comprehensive quality review")
        result = quality_assurance_agent(input_data, config)
        
        # Store the enhanced novel in the database
        mongo_client.store_document_vectors(
            [Document(page_content=result.novel.manuscript, metadata={"stage": "quality_assurance"})],
            {"novel_id": novel.title, "agent": "quality_assurance"}
        )
        
        # Update the state with the result
        updated_state["novel"] = result.novel
        if "quality_assurance" not in updated_state["completed_agents"]:
            updated_state["completed_agents"].append("quality_assurance")
        updated_state["feedback"] = result.notes
        updated_state["iterations"] = state.get("iterations", 0) + 1
        
        return updated_state
    except Exception as e:
        state["error"] = f"Error in quality_assurance_node: {str(e)}"
        return state

def decide_next_agent(state: NovelState) -> str:
    """Decide the next agent based on feedback and iterations"""
    # If we've reached the max iterations, end the process
    if state.get("iterations", 0) >= state.get("max_iterations", 3):
        return "END"
    
    # Parse the quality assurance feedback to determine areas that need improvement
    feedback = state.get("feedback", "").lower()
    
    # Decision logic based on feedback keywords
    if "character" in feedback and "development" in feedback:
        return "character_development"
    elif "dialogue" in feedback:
        return "dialogue_enhancement"
    elif "world" in feedback or "setting" in feedback:
        return "world_building"
    elif "subplot" in feedback:
        return "subplot_weaving"
    elif "plot" in feedback or "story arc" in feedback:
        return "story_arc_analysis"
    elif "continuity" in feedback or "consistency" in feedback:
        return "continuity_editing"
    elif "language" in feedback or "style" in feedback:
        return "language_style"
    elif "excellent" in feedback or "ready" in feedback:
        return "END"
    else:
        # If no specific issues are identified but we haven't reached max iterations
        return "character_development"  # Default to starting over

# Initialize the graph
storybook_graph = build_novel_graph()

# graphs/subgraphs/initialization.py
from typing import Dict, List, Any, Annotated, TypedDict
from langgraph.graph import StateGraph
import uuid
from datetime import datetime

from ...agents.project_management import ProjectLeadAgent, MarketResearchAgent, ConceptDevelopmentAgent
from ...utils.state import NovelState, ProjectStatus, ProjectConcept
from ...config import NovelGenConfig

class InitializationState(TypedDict):
    """State for the initialization subgraph."""
    novel_state: NovelState
    market_analysis: Dict[str, Any]
    concepts: List[ProjectConcept]
    selected_concept: ProjectConcept
    refined_concept: ProjectConcept
    evaluation: Dict[str, Any]

def create_initialization_graph(config: NovelGenConfig):
    """Create the initialization phase subgraph."""
    # Initialize agents
    project_lead = ProjectLeadAgent(config)
    market_research = MarketResearchAgent(config)
    concept_dev = ConceptDevelopmentAgent(config)
    
    # Define state
    workflow = StateGraph(InitializationState)
    
    # Define nodes
    
    # 1. Initialize project
    def initialize_project(state: InitializationState) -> InitializationState:
        if "novel_state" not in state:
            # Create a new novel state with a default name
            project_name = f"New Novel Project - {datetime.now().strftime('%Y-%m-%d')}"
            novel_state = project_lead.initialize_project(project_name)
            return {"novel_state": novel_state}
        return state
    
    # 2. Analyze market trends
    def analyze_market(state: InitializationState) -> InitializationState:
        novel_state = state["novel_state"]
        # Default to fiction if genre not specified
        genres = [novel_state.genre] if novel_state.genre else ["Fiction"]
        # Add subgenres if available
        if hasattr(novel_state, "subgenres") and novel_state.subgenres:
            genres.extend(novel_state.subgenres)
        
        market_analysis = market_research.analyze_market_trends(genres)
        return {**state, "market_analysis": market_analysis}
    
    # 3. Generate novel concepts
    def generate_concepts(state: InitializationState) -> InitializationState:
        market_analysis = state["market_analysis"]["market_analysis"]
        concepts = market_research.generate_novel_concepts(market_analysis, count=3)
        return {**state, "concepts": concepts}
    
    # 4. Select best concept
    def select_concept(state: InitializationState) -> InitializationState:
        # In a real implementation, this could involve human input or more sophisticated selection
        concepts = state["concepts"]
        if not concepts:
            # Create a default concept if none were generated
            selected_concept = ProjectConcept(
                title="Untitled Novel",
                genre="Fiction",
                subgenres=["Literary"],
                target_audience="Adult",
                premise="A character faces challenges and grows through the experience.",
                market_potential=0.5,
                uniqueness_factor=0.5,
                comparable_titles=["Similar Book 1", "Similar Book 2"],
                themes=["Growth", "Change"],
                estimated_word_count=80000
            )
        else:
            # Select concept with highest combined market potential and uniqueness
            selected_concept = max(concepts, 
                                 key=lambda c: c.market_potential * 0.7 + c.uniqueness_factor * 0.3)
        
        return {**state, "selected_concept": selected_concept}
    
    # 5. Refine concept
    def refine_concept(state: InitializationState) -> InitializationState:
        selected_concept = state["selected_concept"]
        refined_concept = concept_dev.refine_concept(selected_concept)
        return {**state, "refined_concept": refined_concept}
    
    # 6. Evaluate concept
    def evaluate_concept(state: InitializationState) -> InitializationState:
        refined_concept = state["refined_concept"]
        novel_state = state["novel_state"]
        evaluation = project_lead.evaluate_concept(refined_concept, novel_state)
        
        # Update the novel state with the evaluation results
        novel_state = evaluation["state"]
        
        return {
            **state, 
            "evaluation": evaluation,
            "novel_state": novel_state
        }
    
    # 7. Set next phase
    def set_next_phase(state: InitializationState) -> InitializationState:
        novel_state = state["novel_state"]
        novel_state = project_lead.set_project_phase(novel_state, ProjectStatus.RESEARCHING)
        return {**state, "novel_state": novel_state}
    
    # Add nodes to graph
    workflow.add_node("initialize_project", initialize_project)
    workflow.add_node("analyze_market", analyze_market)
    workflow.add_node("generate_concepts", generate_concepts)
    workflow.add_node("select_concept", select_concept)
    workflow.add_node("refine_concept", refine_concept)
    workflow.add_node("evaluate_concept", evaluate_concept)
    workflow.add_node("set_next_phase", set_next_phase)
    
    # Define edges
    workflow.add_edge("initialize_project", "analyze_market")
    workflow.add_edge("analyze_market", "generate_concepts")
    workflow.add_edge("generate_concepts", "select_concept")
    workflow.add_edge("select_concept", "refine_concept")
    workflow.add_edge("refine_concept", "evaluate_concept")
    workflow.add_edge("evaluate_concept", "set_next_phase")
    
    # Set entry and exit points
    workflow.set_entry_point("initialize_project")
    
    return workflow

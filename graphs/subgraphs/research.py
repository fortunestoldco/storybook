# graphs/subgraphs/research.py
from typing import Dict, List, Any, Annotated, TypedDict
from langgraph.graph import StateGraph

from ...agents.research import ResearchSupervisorAgent, HistoricalResearchAgent, TechnicalDomainAgent, CulturalAuthenticityAgent
from ...agents.story_architecture import StructureSpecialistAgent, PlotDevelopmentAgent, WorldBuildingAgent
from ...agents.project_management import ProjectLeadAgent
from ...utils.state import NovelState, ProjectStatus, ResearchTopic, ResearchItem, StoryStructure
from ...config import NovelGenConfig

class ResearchState(TypedDict):
    """State for the research subgraph."""
    novel_state: NovelState
    research_topics: List[ResearchTopic]
    research_items: Dict[str, ResearchItem]
    research_compilation: Dict[str, Any]
    structure_recommendations: Dict[str, Any]
    selected_structure: str
    story_structure: StoryStructure
    plot_points: List[Any]
    world_building: Dict[str, Any]
    setting_bible: Dict[str, Any]

def create_research_graph(config: NovelGenConfig):
    """Create the research phase subgraph."""
    # Initialize agents
    research_supervisor = ResearchSupervisorAgent(config)
    historical_research = HistoricalResearchAgent(config)
    technical_domain = TechnicalDomainAgent(config)
    cultural_authenticity = CulturalAuthenticityAgent(config)
    structure_specialist = StructureSpecialistAgent(config)
    plot_development = PlotDevelopmentAgent(config)
    world_building = WorldBuildingAgent(config)
    project_lead = ProjectLeadAgent(config)
    
    # Define state
    workflow = StateGraph(ResearchState)
    
    # Define nodes
    
    # 1. Identify research needs
    def identify_research_needs(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        research_topics = research_supervisor.identify_research_needs(novel_state)
        return {**state, "research_topics": research_topics}
    
    # 2. Conduct research
    def conduct_research(state: ResearchState) -> ResearchState:
        research_topics = state["research_topics"]
        research_items = {}
        
        for topic in research_topics:
            if "historical" in topic.topic.lower() or "period" in topic.topic.lower():
                # Historical research
                period = topic.topic.replace("Historical Period: ", "").replace("History of ", "")
                aspects = [topic.description]
                research_item = historical_research.research_historical_period(period, aspects)
            elif "technical" in topic.topic.lower() or "domain" in topic.topic.lower():
                # Technical domain research
                domain = topic.topic.replace("Technical Domain: ", "").replace("Technology of ", "")
                specific_topics = [topic.description]
                research_item = technical_domain.research_domain(domain, specific_topics)
            elif "cultural" in topic.topic.lower() or "culture" in topic.topic.lower():
                # Cultural research
                culture = topic.topic.replace("Cultural Elements: ", "").replace("Culture of ", "")
                elements = [topic.description]
                research_item = cultural_authenticity.research_cultural_elements(culture, elements)
            else:
                # General research - use whichever agent seems most appropriate
                # For simplicity, we'll use technical domain agent as a fallback
                research_item = technical_domain.research_domain(topic.topic, [topic.description])
            
            # Add the research item to our collection
            research_items[topic.topic] = research_item
        
        return {**state, "research_items": research_items}
    
    # 3. Evaluate research
    def evaluate_research(state: ResearchState) -> ResearchState:
        research_items = state.get("research_items", {})
        novel_state = state["novel_state"]
        
        # Evaluate each research item
        for topic, item in research_items.items():
            evaluation = research_supervisor.evaluate_research(item)
            
            # Update the research item with the evaluation
            item.verified = evaluation["verified"]
            item.relevance_score = evaluation["quality_score"]
            
            # Add to novel state
            novel_state.research[topic] = item
        
        return {**state, "novel_state": novel_state}
    
    # 4. Compile research
    def compile_research(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        compilation = research_supervisor.compile_research(novel_state)
        
        return {**state, "research_compilation": compilation}
    
    # 5. Recommend story structure
    def recommend_structure(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        recommendations = structure_specialist.recommend_structure(novel_state)
        
        return {**state, "structure_recommendations": recommendations}
    
    # 6. Select story structure
    def select_structure(state: ResearchState) -> ResearchState:
        recommendations = state["structure_recommendations"]
        selected_structure = recommendations["recommended_structure"]
        
        return {**state, "selected_structure": selected_structure}
    
    # 7. Design story structure
    def design_structure(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        selected_structure = state["selected_structure"]
        story_structure = structure_specialist.design_structure(novel_state, selected_structure)
        
        return {**state, "story_structure": story_structure}
    
    # 8. Develop plot points
    def develop_plot_points(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        story_structure = state["story_structure"]
        plot_points = plot_development.develop_plot_points(novel_state, story_structure)
        
        # Add plot points to novel state
        novel_state.plot_points = plot_points
        
        return {
            **state, 
            "plot_points": plot_points,
            "novel_state": novel_state
        }
    
    # 9. Develop world building
    def develop_world_building(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        world_building_result = world_building.develop_setting(novel_state)
        
        return {**state, "world_building": world_building_result}
    
    # 10. Create setting bible
    def create_setting_bible(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        world_building_result = state["world_building"]
        
        setting_bible = world_building.create_setting_bible(novel_state, world_building_result)
        
        # Add settings to novel state
        setting_data = setting_bible["sections"]
        novel_state.settings = {
            "primary_locations": {"description": setting_data.get("expanded_locations", "")},
            "world_systems": {"description": setting_data.get("systems_in_depth", "")},
            "timeline": {"description": setting_data.get("timeline", "")},
            "social_structure": {"description": setting_data.get("social_structure", "")},
            "natural_elements": {"description": setting_data.get("flora_fauna_resources", "")}
        }
        
        return {
            **state, 
            "setting_bible": setting_bible,
            "novel_state": novel_state
        }
    
    # 11. Set next phase
    def set_next_phase(state: ResearchState) -> ResearchState:
        novel_state = state["novel_state"]
        novel_state = project_lead.set_project_phase(novel_state, ProjectStatus.CHARACTER_DEVELOPMENT)
        
        # Update completion metrics
        novel_state.phase_progress = 1.0  # This phase is complete
        
        return {**state, "novel_state": novel_state}
    
    # Add nodes to graph
    workflow.add_node("identify_research_needs", identify_research_needs)
    workflow.add_node("conduct_research", conduct_research)
    workflow.add_node("evaluate_research", evaluate_research)
    workflow.add_node("compile_research", compile_research)
    workflow.add_node("recommend_structure", recommend_structure)
    workflow.add_node("select_structure", select_structure)
    workflow.add_node("design_structure", design_structure)
    workflow.add_node("develop_plot_points", develop_plot_points)
    workflow.add_node("develop_world_building", develop_world_building)
    workflow.add_node("create_setting_bible", create_setting_bible)
    workflow.add_node("set_next_phase", set_next_phase)
    
    # Define edges
    workflow.add_edge("identify_research_needs", "conduct_research")
    workflow.add_edge("conduct_research", "evaluate_research")
    workflow.add_edge("evaluate_research", "compile_research")
    workflow.add_edge("compile_research", "recommend_structure")
    workflow.add_edge("recommend_structure", "select_structure")
    workflow.add_edge("select_structure", "design_structure")
    workflow.add_edge("design_structure", "develop_plot_points")
    workflow.add_edge("develop_plot_points", "develop_world_building")
    workflow.add_edge("develop_world_building", "create_setting_bible")
    workflow.add_edge("create_setting_bible", "set_next_phase")
    
    # Set entry and exit points
    workflow.set_entry_point("identify_research_needs")
    
    return workflow

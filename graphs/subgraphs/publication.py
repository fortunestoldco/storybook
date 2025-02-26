# graphs/subgraphs/publication.py
from typing import Dict, List, Any, Annotated, TypedDict
from langgraph.graph import StateGraph

from ...agents.publication import BlurbGeneratorAgent, BookTitleOptimizerAgent, ComparableTitleAnalystAgent, TagAndCategorySpecialistAgent
from ...agents.project_management import ProjectLeadAgent
from ...utils.state import NovelState, ProjectStatus
from ...config import NovelGenConfig

class PublicationState(TypedDict):
    """State for the publication subgraph."""
    novel_state: NovelState
    blurbs: Dict[str, str]
    blurb_analyses: Dict[str, Dict[str, Any]]
    title_options: List[Dict[str, Any]]
    title_analyses: Dict[str, Dict[str, Any]]
    comp_titles: List[Dict[str, Any]]
    positioning_statement: Dict[str, str]
    metadata: Dict[str, Any]
    category_optimization: Dict[str, Any]
    publication_package: Dict[str, Any]

def create_publication_graph(config: NovelGenConfig):
    """Create the publication preparation phase subgraph."""
    # Initialize agents
    blurb_generator = BlurbGeneratorAgent(config)
    title_optimizer = BookTitleOptimizerAgent(config)
    comp_title_analyst = ComparableTitleAnalystAgent(config)
    tag_specialist = TagAndCategorySpecialistAgent(config)
    project_lead = ProjectLeadAgent(config)
    
    # Define state
    workflow = StateGraph(PublicationState)
    
    # Define nodes
    
    # 1. Generate blurbs
    def generate_blurbs(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        blurbs = blurb_generator.generate_blurbs(novel_state)
        
        return {**state, "blurbs": blurbs}
    
    # 2. Analyze blurbs
    def analyze_blurbs(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        blurbs = state["blurbs"]
        blurb_analyses = {}
        
        for style, blurb in blurbs.items():
            analysis = blurb_generator.analyze_blurb_effectiveness(
                blurb=blurb,
                genre=novel_state.genre,
                target_audience=novel_state.target_audience
            )
            blurb_analyses[style] = analysis
        
        return {**state, "blurb_analyses": blurb_analyses}
    
    # 3. Generate title options
    def generate_title_options(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        title_options = title_optimizer.generate_title_options(novel_state)
        
        return {**state, "title_options": title_options}
    
    # 4. Analyze titles
    def analyze_titles(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        title_options = state["title_options"]
        title_analyses = {}
        
        # Analyze the top 3 titles by marketability score
        top_titles = sorted(title_options, key=lambda t: t.get("marketability", 0), reverse=True)[:3]
        
        for title_data in top_titles:
            title = title_data["title"]
            analysis = title_optimizer.test_title_effectiveness(
                title=title,
                genre=novel_state.genre,
                target_audience=novel_state.target_audience
            )
            title_analyses[title] = analysis
        
        return {**state, "title_analyses": title_analyses}
    
    # 5. Identify comp titles
    def identify_comp_titles(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        comp_titles = comp_title_analyst.identify_comp_titles(novel_state)
        
        return {**state, "comp_titles": comp_titles}
    
    # 6. Create positioning statement
    def create_positioning_statement(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        comp_titles = state["comp_titles"]
        
        positioning_statement = comp_title_analyst.create_positioning_statement(
            state=novel_state,
            comp_titles=comp_titles
        )
        
        return {**state, "positioning_statement": positioning_statement}
    
    # 7. Generate metadata
    def generate_metadata(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        metadata = tag_specialist.generate_metadata(novel_state)
        
        return {**state, "metadata": metadata}
    
    # 8. Optimize categories
    def optimize_categories(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        metadata = state["metadata"]
        
        # Optimize for Amazon as the default platform
        category_optimization = tag_specialist.optimize_categories(metadata, "Amazon")
        
        return {**state, "category_optimization": category_optimization}
    
    # 9. Prepare publication package
    def prepare_publication_package(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        blurbs = state["blurbs"]
        blurb_analyses = state["blurb_analyses"]
        title_options = state["title_options"]
        title_analyses = state["title_analyses"]
        comp_titles = state["comp_titles"]
        positioning_statement = state["positioning_statement"]
        metadata = state["metadata"]
        category_optimization = state["category_optimization"]
        
        # Determine the best blurb based on analyses
        best_blurb_style = max(blurb_analyses.items(), key=lambda x: x[1]["overall_score"])[0]
        best_blurb = blurbs[best_blurb_style]
        
        # Determine the best title based on analyses
        best_title = max(title_analyses.items(), key=lambda x: x[1]["overall_score"])[0]
        
        # Create the publication package
        publication_package = {
            "title": best_title,
            "blurb": best_blurb,
            "positioning": positioning_statement["primary_positioning"],
            "alternative_positionings": positioning_statement.get("alternatives", []),
            "comp_titles": [
                f"{comp['title']} by {comp['author']}" for comp in comp_titles[:5]
            ],
            "categories": category_optimization["optimized_categories"],
            "keywords": metadata["metadata"]["keywords"],
            "audience_tags": metadata["metadata"]["audience_tags"],
            "content_descriptors": metadata["metadata"]["content_descriptors"],
            "manuscript": {
                "total_word_count": novel_state.current_word_count,
                "chapter_count": len(novel_state.chapters),
                "completed": True
            }
        }
        
        return {**state, "publication_package": publication_package}
    
    # 10. Set next phase
    def set_next_phase(state: PublicationState) -> PublicationState:
        novel_state = state["novel_state"]
        novel_state = project_lead.set_project_phase(novel_state, ProjectStatus.COMPLETED)
        
        # Update completion metrics
        novel_state.phase_progress = 1.0  # This phase is complete
        
        return {**state, "novel_state": novel_state}
    
    # Add nodes to graph
    workflow.add_node("generate_blurbs", generate_blurbs)
    workflow.add_node("analyze_blurbs", analyze_blurbs)
    workflow.add_node("generate_title_options", generate_title_options)
    workflow.add_node("analyze_titles", analyze_titles)
    workflow.add_node("identify_comp_titles", identify_comp_titles)
    workflow.add_node("create_positioning_statement", create_positioning_statement)
    workflow.add_node("generate_metadata", generate_metadata)
    workflow.add_node("optimize_categories", optimize_categories)
    workflow.add_node("prepare_publication_package", prepare_publication_package)
    workflow.add_node("set_next_phase", set_next_phase)
    
    # Define edges
    workflow.add_edge("generate_blurbs", "analyze_blurbs")
    workflow.add_edge("analyze_blurbs", "generate_title_options")
    workflow.add_edge("generate_title_options", "analyze_titles")
    workflow.add_edge("analyze_titles", "identify_comp_titles")
    workflow.add_edge("identify_comp_titles", "create_positioning_statement")
    workflow.add_edge("create_positioning_statement", "generate_metadata")
    workflow.add_edge("generate_metadata", "optimize_categories")
    workflow.add_edge("optimize_categories", "prepare_publication_package")
    workflow.add_edge("prepare_publication_package", "set_next_phase")
    
    # Set entry and exit points
    workflow.set_entry_point("generate_blurbs")
    
    return workflow

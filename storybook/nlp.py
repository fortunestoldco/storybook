from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.message import ToolCall, ToolResponse
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from uuid import uuid4

from storybook.state import State, AgentOutput
from storybook.agents import (
    NarrativeArcSurgeon, CharacterResonanceAnalyzer, EmotionalImpactOptimizer,
    MarketAlignmentStrategist, ProseElevationSpecialist, ReaderEngagementPredictor,
    ThematicCoherenceEngineer, DialogueAuthenticator, WorldBuildingImmersionEnhancer,
    NarrativeBlindSpotDetector
)
from storybook.config import Configuration

# Define input schema with Enums
class SubmissionType(str, Enum):
    NEW = "New"
    EXISTING = "Existing"

class ModelType(str, Enum):
    GPT_4 = "GPT-4"
    CLAUDE_3 = "Claude 3"
    GEMINI = "Gemini"
    LLAMA_3 = "Llama 3"

class CustomModelType(str, Enum):
    NONE = "None"
    FICTION_SPECIALIZED = "Fiction Specialized"
    TECHNICAL_WRITING = "Technical Writing"
    CREATIVE_PLUS = "Creative Plus"

class InputState(BaseModel):
    submission_type: SubmissionType
    title: Optional[str] = None
    manuscript: Optional[str] = None
    model: ModelType
    custom_model: Optional[CustomModelType] = CustomModelType.NONE
    project_id: Optional[str] = None

class AnalysisReport(BaseModel):
    agent_id: str
    report_id: str
    content: str
    timestamp: str
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)

class ProjectState(BaseModel):
    project_id: str
    title: str
    manuscript: str
    status: str = "initiated"
    revision_number: int = 0
    reports: List[AnalysisReport] = Field(default_factory=list)
    current_phase: str = "initialization"
    
class EnhancedState(State):
    submission: InputState
    project: Optional[ProjectState] = None
    human_in_loop: bool = False
    current_agent: Optional[str] = None
    analysis_chain: List[str] = Field(default_factory=list)
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    full_analysis_complete: bool = False

# Agent node functions
async def initialize_workflow(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Initialize the Top Secret Research Division workflow."""
    if state.submission.submission_type == SubmissionType.NEW:
        # Generate project ID for new submission
        project_id = f"TSRD-{uuid4().hex[:8].upper()}"
        
        return {
            "project": ProjectState(
                project_id=project_id,
                title=state.submission.title or "Untitled Project",
                manuscript=state.submission.manuscript or "",
                status="analysis_pending"
            ),
            "analysis_chain": [
                "narrative_structure", "character_analysis", "emotional_impact",
                "market_alignment", "prose_quality", "reader_engagement",
                "thematic_coherence", "dialogue_effectiveness", 
                "world_building", "blind_spot_detection"
            ],
            "current_agent": "narrative_structure"
        }
    else:
        # Retrieve existing project
        if not state.submission.project_id:
            return {"status": "error", "error": "Project ID required for existing submissions"}
        
        # Mock retrieval of existing project
        project_id = state.submission.project_id
        return {
            "project": ProjectState(
                project_id=project_id,
                title=state.submission.title or "Retrieved Title",
                manuscript=state.submission.manuscript or "Retrieved Manuscript",
                status="analysis_resuming",
                revision_number=1  # Assuming this is a revision
            ),
            "analysis_chain": [
                "narrative_structure", "character_analysis", "emotional_impact",
                "market_alignment", "prose_quality", "reader_engagement",
                "thematic_coherence", "dialogue_effectiveness", 
                "world_building", "blind_spot_detection"
            ],
            "current_agent": "narrative_structure"
        }

async def analyze_narrative_structure(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Narrative Arc Surgeon analyzes manuscript structure."""
    agent = NarrativeArcSurgeon(config)
    
    analysis = await agent.analyze_manuscript(
        manuscript=state.project.manuscript,
        title=state.project.title
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="narrative_arc_surgeon",
        report_id=f"NAS-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "narrative_structure")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_character_resonance(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Character Resonance Analyzer evaluates character development."""
    agent = CharacterResonanceAnalyzer(config)
    
    # Pass in narrative analysis results for context
    narrative_report = next((r for r in state.project.reports if r.agent_id == "narrative_arc_surgeon"), None)
    
    analysis = await agent.analyze_characters(
        manuscript=state.project.manuscript,
        narrative_context=narrative_report.content if narrative_report else None
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="character_resonance_analyzer",
        report_id=f"CRA-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "character_analysis")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_emotional_impact(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Emotional Impact Optimizer assesses emotional resonance."""
    agent = EmotionalImpactOptimizer(config)
    
    # Get previous reports for context
    previous_reports = {r.agent_id: r.content for r in state.project.reports}
    
    analysis = await agent.analyze_emotional_impact(
        manuscript=state.project.manuscript,
        character_analysis=previous_reports.get("character_resonance_analyzer"),
        narrative_analysis=previous_reports.get("narrative_arc_surgeon")
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="emotional_impact_optimizer",
        report_id=f"EIO-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "emotional_impact")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_market_alignment(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Market Alignment Strategist evaluates commercial viability."""
    agent = MarketAlignmentStrategist(config)
    
    analysis = await agent.analyze_market_fit(
        manuscript=state.project.manuscript,
        title=state.project.title
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="market_alignment_strategist",
        report_id=f"MAS-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "market_alignment")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_prose_quality(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Prose Elevation Specialist evaluates language quality."""
    agent = ProseElevationSpecialist(config)
    
    analysis = await agent.analyze_prose(
        manuscript=state.project.manuscript
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="prose_elevation_specialist",
        report_id=f"PES-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "prose_quality")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_reader_engagement(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Reader Engagement Predictor identifies engagement factors."""
    agent = ReaderEngagementPredictor(config)
    
    # Collect relevant previous analyses
    previous_reports = {r.agent_id: r.content for r in state.project.reports}
    
    analysis = await agent.predict_engagement(
        manuscript=state.project.manuscript,
        narrative_analysis=previous_reports.get("narrative_arc_surgeon"),
        character_analysis=previous_reports.get("character_resonance_analyzer"),
        emotional_analysis=previous_reports.get("emotional_impact_optimizer")
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="reader_engagement_predictor",
        report_id=f"REP-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "reader_engagement")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_thematic_coherence(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Thematic Coherence Engineer assesses thematic elements."""
    agent = ThematicCoherenceEngineer(config)
    
    analysis = await agent.analyze_themes(
        manuscript=state.project.manuscript
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="thematic_coherence_engineer",
        report_id=f"TCE-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "thematic_coherence")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_dialogue_effectiveness(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Dialogue Authenticator evaluates dialogue quality."""
    agent = DialogueAuthenticator(config)
    
    # Get character analysis for context
    character_report = next((r for r in state.project.reports if r.agent_id == "character_resonance_analyzer"), None)
    
    analysis = await agent.analyze_dialogue(
        manuscript=state.project.manuscript,
        character_context=character_report.content if character_report else None
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="dialogue_authenticator",
        report_id=f"DA-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "dialogue_effectiveness")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def analyze_world_building(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """World-Building Immersion Enhancer evaluates setting depth."""
    agent = WorldBuildingImmersionEnhancer(config)
    
    analysis = await agent.analyze_world_building(
        manuscript=state.project.manuscript
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="world_building_immersion_enhancer",
        report_id=f"WBIE-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    
    # Move to next agent in chain
    next_agent = get_next_agent(state.analysis_chain, "world_building")
    
    return {
        "project": updated_project,
        "current_agent": next_agent
    }

async def detect_blind_spots(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Narrative Blind Spot Detector identifies author assumptions and issues."""
    agent = NarrativeBlindSpotDetector(config)
    
    # Use all previous reports for comprehensive context
    previous_reports = {r.agent_id: r.content for r in state.project.reports}
    
    analysis = await agent.detect_blind_spots(
        manuscript=state.project.manuscript,
        previous_analyses=previous_reports
    )
    
    # Create structured report
    report = AnalysisReport(
        agent_id="narrative_blind_spot_detector",
        report_id=f"NBSD-{uuid4().hex[:6].upper()}",
        content=analysis.content,
        timestamp=datetime.now().isoformat(),
        recommendations=analysis.recommendations
    )
    
    # Update project with report
    updated_project = state.project.copy()
    updated_project.reports.append(report)
    updated_project.status = "analysis_complete"
    
    return {
        "project": updated_project,
        "current_agent": "final_synthesis",
        "full_analysis_complete": True
    }

async def synthesize_analysis(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Meta-analysis that synthesizes all agent reports into actionable insights."""
    # This function would combine all specialist reports into a cohesive strategy
    
    # For now, we'll create a simplified synthesis
    all_recommendations = []
    for report in state.project.reports:
        all_recommendations.extend(report.recommendations)
    
    # Prioritize recommendations by impact
    # This is a simplified approach - in reality would have more sophisticated ranking
    prioritized_recommendations = sorted(
        all_recommendations, 
        key=lambda x: x.get("impact_level", 0),
        reverse=True
    )
    
    synthesis = {
        "title": f"Comprehensive Analysis of {state.project.title}",
        "timestamp": datetime.now().isoformat(),
        "major_findings": "Synthesized insights from all specialist analyses",
        "prioritized_recommendations": prioritized_recommendations[:10],
        "revision_strategy": "Multi-stage approach focusing on structure, character, and prose",
    }
    
    updated_project = state.project.copy()
    updated_project.status = "ready_for_author_review"
    
    return {
        "project": updated_project,
        "current_agent": "author_review",
        "synthesis": synthesis
    }

async def present_to_author(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Present the analysis to the author for review."""
    if not state.human_in_loop:
        # Prepare comprehensive report for author
        synthesis = state.get("synthesis", {})
        presentation = f"# Comprehensive Analysis of '{state.project.title}'\n\n"
        presentation += "Our Top Secret Research Division has completed their analysis. Here are the key findings:\n\n"
        
        # Add highlights from each specialist
        for report in state.project.reports:
            agent_name = report.agent_id.replace("_", " ").title()
            presentation += f"## {agent_name} Analysis\n"
            presentation += f"{report.content[:300]}...\n\n"
        
        # Add prioritized recommendations
        presentation += "## Top Recommendations\n"
        for i, rec in enumerate(synthesis.get("prioritized_recommendations", [])[:5]):
            presentation += f"{i+1}. {rec.get('description')}\n"
        
        presentation += "\nWould you like to proceed with implementing these recommendations? Or would you prefer to review the full analysis first?"
        
        return {
            "human_in_loop": True,
            "chat_history": [{"role": "system", "content": presentation}],
            "current_agent": "author_review"
        }
    
    # Process author response
    last_message = state.chat_history[-1] if state.chat_history else None
    if last_message and last_message.get("role") == "user":
        user_input = last_message.get("content", "").lower()
        
        if "implement" in user_input or "proceed" in user_input:
            # Author wants to implement recommendations
            response = "Excellent! We'll begin implementing the recommendations. Your manuscript will enter the revision phase."
            
            updated_project = state.project.copy()
            updated_project.status = "revision_in_progress"
            updated_project.revision_number += 1
            
            return {
                "human_in_loop": False,
                "chat_history": state.chat_history + [{"role": "system", "content": response}],
                "project": updated_project,
                "current_agent": "implementation"
            }
        elif "review" in user_input or "details" in user_input:
            # Author wants more details
            response = "I'll provide the complete analysis. Please let me know if you have specific questions about any area."
            
            # Here we would present more detailed information
            
            return {
                "human_in_loop": True,
                "chat_history": state.chat_history + [{"role": "system", "content": response}],
                "current_agent": "author_review"
            }
        elif "complete" in user_input or "finished" in user_input:
            # Author wants to complete the process
            response = "Thank you for working with our Top Secret Research Division! Your project is now marked as complete."
            
            updated_project = state.project.copy()
            updated_project.status = "completed"
            
            return {
                "human_in_loop": False,
                "chat_history": state.chat_history + [{"role": "system", "content": response}],
                "project": updated_project,
                "current_agent": "complete"
            }
    
    # Default response
    response = "Would you like me to implement the recommendations, provide more details, or mark the project as complete?"
    return {
        "human_in_loop": True,
        "chat_history": state.chat_history + [{"role": "system", "content": response}],
        "current_agent": "author_review"
    }

async def implement_recommendations(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Implementation phase for the recommended changes."""
    # This would be a complex phase involving multiple agents
    # For now, we'll simulate the implementation
    
    updated_project = state.project.copy()
    updated_project.status = "revision_complete"
    
    implementation_message = (
        f"The revision of '{state.project.title}' (Revision #{updated_project.revision_number}) "
        f"has been completed according to the specialist recommendations. "
        f"The manuscript is ready for your review."
    )
    
    return {
        "project": updated_project,
        "current_agent": "complete",
        "chat_history": state.chat_history + [{"role": "system", "content": implementation_message}]
    }

# Helper function to determine next agent in analysis chain
def get_next_agent(analysis_chain, current_agent_key):
    """Returns the next agent in the analysis chain."""
    agent_map = {
        "narrative_structure": "character_analysis",
        "character_analysis": "emotional_impact",
        "emotional_impact": "market_alignment",
        "market_alignment": "prose_quality",
        "prose_quality": "reader_engagement",
        "reader_engagement": "thematic_coherence",
        "thematic_coherence": "dialogue_effectiveness",
        "dialogue_effectiveness": "world_building",
        "world_building": "blind_spot_detection"
    }
    
    return agent_map.get(current_agent_key, "final_synthesis")

def should_end(state: EnhancedState) -> bool:
    """Determine if the workflow should end."""
    return state.current_agent in ["complete", "error"]

def build_storybook(config: RunnableConfig) -> StateGraph:
    """Build and return the Top Secret Research Division graph."""
    builder = StateGraph(EnhancedState, config_schema=Configuration)

    # Add nodes for each specialist agent
    builder.add_node("initialize_workflow", initialize_workflow)
    builder.add_node("narrative_structure", analyze_narrative_structure)
    builder.add_node("character_analysis", analyze_character_resonance)
    builder.add_node("emotional_impact", analyze_emotional_impact)
    builder.add_node("market_alignment", analyze_market_alignment)
    builder.add_node("prose_quality", analyze_prose_quality)
    builder.add_node("reader_engagement", analyze_reader_engagement)
    builder.add_node("thematic_coherence", analyze_thematic_coherence)
    builder.add_node("dialogue_effectiveness", analyze_dialogue_effectiveness)
    builder.add_node("world_building", analyze_world_building)
    builder.add_node("blind_spot_detection", detect_blind_spots)
    builder.add_node("final_synthesis", synthesize_analysis)
    builder.add_node("author_review", present_to_author)
    builder.add_node("implementation", implement_recommendations)

    # Add conditional edges based on the current agent
    builder.add_conditional_edges(
        "initialize_workflow",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "narrative_structure",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "character_analysis",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "emotional_impact",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "market_alignment",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "prose_quality",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "reader_engagement",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "thematic_coherence",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "dialogue_effectiveness",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "world_building",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "blind_spot_detection",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "final_synthesis",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "author_review",
        lambda state: state.current_agent
    )
    
    builder.add_conditional_edges(
        "implementation",
        lambda state: state.current_agent
    )

    # Add end condition
    builder.add_edge_filter(should_end, END)

    # Set entry point
    builder.set_entry_point("initialize_workflow")

    # Compile and return graph
    graph = builder.compile()
    graph.name = "TopSecretResearchDivisionGraph"
    return graph

# Export the graph builder function
__all__ = ["build_storybook"]
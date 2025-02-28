from __future__ import annotations

from typing import Dict, Any, Optional, List, TypedDict, Literal
from pathlib import Path
from langgraph.graph import StateGraph
from langgraph.channels import LastValue

# Project Management
from .agents.project_management.project_lead_agent import ProjectLeadAgent
from .agents.project_management.market_research_agent import MarketResearchAgent
from .agents.project_management.novel_identity_agent import NovelIdentityAgent

# Cultural Relevance
from .agents.cultural_relevance.zeitgeist_analysis_agent import ZeitgeistAnalysisAgent
from .agents.cultural_relevance.trend_forecasting_agent import TrendForecastingAgent
from .agents.cultural_relevance.cultural_conversation_agent import CulturalConversationAgent

# Story Architecture
from .agents.story_architecture.structure_specialist_agent import StructureSpecialistAgent
from .agents.story_architecture.plot_development_agent import PlotDevelopmentAgent
from .agents.story_architecture.genre_innovation_agent import GenreInnovationAgent
from .agents.story_architecture.architecture_coordinator import ArchitectureCoordinator

# Core Creative
from .agents.character_analyst import CharacterAnalyst
from .agents.subplot_weaver import SubplotWeaver
from .agents.world_builder import WorldBuilder

# Writing
from .agents.writing.chapter_writer_agent import ChapterWriterAgent
from .agents.writing.continuity_manager import ContinuityManager
from .agents.writing.description_specialist import DescriptionSpecialist

# Research Team
from .agents.research.historical_research_agent import HistoricalResearchAgent
from .agents.research.technical_domain_agent import TechnicalDomainAgent
from .agents.research.cultural_authenticity_agent import CulturalAuthenticityAgent
from .agents.research.research_supervisor_agent import ResearchSupervisorAgent

# Creative Coordinator
from .agents.creative.creative_coordinator import CreativeCoordinator

from .models.schema import StoryInput, StoryState, AgentConfig
from .llm.provider import LLMProvider

def build_storybook(manuscript_text: str, config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """Build the storybook processing graph."""
    workflow = StateGraph()
    
    # Create story input with only manuscript text required
    story_input = StoryInput(
        manuscript_text=manuscript_text,
        **(config or {})
    )
    
    # Generate unique ID for the manuscript
    manuscript_id = story_input.generate_id()
    
    # Initialize state with manuscript text and generated ID
    initial_state: StoryState = {
        "manuscript": {
            "id": manuscript_id,
            "text": story_input.manuscript_text
        },
        "market_analysis": {},
        "cultural_analysis": {},
        "research_findings": {},
        "characters": [],
        "structure": {},
        "subplots": [],
        "world_building": {},
        "chapters": [],
        "state": "start"
    }
    
    # Validate input configuration
    story_config = StoryInput(**config) if config else StoryInput()
    
    # Initialize agents with configured LLMs
    agents = {}
    for agent_name, agent_config in story_config.agent_configuration.items():
        llm = LLMProvider.initialize_llm(agent_config.llm)
        agents[agent_name] = agent_class_map[agent_name](
            llm=llm,
            config=agent_config
        )
    
    # Add channels with proper typing
    workflow.add_channel("manuscript", LastValue[Dict[str, Any]])
    workflow.add_channel("market_analysis", LastValue[Dict[str, Any]])
    workflow.add_channel("cultural_analysis", LastValue[Dict[str, Any]])
    workflow.add_channel("research_findings", LastValue[Dict[str, Any]])
    workflow.add_channel("characters", LastValue[List[Dict[str, Any]]])
    workflow.add_channel("structure", LastValue[Dict[str, Any]])
    workflow.add_channel("subplots", LastValue[List[Dict[str, Any]]])
    workflow.add_channel("world_building", LastValue[Dict[str, Any]])
    workflow.add_channel("chapters", LastValue[List[Dict[str, Any]]])
    workflow.add_channel("state", LastValue[str])

    @workflow.node
    def project_setup(state):
        """Initialize project setup with management team."""
        project_results = agents["project_lead"].process_manuscript(state["manuscript"]["id"])
        market_results = agents["market_research"].process_manuscript(state["manuscript"]["id"])
        identity_results = agents["novel_identity"].process_manuscript(state["manuscript"]["id"])
        
        return {
            "manuscript": {**state["manuscript"], **identity_results},
            "market_analysis": market_results,
            "state": "setup_complete"
        }

    @workflow.node
    async def research_phase(state):
        """Conduct comprehensive research."""
        # Historical research
        historical_results = await agents["historical_research"].process_manuscript(
            state["manuscript"]["id"]
        )
        
        # Technical domain research
        technical_results = await agents["technical_research"].process_manuscript(
            state["manuscript"]["id"]
        )
        
        # Cultural authenticity research
        cultural_results = await agents["cultural_research"].process_manuscript(
            state["manuscript"]["id"]
        )
        
        # Supervise and analyze research
        research_summary = await agents["research_supervisor"].process_manuscript(
            state["manuscript"]["id"]
        )
        
        return {
            "research_findings": {
                "historical": historical_results,
                "technical": technical_results,
                "cultural": cultural_results,
                "summary": research_summary
            },
            "state": "research_complete"
        }

    @workflow.node
    def cultural_analysis(state):
        """Analyze cultural context."""
        zeitgeist_results = agents["zeitgeist"].process_manuscript(state["manuscript"]["id"])
        trend_results = agents["trend_forecaster"].process_manuscript(state["manuscript"]["id"])
        conversation_results = agents["cultural_conversation"].process_manuscript(state["manuscript"]["id"])
        
        return {
            "cultural_analysis": {
                "zeitgeist": zeitgeist_results,
                "trends": trend_results,
                "conversations": conversation_results
            },
            "state": "culture_analyzed"
        }

    @workflow.node
    async def story_architecture(state):
        """Define story structure integrating research findings."""
        # Get architecture plan
        architecture_plan = await agents["architecture_coordinator"].process_architecture(state)
        
        # Process with specialized agents
        structure_results = await agents["structure_specialist"].process_manuscript(
            state["manuscript"]["id"],
            context=architecture_plan
        )
        
        # Plot development with research context
        plot_results = await agents["plot_development"].process_manuscript(
            state["manuscript"]["id"],
            context={
                **architecture_plan,
                "structure": structure_results
            }
        )
        
        # Genre innovation with full context
        genre_results = await agents["genre_innovation"].process_manuscript(
            state["manuscript"]["id"],
            context={
                **architecture_plan,
                "structure": structure_results,
                "plot": plot_results
            }
        )
        
        # Validate architecture cohesion
        needs_research = any([
            structure_results.get("needs_research", False),
            plot_results.get("research_elements", {}).get("needs_validation", False),
            genre_results.get("research_validation", {}).get("needs_review", False)
        ])
        
        return {
            "structure": {
                "framework": structure_results,
                "plot": plot_results,
                "genre": genre_results,
                "research_integration": architecture_plan,
                "needs_research": needs_research
            },
            "state": "architecture_complete"
        }

    @workflow.node
    async def creative_development(state):
        """Develop creative elements based on architecture."""
        # Get creative development plan
        creative_plan = await agents["creative_coordinator"].process_creative_phase(state)
        
        # Character development with architectural context
        character_results = await agents["character_analyst"].process_manuscript(
            state["manuscript"]["id"],
            context={
                "structure": state["structure"],
                "plan": creative_plan
            }
        )
        
        # World building with character context
        world_results = await agents["world_builder"].process_manuscript(
            state["manuscript"]["id"],
            context={
                "structure": state["structure"],
                "characters": character_results,
                "plan": creative_plan
            }
        )
        
        # Subplot development with full context
        subplot_results = await agents["subplot_weaver"].process_manuscript(
            state["manuscript"]["id"],
            characters=character_results,
            context={
                "structure": state["structure"],
                "world": world_results,
                "plan": creative_plan
            }
        )
        
        # Validate creative cohesion
        needs_architecture_review = _validate_creative_cohesion(
            creative_plan,
            character_results,
            world_results,
            subplot_results
        )
        
        return {
            "characters": character_results,
            "world_building": world_results,
            "subplots": subplot_results,
            "development_plan": creative_plan,
            "state": "development_complete",
            "needs_architecture_review": needs_architecture_review
        }

    def _validate_creative_cohesion(
        plan: Dict[str, Any],
        characters: Dict[str, Any],
        world: Dict[str, Any],
        subplots: Dict[str, Any]
    ) -> bool:
        """Validate coherence between creative elements."""
        # Implementation for validation
        pass

    @workflow.node
    async def chapter_development(state):
        """Develop and write chapters."""
        chapter_results = await agents["chapter_writer"].process_manuscript(
            state["manuscript"]["id"],
            context={
                "structure": state["structure"],
                "characters": state["characters"],
                "world": state["world_building"],
                "subplots": state["subplots"]
            }
        )
        
        continuity_results = await agents["continuity_manager"].process_manuscript(
            state["manuscript"]["id"],
            context={
                "chapters": chapter_results,
                "structure": state["structure"]
            }
        )
        
        description_results = await agents["description_specialist"].process_manuscript(
            state["manuscript"]["id"],
            context={
                "chapters": chapter_results,
                "world": state["world_building"]
            }
        )
        
        return {
            "chapters": chapter_results,
            "continuity": continuity_results,
            "descriptions": description_results,
            "state": "chapter_complete"
        }

    # Configure workflow routing
    workflow.set_entry_point("project_setup")
    workflow.add_conditional_edges(
        lambda state: {
            "start": "project_setup",
            "setup_complete": "research_phase",  # Add research phase
            "research_complete": "story_architecture",
            "architecture_complete": "creative_development",
            "development_complete": (
                "story_architecture" if state.get("needs_architecture_review", False)
                else "chapter_development"
            ),
            "chapter_complete": "end"
        }.get(state["state"], "end")
    )
    
    workflow.set_initial_state(initial_state)
    
    return workflow
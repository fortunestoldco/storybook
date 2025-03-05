"""Factory for creating specialized novel writing agents."""

from typing import Dict, Any, List, Callable, Optional, Type
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from storybook.configuration import Configuration
from storybook.state import NovelSystemState
from storybook.utils import load_chat_model
from storybook.agents.base import BaseAgent
from storybook.agents.phase_agents import (
    ExecutiveDirector,
    CreativeDirector,
    HumanFeedbackManager,
    QualityAssessmentDirector,
    ProjectTimelineManager,
    MarketAlignmentDirector,
    StructureArchitect,
    PlotDevelopmentSpecialist,
    WorldBuildingExpert,
    CharacterPsychologySpecialist,
    CharacterVoiceDesigner,
    CharacterRelationshipMapper,
    DomainKnowledgeSpecialist,
    CulturalAuthenticityExpert,
    ContentDevelopmentDirector,
    ChapterDrafters,
    SceneConstructionSpecialists,
    DialogueCrafters,
    ContinuityManager,
    VoiceConsistencyMonitor,
    EmotionalArcDesigner,
    EditorialDirector,
    StructuralEditor,
    CharacterArcEvaluator,
    ThematicCoherenceAnalyst,
    ProseEnhancementSpecialist,
    DialogueRefinementExpert,
    RhythmCadenceOptimizer,
    GrammarConsistencyChecker,
    FactVerificationSpecialist,
    PositioningSpecialist,
    TitleBlurbOptimizer,
    DifferentiationStrategist,
    FormattingStandardsExpert
)
from storybook.tools import tool_registry
from storybook.research.agents import (
    DomainKnowledgeSpecialist,
    CulturalAuthenticityExpert, 
    MarketAlignmentDirector,
    FactVerificationSpecialist
)
from .research.graphs import create_research_subgraph
from ..research.agents import ResearchAgent
from ..research.states import (
    DomainResearchState,
    CulturalResearchState, 
    MarketResearchState,
    FactVerificationState
)
from ..research import graphs


class AgentFactory:
    """Factory for creating specialized novel writing agents."""

    def __init__(self, config: Configuration):
        """Initialize the agent factory.

        Args:
            config: System configuration.
        """
        self.config = config
        self.base_model = load_chat_model(config.model)
        self.agent_roles = config.agent_roles
        
        # Map agent names to their class implementations
        self.agent_classes: Dict[str, Type[BaseAgent]] = {
            "executive_director": ExecutiveDirector,
            "creative_director": CreativeDirector,
            "human_feedback_manager": HumanFeedbackManager,
            "quality_assessment_director": QualityAssessmentDirector,
            "project_timeline_manager": ProjectTimelineManager,
            "market_alignment_director": MarketAlignmentDirector,
            "structure_architect": StructureArchitect,
            "plot_development_specialist": PlotDevelopmentSpecialist,
            "world_building_expert": WorldBuildingExpert,
            "character_psychology_specialist": CharacterPsychologySpecialist,
            "character_voice_designer": CharacterVoiceDesigner,
            "character_relationship_mapper": CharacterRelationshipMapper,
            "domain_knowledge_specialist": DomainKnowledgeSpecialist,
            "cultural_authenticity_expert": CulturalAuthenticityExpert,
            "content_development_director": ContentDevelopmentDirector,
            "chapter_drafters": ChapterDrafters,
            "scene_construction_specialists": SceneConstructionSpecialists,
            "dialogue_crafters": DialogueCrafters,
            "continuity_manager": ContinuityManager,
            "voice_consistency_monitor": VoiceConsistencyMonitor,
            "emotional_arc_designer": EmotionalArcDesigner,
            "editorial_director": EditorialDirector,
            "structural_editor": StructuralEditor,
            "character_arc_evaluator": CharacterArcEvaluator,
            "thematic_coherence_analyst": ThematicCoherenceAnalyst,
            "prose_enhancement_specialist": ProseEnhancementSpecialist,
            "dialogue_refinement_expert": DialogueRefinementExpert,
            "rhythm_cadence_optimizer": RhythmCadenceOptimizer,
            "grammar_consistency_checker": GrammarConsistencyChecker,
            "fact_verification_specialist": FactVerificationSpecialist,
            "positioning_specialist": PositioningSpecialist,
            "title_blurb_optimizer": TitleBlurbOptimizer,
            "differentiation_strategist": DifferentiationStrategist,
            "formatting_standards_expert": FormattingStandardsExpert
        }

    def create_agent(self, agent_name: str, project_id: str) -> BaseAgent:
        """Create an agent instance."""
        
        # Research agents use the deep research subgraph
        research_agents = {
            "domain_knowledge_specialist": {
                "research_type": "domain",
                "state_class": DomainResearchState
            },
            "cultural_authenticity_expert": {
                "research_type": "cultural",
                "state_class": CulturalResearchState  
            },
            "market_alignment_director": {
                "research_type": "market",
                "state_class": MarketResearchState
            },
            "fact_verification_specialist": {
                "research_type": "fact",
                "state_class": FactVerificationState
            }
        }
        
        if agent_name in research_agents:
            # Create research subgraph for this agent
            research_config = research_agents[agent_name]
            research_graph = create_research_subgraph(research_config["research_type"])
            
            return ResearchAgent(
                name=agent_name,
                research_graph=research_graph,
                state_class=research_config["state_class"],
                config=self.get_research_config(agent_name)
            )
            
        if agent_name not in self.agent_roles:
            raise ValueError(f"Unknown agent role: {agent_name}")

        role_description = self.agent_roles[agent_name]
        agent_tools = tool_registry.get_tools_for_agent(agent_name)
        
        # Initialize agent with all required parameters
        if agent_name in self.agent_classes:
            agent_class = self.agent_classes[agent_name]
            agent_instance = agent_class(
                name=agent_name,
                chat_model=self.base_model,
                tools=agent_tools,
                project_id=project_id,
                role_description=role_description
            )
        else:
            agent_instance = BaseAgent(
                name=agent_name,
                chat_model=self.base_model,
                tools=agent_tools,
                project_id=project_id,
                role_description=role_description
            )

        async def agent_function(state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
            """The agent function that processes state and returns an updated state.

            Args:
                state: Current system state.
                config: Runtime configuration.

            Returns:
                Dictionary with updates to the state.
            """
            configuration = Configuration.from_runnable_config(config)
            
            # Let the agent instance process the state
            result = await agent_instance.process(state, configuration)
            
            # Format the result for the state update
            return {
                "messages": [result.message],
                "current_agent": agent_name,
                "agent_outputs": {
                    **state.agent_outputs,
                    agent_name: state.agent_outputs.get(agent_name, []) + [{
                        "timestamp": datetime.now().isoformat(),
                        "task": state.current_input.get("task", ""),
                        "response": result.message.content
                    }]
                }
            }

        return agent_function

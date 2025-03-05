"""Agent factory module."""
from typing import Dict, Type, Any
from storybook.configuration import Configuration
from storybook.agents.base_agent import BaseAgent

# Initialization Phase
from storybook.agents.initialization.executive_director import ExecutiveDirector
from storybook.agents.initialization.quality_assessment_director import QualityAssessmentDirector
from storybook.agents.initialization.human_feedback_manager import HumanFeedbackManager

# Development Phase  
from storybook.agents.development.creative_director import CreativeDirector
from storybook.agents.development.structure_architect import StructureArchitect
from storybook.agents.development.plot_development_specialist import PlotDevelopmentSpecialist
from storybook.agents.development.world_building_expert import WorldBuildingExpert
from storybook.agents.development.domain_knowledge_specialist import DomainKnowledgeSpecialist
from storybook.agents.development.cultural_authenticity_expert import CulturalAuthenticityExpert

# Creation Phase
from storybook.agents.creation import (
    ContentDevelopmentDirector,
    ChapterDrafter,
    DialogueCrafter,
    ContinuityManager,
    SceneConstructionSpecialist,
    VoiceConsistencyMonitor,
    EmotionalArcDesigner
)

from storybook.agents.refinement.editorial_director import EditorialDirector
from storybook.agents.refinement.prose_enhancement_specialist import ProseEnhancementSpecialist
from storybook.agents.refinement.structural_editor import StructuralEditor

from storybook.agents.finalization.formatting_standards_expert import FormattingStandardsExpert
from storybook.agents.finalization.positioning_specialist import PositioningSpecialist
from storybook.agents.finalization.market_alignment_director import MarketAlignmentDirector

class AgentFactory:
    """Factory for creating novel writing agents."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self._init_agent_registry()
    
    def _init_agent_registry(self) -> None:
        """Initialize the agent registry."""
        self.agent_registry: Dict[str, Type[BaseAgent]] = {
            # Initialization Phase
            "executive_director": ExecutiveDirector,
            "quality_assessment_director": QualityAssessmentDirector,
            "human_feedback_manager": HumanFeedbackManager,
            
            # Development Phase
            "creative_director": CreativeDirector,
            "structure_architect": StructureArchitect,
            "plot_development_specialist": PlotDevelopmentSpecialist,
            "world_building_expert": WorldBuildingExpert,
            "domain_knowledge_specialist": DomainKnowledgeSpecialist,
            "cultural_authenticity_expert": CulturalAuthenticityExpert,
            
            # Creation Phase
            "content_development_director": ContentDevelopmentDirector,
            "chapter_drafter": ChapterDrafter,
            "dialogue_crafter": DialogueCrafter,
            "continuity_manager": ContinuityManager,
            "scene_construction_specialist": SceneConstructionSpecialist,
            "voice_consistency_monitor": VoiceConsistencyMonitor,
            "emotional_arc_designer": EmotionalArcDesigner,
            
            # Refinement Phase
            "editorial_director": EditorialDirector,
            "prose_enhancement_specialist": ProseEnhancementSpecialist,
            "structural_editor": StructuralEditor,
            
            # Finalization Phase
            "formatting_standards_expert": FormattingStandardsExpert,
            "positioning_specialist": PositioningSpecialist,
            "market_alignment_director": MarketAlignmentDirector
        }
    
    def create_agent(self, agent_name: str) -> BaseAgent:
        """Create an agent instance."""
        if agent_name not in self.agent_registry:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        return self.agent_registry[agent_name]()
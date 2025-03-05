"""Agent factory module."""
from typing import Dict, Type, Any
from storybook.configuration import Configuration
from storybook.agents.base_agent import BaseAgent

# Initialization Phase
from storybook.agents.initialization.executive_director import ExecutiveDirector
from storybook.agents.initialization.quality_assessment_director import QualityAssessmentDirector
from storybook.agents.initialization.human_feedback_manager import HumanFeedbackManager
from storybook.agents.initialization.project_timeline_manager import ProjectTimelineManager

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
from storybook.agents.creation.chapter_drafters import ChapterDrafters
from storybook.agents.creation.scene_construction_specialists import SceneConstructionSpecialists
from storybook.agents.creation.dialogue_crafters import DialogueCrafters

# Refinement Phase
from storybook.agents.refinement.editorial_director import EditorialDirector
from storybook.agents.refinement.prose_enhancement_specialist import ProseEnhancementSpecialist
from storybook.agents.refinement.structural_editor import StructuralEditor
from storybook.agents.refinement.fact_verification_specialist import FactVerificationSpecialist

# Finalization Phase
from storybook.agents.finalization.formatting_standards_expert import FormattingStandardsExpert
from storybook.agents.finalization.positioning_specialist import PositioningSpecialist
from storybook.agents.finalization.market_alignment_director import MarketAlignmentDirector
from storybook.agents.finalization.differentiation_strategist import DifferentiationStrategist

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
            "project_timeline_manager": ProjectTimelineManager,
            
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
            "chapter_drafters": ChapterDrafters,
            "scene_construction_specialists": SceneConstructionSpecialists,
            "dialogue_crafters": DialogueCrafters,
            
            # Refinement Phase
            "editorial_director": EditorialDirector,
            "prose_enhancement_specialist": ProseEnhancementSpecialist,
            "structural_editor": StructuralEditor,
            "fact_verification_specialist": FactVerificationSpecialist,
            
            # Finalization Phase
            "formatting_standards_expert": FormattingStandardsExpert,
            "positioning_specialist": PositioningSpecialist,
            "market_alignment_director": MarketAlignmentDirector,
            "differentiation_strategist": DifferentiationStrategist
        }
    
    def create_agent(self, agent_name: str) -> BaseAgent:
        """Create an agent instance."""
        if agent_name not in self.agent_registry:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        return self.agent_registry[agent_name]()

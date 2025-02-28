from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any, Optional
import logging

# Local imports
from storybook.agents.base import BaseAgent
from storybook.agents.character_developer import CharacterDeveloper
from storybook.agents.dialogue_enhancer import DialogueEnhancer
from storybook.agents.world_builder import WorldBuilder
from storybook.agents.subplot_weaver import SubplotWeaver
from storybook.agents.story_arc_analyst import StoryArcAnalyst
from storybook.agents.continuity_editor import ContinuityEditor
from storybook.agents.language_polisher import LanguagePolisher
from storybook.agents.quality_reviewer import QualityReviewer
from storybook.agents.market_researcher import MarketResearcher
from storybook.agents.content_analyzer import ContentAnalyzer
from .base_agent import BaseAgent
from .project_management.project_lead_agent import ProjectLeadAgent
from .project_management.market_research_agent import MarketResearchAgent
from .project_management.novel_identity_agent import NovelIdentityAgent
from .cultural_relevance.zeitgeist_analysis_agent import ZeitgeistAnalysisAgent
from .cultural_relevance.trend_forecasting_agent import TrendForecastingAgent
from .cultural_relevance.cultural_conversation_agent import CulturalConversationAgent
from .story_architecture.structure_specialist_agent import StructureSpecialistAgent
from .story_architecture.plot_development_agent import PlotDevelopmentAgent
from .story_architecture.genre_innovation_agent import GenreInnovationAgent
from .writing.chapter_writer_agent import ChapterWriterAgent
from .writing.continuity_manager import ContinuityManager
from .writing.description_specialist import DescriptionSpecialist

logger = logging.getLogger(__name__)

__all__ = [
    "BaseAgent",
    "CharacterDeveloper",
    "DialogueEnhancer",
    "WorldBuilder",
    "SubplotWeaver",
    "StoryArcAnalyst",
    "ContinuityEditor",
    "LanguagePolisher",
    "QualityReviewer",
    "MarketResearcher",
    "ContentAnalyzer",
    'ProjectLeadAgent',
    'MarketResearchAgent',
    'NovelIdentityAgent',
    'ZeitgeistAnalysisAgent',
    'TrendForecastingAgent',
    'CulturalConversationAgent',
    'StructureSpecialistAgent',
    'PlotDevelopmentAgent',
    'GenreInnovationAgent',
    'ChapterWriterAgent',
    'ContinuityManager',
    'DescriptionSpecialist'
]

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ThematicElement(BaseModel):
    """Model for a thematic element."""
    element_id: str = Field(description="Unique identifier for the thematic element")
    type: str = Field(description="Type of thematic element (theme, motif, symbol, etc.)")
    name: str = Field(description="Name or identifier of the element")
    description: str = Field(description="Detailed description of the element")
    manifestations: List[Dict[str, Any]] = Field(description="How the element manifests in the story")
    relationships: Dict[str, Any] = Field(description="Relationships to other thematic elements")
    development: List[Dict[str, Any]] = Field(description="Development of the element through the story")
    impact: Dict[str, Any] = Field(description="Impact on story elements")

class ThematicAnalysis(BaseModel):
    """Model for thematic analysis results."""
    theme_development: Dict[str, Any] = Field(description="Analysis of theme development")
    motif_patterns: Dict[str, Any] = Field(description="Analysis of motif patterns")
    symbol_usage: Dict[str, Any] = Field(description="Analysis of symbol usage")
    thematic_coherence: Dict[str, Any] = Field(description="Assessment of thematic coherence")
    character_themes: Dict[str, Any] = Field(description="Character-theme relationships")
    plot_themes: Dict[str, Any] = Field(description="Plot-theme relationships")
    recommendations: List[Dict[str, Any]] = Field(description="Suggested improvements")

class ThematicAnalysisAgent(BaseAgent):
    def __init__(self, tools_service: ToolsService):
        super().__init__(tools_service)
        self.logger = logging.getLogger(__name__)
        self.tools_service = tools_service
        self.llm_router = LLMRouter(tools_service)
        
    async def analyze_theme(self, content: str, parser: PydanticOutputParser) -> Any:
        try:
            return await self.llm_router.process_with_streaming(
                task="plot_analysis",  # Uses the correct model from config
                prompt=content,
                parser=parser
            )
        except Exception as e:
            self.logger.error(f"Error in theme analysis: {str(e)}")
            raise

        self.state = {
            "thematic_elements": {},
            "element_relationships": {},
            "analysis_history": [],
            "last_update": datetime(2025, 2, 24, 23, 46, 23)  # Current timestamp
        }

        # Initialize thematic categories
        self.thematic_categories = {
            "universal_themes": self._get_universal_themes(),
            "character_themes": self._get_character_themes(),
            "societal_themes": self._get_societal_themes(),
            "philosophical_themes": self._get_philosophical_themes()
        }

    async def create_thematic_element(
        self,
        element_type: str,
        name: str,
        description: str,
        story_bible: Dict[str, Any]
    ) -> ThematicElement:
        """Create a new thematic element."""
        try:
            # Validate element type
            valid_types = ['theme', 'motif', 'symbol', 'allegory', 'archetype']
            if element_type not in valid_types:
                raise ValueError(f"Invalid element type. Must be one of: {valid_types}")

            # Generate unique element ID
            element_id = f"THEME_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Analyze potential manifestations
            manifestations = await self._analyze_potential_manifestations(
                element_type,
                name,
                description,
                story_bible
            )

            # Identify relationships with existing elements
            relationships = await self._identify_thematic_relationships(
                element_type,
                name,
                description,
                story_bible
            )

            # Create initial development timeline
            development = await self._create_development_timeline(
                element_type,
                name,
                story_bible
            )

            # Analyze potential impact
            impact = await self._analyze_thematic_impact(
                element_type,
                name,
                description,
                story_bible
            )

            # Create element
            element = ThematicElement(
                element_id=element_id,
                type=element_type,
                name=name,
                description=description,
                manifestations=manifestations,
                relationships=relationships,
                development=development,
                impact=impact
            )

            # Validate thematic consistency
            await self._validate_thematic_consistency(element, story_bible)

            # Add to state
            self.state["thematic_elements"][element_id] = element
            self.state["last_update"] = datetime.utcnow()

            # Update relationships
            await self._update_thematic_relationships(element)

            return element

        except Exception as e:
            self.logger.error(f"Error creating thematic element: {str(e)}")
            raise

    async def analyze_themes(
        self,
        story_bible: Dict[str, Any]
    ) -> ThematicAnalysis:
        """Perform comprehensive thematic analysis."""
        try:
            # Analyze theme development
            theme_development = await self._analyze_theme_development(story_bible)

            # Analyze motif patterns
            motif_patterns = await self._analyze_motif_patterns(story_bible)

            # Analyze symbol usage
            symbol_usage = await self._analyze_symbol_usage(story_bible)

            # Assess thematic coherence
            thematic_coherence = await self._assess_thematic_coherence(
                theme_development,
                motif_patterns,
                symbol_usage,
                story_bible
            )

            # Analyze character-theme relationships
            character_themes = await self._analyze_character_themes(story_bible)

            # Analyze plot-theme relationships
            plot_themes = await self._analyze_plot_themes(story_bible)

            # Generate recommendations
            recommendations = await self._generate_thematic_recommendations(
                theme_development,
                motif_patterns,
                symbol_usage,
                thematic_coherence,
                character_themes,
                plot_themes
            )

            analysis = ThematicAnalysis(
                theme_development=theme_development,
                motif_patterns=motif_patterns,
                symbol_usage=symbol_usage,
                thematic_coherence=thematic_coherence,
                character_themes=character_themes,
                plot_themes=plot_themes,
                recommendations=recommendations
            )

            # Update state
            self.state["analysis_history"].append({
                "timestamp": datetime.utcnow(),
                "analysis": analysis
            })

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing themes: {str(e)}")
            raise

    async def track_theme_development(
        self,
        theme_id: str,
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track the development of a specific theme through the story."""
        try:
            if theme_id not in self.state["thematic_elements"]:
                raise ValueError(f"Theme {theme_id} not found")

            theme = self.state["thematic_elements"][theme_id]
            development_tracking = {
                "theme": theme.name,
                "progression": [],
                "strength": [],
                "variations": [],
                "interactions": [],
                "impact_points": []
            }

            # Track progression through story
            progression = await self._track_theme_progression(theme, story_bible)
            development_tracking["progression"] = progression

            # Analyze theme strength at different points
            strength_analysis = await self._analyze_theme_strength(
                theme,
                progression,
                story_bible
            )
            development_tracking["strength"] = strength_analysis

            # Track thematic variations
            variations = await self._track_thematic_variations(
                theme,
                progression,
                story_bible
            )
            development_tracking["variations"] = variations

            # Analyze theme interactions
            interactions = await self._analyze_theme_interactions(
                theme,
                story_bible
            )
            development_tracking["interactions"] = interactions

            # Identify key impact points
            impact_points = await self._identify_theme_impact_points(
                theme,
                progression,
                story_bible
            )
            development_tracking["impact_points"] = impact_points

            return development_tracking

        except Exception as e:
            self.logger.error(f"Error tracking theme development: {str(e)}")
            raise

    async def _analyze_potential_manifestations(
        self,
        element_type: str,
        name: str,
        description: str,
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze how a thematic element could manifest in the story."""
        try:
            manifestations = []
            
            # Generate manifestation ideas based on element type
            prompt = f"""
            Analyze potential manifestations for this thematic element:
            Type: {element_type}
            Name: {name}
            Description: {description}

            Consider manifestations in:
            1. Character actions and decisions
            2. Dialogue and internal monologue
            3. Setting and atmosphere
            4. Plot events and turning points
            5. Imagery and symbolism
            
            Return a list of specific manifestation opportunities.
            """

            response = await self.llm.agenerate([prompt])
            manifestation_ideas = self._parse_manifestation_response(
                response.generations[0].text
            )

            # Validate manifestations against story context
            for idea in manifestation_ideas:
                if await self._validate_manifestation(idea, story_bible):
                    manifestations.append(idea)

            return manifestations

        except Exception as e:
            self.logger.error(f"Error analyzing potential manifestations: {str(e)}")
            return []

    def _parse_manifestation_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured manifestation ideas."""
        try:
            import json
            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
                
            return json.loads(cleaned_response)

        except Exception as e:
            self.logger.error(f"Error parsing manifestation response: {str(e)}")
            return []

    async def _validate_manifestation(
        self,
        manifestation: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> bool:
        """Validate if a manifestation fits within the story context."""
        try:
            # Check character availability
            if "characters" in manifestation:
                characters = set(manifestation["characters"])
                available_characters = set(story_bible.get("characters", {}).keys())
                if not characters.issubset(available_characters):
                    return False

            # Check location availability
            if "location" in manifestation:
                if manifestation["location"] not in story_bible.get("locations", {}):
                    return False

            # Check timeline consistency
            if "timeline_position" in manifestation:
                if not self._is_timeline_position_valid(
                    manifestation["timeline_position"],
                    story_bible
                ):
                    return False

            # Check thematic consistency
            if "thematic_elements" in manifestation:
                if not await self._check_thematic_consistency(
                    manifestation["thematic_elements"],
                    story_bible
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating manifestation: {str(e)}")
            return False

    def _is_timeline_position_valid(
        self,
        position: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> bool:
        """Check if a timeline position is valid within the story."""
        try:
            timeline = story_bible.get("timeline", {})
            
            # Check if position fits within story bounds
            if "absolute_time" in position:
                story_start = timeline.get("start_time")
                story_end = timeline.get("end_time")
                
                if story_start and story_end:
                    if position["absolute_time"] < story_start or \
                       position["absolute_time"] > story_end:
                        return False

            # Check for conflicts with existing events
            if "events" in timeline:
                for event in timeline["events"]:
                    if self._events_conflict(position, event):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking timeline position: {str(e)}")
            return False

    def _events_conflict(
        self,
        position1: Dict[str, Any],
        position2: Dict[str, Any]
    ) -> bool:
        """Check if two timeline positions conflict."""
        try:
            # Check absolute time conflicts
            if "absolute_time" in position1 and "absolute_time" in position2:
                return position1["absolute_time"] == position2["absolute_time"]

            # Check relative time conflicts
            if "relative_position" in position1 and "relative_position" in position2:
                return position1["relative_position"] == position2["relative_position"]

            # Check sequence conflicts
            if "sequence_number" in position1 and "sequence_number" in position2:
                return position1["sequence_number"] == position2["sequence_number"]

            return False

        except Exception as e:
            self.logger.error(f"Error checking event conflicts: {str(e)}")
            return True

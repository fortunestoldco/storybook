from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config.llm_config import LLMRouter

class NarrativeEvent(BaseModel):
    event_id: str = Field(description="Unique identifier for the event")
    type: str = Field(description="Type of event (scene, sequence, chapter, act)")
    title: str = Field(description="Title or brief description")
    content: Dict[str, Any] = Field(description="Event content and details")
    characters: List[str] = Field(description="Characters involved")
    location: str = Field(description="Location where event takes place")
    timeline_position: Dict[str, Any] = Field(description="Position in story timeline")
    story_elements: Dict[str, Any] = Field(description="Related story elements")
    narrative_functions: List[str] = Field(description="Functions this event serves")
    connections: Dict[str, Any] = Field(description="Connections to other events")

class StructureAnalysis(BaseModel):
    pacing: Dict[str, Any] = Field(description="Analysis of story pacing")
    plot_progression: Dict[str, Any] = Field(description="Plot progression analysis")
    character_arcs: Dict[str, Any] = Field(description="Character arc progression")
    theme_development: Dict[str, Any] = Field(description="Theme development analysis")
    narrative_balance: Dict[str, Any] = Field(description="Balance of narrative elements")
    recommendations: List[Dict[str, Any]] = Field(description="Improvement recommendations")

class NarrativeStructureAgent(BaseAgent):
    def __init__(self, tools_service: ToolsService):
        super().__init__(tools_service)
        self.logger = logging.getLogger(__name__)
        self.tools_service = tools_service
        self.llm_router = LLMRouter(tools_service)
        
        self.state = {
            "narrative_events": {},
            "structure_cache": {},
            "analysis_history": [],
            "last_update": None
        }
        
        self.structure_templates = {
            "three_act": self._get_three_act_template(),
            "hero_journey": self._get_hero_journey_template(),
            "five_act": self._get_five_act_template(),
            "seven_point": self._get_seven_point_template()
        }

    async def analyze_structure(self, content: str, parser: Optional[PydanticOutputParser] = None) -> Any:
        try:
            return await self.llm_router.process_with_streaming(
                task="structure_analysis",
                prompt=content,
                parser=parser
            )
        except Exception as e:
            self.logger.error(f"Error in structure analysis: {str(e)}")
            raise

    async def create_narrative_event(self, event_type: str, title: str, content: Dict[str, Any], 
                                   story_bible: Dict[str, Any]) -> NarrativeEvent:
        try:
            valid_types = ["scene", "sequence", "chapter", "act"]
            if event_type not in valid_types:
                raise ValueError(f"Invalid event type. Must be one of: {valid_types}")

            event_id = f"EVENT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            characters = await self._extract_characters(content, story_bible)
            location = await self._extract_location(content, story_bible)
            timeline_position = await self._determine_timeline_position(content, story_bible)
            story_elements = await self._identify_story_elements(content, story_bible)
            narrative_functions = await self._analyze_narrative_functions(content, event_type, story_bible)

            event = NarrativeEvent(
                event_id=event_id,
                type=event_type,
                title=title,
                content=content,
                characters=characters,
                location=location,
                timeline_position=timeline_position,
                story_elements=story_elements,
                narrative_functions=narrative_functions,
                connections={}
            )

            await self._validate_structural_consistency(event, story_bible)
            self.state["narrative_events"][event_id] = event
            self.state["last_update"] = datetime.utcnow()
            await self._update_narrative_connections(event)

            return event

        except Exception as e:
            self.logger.error(f"Error creating narrative event: {str(e)}")
            raise

    async def _extract_characters(self, content: Dict[str, Any], story_bible: Dict[str, Any]) -> List[str]:
        try:
            characters = set()
            
            if "characters" in content:
                characters.update(content["characters"])
            
            if "dialogue" in content:
                for dialogue in content["dialogue"]:
                    if "speaker" in dialogue:
                        characters.add(dialogue["speaker"])
            
            if "actions" in content:
                for action in content["actions"]:
                    if "actor" in action:
                        characters.add(action["actor"])

            valid_characters = set(story_bible.get("characters", {}).keys())
            characters = characters.intersection(valid_characters)

            return list(characters)

        except Exception as e:
            self.logger.error(f"Error extracting characters: {str(e)}")
            return []

    async def _extract_location(self, content: Dict[str, Any], story_bible: Dict[str, Any]) -> str:
        try:
            location = "unknown"
            
            if "location" in content:
                location = content["location"]
            else:
                prompt = (
                    "Extract the primary location from the following content:\n\n"
                    f"{content.get('text', '')}"
                )
                result = await self.llm_router.process_with_streaming(
                    task="analysis",
                    prompt=prompt
                )
                if isinstance(result, dict) and "location" in result:
                    location = result["location"]

            valid_locations = story_bible.get("locations", {})
            return location if location in valid_locations else "unknown"

        except Exception as e:
            self.logger.error(f"Error extracting location: {str(e)}")
            return "unknown"

    async def _determine_timeline_position(self, content: Dict[str, Any], story_bible: Dict[str, Any]) -> Dict[str, Any]:
        try:
            position = {
                "absolute_time": None,
                "relative_time": None,
                "sequence_number": len(self.state["narrative_events"]) + 1,
                "concurrent_events": []
            }

            if "timestamp" in content:
                position["absolute_time"] = content["timestamp"]

            if "relative_time" in content:
                position["relative_time"] = content["relative_time"]

            concurrent_events = await self._identify_concurrent_events(content, story_bible)
            position["concurrent_events"] = concurrent_events

            return position

        except Exception as e:
            self.logger.error(f"Error determining timeline position: {str(e)}")
            raise

    async def _identify_concurrent_events(self, content: Dict[str, Any], story_bible: Dict[str, Any]) -> List[str]:
        try:
            concurrent_events = []
            current_time = content.get("timestamp")
            
            if current_time:
                for event_id, event in self.state["narrative_events"].items():
                    event_time = event.content.get("timestamp")
                    if event_time and event_time == current_time:
                        concurrent_events.append(event_id)

            return concurrent_events

        except Exception as e:
            self.logger.error(f"Error identifying concurrent events: {str(e)}")
            return []

    async def _identify_story_elements(self, content: Dict[str, Any], story_bible: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = (
                "Analyze the following content for key story elements:\n\n"
                f"{content.get('text', '')}\n\n"
                "Include: plot points, themes, symbols, foreshadowing, and callbacks"
            )

            result = await self.llm_router.process_with_streaming(
                task="analysis",
                prompt=prompt
            )

            return {
                "plot_points": result.get("plot_points", []),
                "themes": result.get("themes", []),
                "symbols": result.get("symbols", []),
                "foreshadowing": result.get("foreshadowing", []),
                "callbacks": result.get("callbacks", [])
            }

        except Exception as e:
            self.logger.error(f"Error identifying story elements: {str(e)}")
            raise

    async def _analyze_narrative_functions(self, content: Dict[str, Any], event_type: str, 
                                        story_bible: Dict[str, Any]) -> List[str]:
        try:
            prompt = (
                "Analyze the narrative functions of the following content:\n\n"
                f"Type: {event_type}\n"
                f"Content: {content.get('text', '')}\n\n"
                "Consider: plot advancement, character development, world-building, "
                "theme reinforcement, and tension management"
            )

            result = await self.llm_router.process_with_streaming(
                                task="analysis",
                prompt=prompt
            )

            functions = []
            if result.get("plot_advancement", False):
                functions.append("plot_advancement")
            if result.get("character_development", False):
                functions.append("character_development")
            if result.get("world_building", False):
                functions.append("world_building")
            if result.get("theme_reinforcement", False):
                functions.append("theme_reinforcement")
            if result.get("tension_management", False):
                functions.append("tension_management")

            return functions

        except Exception as e:
            self.logger.error(f"Error analyzing narrative functions: {str(e)}")
            return []

    async def _validate_structural_consistency(self, event: NarrativeEvent, story_bible: Dict[str, Any]) -> None:
        try:
            await self._check_event_sequence(event)
            await self._check_character_consistency(event, story_bible)
            await self._check_plot_progression(event, story_bible)
            await self._check_timeline_consistency(event)
            await self._check_thematic_consistency(event, story_bible)
        except Exception as e:
            self.logger.error(f"Error validating structural consistency: {str(e)}")
            raise

    async def _check_event_sequence(self, event: NarrativeEvent) -> None:
        try:
            previous_events = [e for e in self.state["narrative_events"].values() 
                             if e.timeline_position["sequence_number"] < event.timeline_position["sequence_number"]]
            
            if previous_events:
                latest_event = max(previous_events, key=lambda x: x.timeline_position["sequence_number"])
                if not self._events_are_compatible(latest_event, event):
                    raise ValueError("Event sequence inconsistency detected")
        except Exception as e:
            self.logger.error(f"Error checking event sequence: {str(e)}")
            raise

    async def _check_character_consistency(self, event: NarrativeEvent, story_bible: Dict[str, Any]) -> None:
        try:
            valid_characters = set(story_bible.get("characters", {}).keys())
            event_characters = set(event.characters)
            
            if not event_characters.issubset(valid_characters):
                invalid_chars = event_characters - valid_characters
                raise ValueError(f"Invalid characters detected: {invalid_chars}")
                
            for char in event_characters:
                char_data = story_bible["characters"][char]
                if not self._character_action_is_consistent(char, event.content, char_data):
                    raise ValueError(f"Character consistency error for {char}")
        except Exception as e:
            self.logger.error(f"Error checking character consistency: {str(e)}")
            raise

    async def _check_plot_progression(self, event: NarrativeEvent, story_bible: Dict[str, Any]) -> None:
        try:
            plot_points = story_bible.get("plot_points", {})
            current_plot_point = event.story_elements.get("plot_points", [])
            
            if current_plot_point:
                for point in current_plot_point:
                    if point in plot_points and not self._plot_point_is_valid(point, event, plot_points):
                        raise ValueError(f"Plot progression error at point: {point}")
        except Exception as e:
            self.logger.error(f"Error checking plot progression: {str(e)}")
            raise

    async def _check_timeline_consistency(self, event: NarrativeEvent) -> None:
        try:
            all_events = self.state["narrative_events"].values()
            timeline_issues = []
            
            for other_event in all_events:
                if other_event.event_id != event.event_id:
                    if self._events_have_temporal_conflict(event, other_event):
                        timeline_issues.append(other_event.event_id)
                        
            if timeline_issues:
                raise ValueError(f"Timeline consistency issues with events: {timeline_issues}")
        except Exception as e:
            self.logger.error(f"Error checking timeline consistency: {str(e)}")
            raise

    async def _check_thematic_consistency(self, event: NarrativeEvent, story_bible: Dict[str, Any]) -> None:
        try:
            story_themes = set(story_bible.get("themes", {}).keys())
            event_themes = set(event.story_elements.get("themes", []))
            
            if not event_themes.issubset(story_themes):
                invalid_themes = event_themes - story_themes
                raise ValueError(f"Invalid themes detected: {invalid_themes}")
                
            for theme in event_themes:
                if not self._theme_development_is_consistent(theme, event, story_bible):
                    raise ValueError(f"Theme consistency error for {theme}")
        except Exception as e:
            self.logger.error(f"Error checking thematic consistency: {str(e)}")
            raise

    async def _update_narrative_connections(self, event: NarrativeEvent) -> None:
        try:
            connections = {
                "previous_events": [],
                "next_events": [],
                "parallel_events": [],
                "thematic_connections": [],
                "causal_connections": []
            }
            
            temporal_connections = await self._find_temporal_connections(event)
            connections.update(temporal_connections)
            
            thematic_connections = await self._find_thematic_connections(event)
            connections["thematic_connections"] = thematic_connections
            
            causal_connections = await self._find_causal_connections(event)
            connections["causal_connections"] = causal_connections
            
            event.connections = connections
            await self._update_connected_events(event)
            
        except Exception as e:
            self.logger.error(f"Error updating narrative connections: {str(e)}")
            raise

    def _get_three_act_template(self) -> Dict[str, Any]:
        return {
            "name": "three_act",
            "acts": [
                {
                    "name": "Setup",
                    "required_elements": [
                        "introduction",
                        "ordinary_world", 
                        "inciting_incident",
                        "call_to_action"
                    ],
                    "target_length": 0.25
                },
                {
                    "name": "Confrontation",
                    "required_elements": [
                        "rising_action",
                        "complications",
                        "midpoint",
                        "mounting_pressure"
                    ],
                    "target_length": 0.5
                },
                {
                    "name": "Resolution",
                    "required_elements": [
                        "climax",
                        "falling_action", 
                        "resolution",
                        "denouement"
                    ],
                    "target_length": 0.25
                }
            ],
            "key_points": [
                "inciting_incident",
                "first_plot_point",
                "midpoint", 
                "second_plot_point",
                "climax"
            ]
        }

    def _get_hero_journey_template(self) -> Dict[str, Any]:
        return {
            "name": "hero_journey",
            "stages": [
                {
                    "name": "Ordinary World",
                    "required_elements": [
                        "hero_introduction",
                        "world_establishment",
                        "status_quo"
                    ]
                },
                {
                    "name": "Call to Adventure",
                    "required_elements": [
                        "inciting_incident",
                        "hero_reluctance",
                        "stakes_establishment"
                    ]
                },
                {
                    "name": "Crossing the Threshold",
                    "required_elements": [
                        "world_change",
                        "point_of_no_return",
                        "first_challenge"  
                    ]
                },
                {
                    "name": "Tests & Allies",
                    "required_elements": [
                        "minor_challenges",
                        "ally_introductions",
                        "skill_development"
                    ]
                },
                {
                    "name": "Approach & Ordeal",
                    "required_elements": [
                        "major_crisis",
                        "apparent_defeat",
                        "inner_revelation"
                    ]
                },
                {
                    "name": "Return",
                    "required_elements": [
                        "final_battle",
                        "transformation",
                        "return_changed"
                    ]
                }
            ]
        }

    def _get_five_act_template(self) -> Dict[str, Any]:
        return {
            "name": "five_act",
            "acts": [
                {
                    "name": "Exposition",
                    "required_elements": [
                        "setting_establishment",
                        "character_introduction",
                        "initial_conflict"
                    ],
                    "target_length": 0.15
                },
                {
                    "name": "Rising Action",
                    "required_elements": [
                        "conflict_development",
                        "complications",
                        "subplot_introduction"
                    ],
                    "target_length": 0.25
                },
                {
                    "name": "Climax",
                    "required_elements": [
                        "crisis_point",
                        "major_confrontation",
                        "turning_point"
                    ],
                    "target_length": 0.25
                },
                {
                    "name": "Falling Action",
                    "required_elements": [
                        "consequence_revelation",
                        "subplot_resolution",
                        "character_transformation"
                    ],
                    "target_length": 0.20
                },
                {
                    "name": "Resolution",
                    "required_elements": [
                        "conflict_resolution",
                        "character_closure",
                        "final_image"
                    ],
                    "target_length": 0.15
                }
            ]
        }

    def _get_seven_point_template(self) -> Dict[str, Any]:
        return {
            "name": "seven_point",
            "points": [
                {
                    "name": "Hook",
                    "required_elements": [
                        "character_introduction",
                        "world_establishment",
                        "initial_situation"
                    ]
                },
                {
                    "name": "Plot Turn 1",
                    "required_elements": [
                        "inciting_incident",
                        "direction_change",
                        "goal_establishment"
                    ]
                },
                {
                    "name": "Pinch Point 1",
                    "required_elements": [
                        "antagonist_power",
                        "pressure_increase",
                        "stakes_revelation"
                    ]
                },
                {
                    "name": "Midpoint",
                    "required_elements": [
                        "character_shift",
                        "reactive_to_active",
                        "raised_stakes"
                    ]
                },
                {
                    "name": "Pinch Point 2",
                    "required_elements": [
                        "major_setback",
                        "hope_loss",
                        "darkness_moment"
                    ]
                },
                {
                    "name": "Plot Turn 2",
                    "required_elements": [
                        "final_piece",
                        "resolution_key",
                        "preparation_completion"
                    ]
                },
                {
                    "name": "Resolution",
                    "required_elements": [
                        "final_battle",
                        "character_victory",
                        "theme_proof"
                    ]
                }
            ]
        }

    def _events_are_compatible(self, event1: NarrativeEvent, event2: NarrativeEvent) -> bool:
        if not event1 or not event2:
            return True
            
        time1 = event1.timeline_position.get("absolute_time")
        time2 = event2.timeline_position.get("absolute_time")
        
        if time1 and time2 and time1 > time2:
            return False
            
        return True

    def _character_action_is_consistent(self, character: str, content: Dict[str, Any], 
                                     character_data: Dict[str, Any]) -> bool:
        # Basic implementation - expand based on character consistency rules
        return True

    def _plot_point_is_valid(self, point: str, event: NarrativeEvent, 
                            plot_points: Dict[str, Any]) -> bool:
        # Basic implementation - expand based on plot progression rules
        return True

    def _events_have_temporal_conflict(self, event1: NarrativeEvent, event2: NarrativeEvent) -> bool:
        time1 = event1.timeline_position.get("absolute_time")
        time2 = event2.timeline_position.get("absolute_time")
        
        if time1 and time2:
            if time1 == time2 and event1.location == event2.location:
                chars1 = set(event1.characters)
                chars2 = set(event2.characters)
                if chars1.intersection(chars2):
                    return True
        
        return False

    def _theme_development_is_consistent(self, theme: str, event: NarrativeEvent, 
                                       story_bible: Dict[str, Any]) -> bool:
        # Basic implementation - expand based on thematic consistency rules
        return True

    async def _find_temporal_connections(self, event: NarrativeEvent) -> Dict[str, List[str]]:
        connections = {
            "previous_events": [],
            "next_events": [],
            "parallel_events": []
        }
        
        for other_id, other_event in self.state["narrative_events"].items():
            if other_event.event_id == event.event_id:
                continue
                
            if self._events_are_temporally_connected(event, other_event):
                if other_event.timeline_position["sequence_number"] < event.timeline_position["sequence_number"]:
                    connections["previous_events"].append(other_id)
                else:
                    connections["next_events"].append(other_id)
                    
            if self._events_are_parallel(event, other_event):
                connections["parallel_events"].append(other_id)
                
        return connections

    async def _find_thematic_connections(self, event: NarrativeEvent) -> List[str]:
        connected_events = []
        event_themes = set(event.story_elements.get("themes", []))
        
        for other_id, other_event in self.state["narrative_events"].items():
            if other_event.event_id == event.event_id:
                continue
                
            other_themes = set(other_event.story_elements.get("themes", []))
            if event_themes.intersection(other_themes):
                connected_events.append(other_id)
                
        return connected_events

    async def _find_causal_connections(self, event: NarrativeEvent) -> List[str]:
        connected_events = []
        
        for other_id, other_event in self.state["narrative_events"].items():
            if other_event.event_id == event.event_id:
                continue
                
            if self._events_are_causally_connected(event, other_event):
                connected_events.append(other_id)
                
        return connected_events

    async def _update_connected_events(self, event: NarrativeEvent) -> None:
        for event_id in event.connections.get("previous_events", []):
            if event_id in self.state["narrative_events"]:
                connected_event = self.state["narrative_events"][event_id]
                if event.event_id not in connected_event.connections.get("next_events", []):
                    connected_event.connections.setdefault("next_events", []).append(event.event_id)

                for event_id in event.connections.get("next_events", []):
                    if event_id in self.state["narrative_events"]:
                        connected_event = self.state["narrative_events"][event_id]
                        if event.event_id not in connected_event.connections.get("previous_events", []):
                            connected_event.connections.setdefault("previous_events", []).append(event.event_id)

        for event_id in event.connections.get("parallel_events", []):
            if event_id in self.state["narrative_events"]:
                connected_event = self.state["narrative_events"][event_id]
                if event.event_id not in connected_event.connections.get("parallel_events", []):
                    connected_event.connections.setdefault("parallel_events", []).append(event.event_id)

        for event_id in event.connections.get("thematic_connections", []):
            if event_id in self.state["narrative_events"]:
                connected_event = self.state["narrative_events"][event_id]
                if event.event_id not in connected_event.connections.get("thematic_connections", []):
                    connected_event.connections.setdefault("thematic_connections", []).append(event.event_id)

        for event_id in event.connections.get("causal_connections", []):
            if event_id in self.state["narrative_events"]:
                connected_event = self.state["narrative_events"][event_id]
                if event.event_id not in connected_event.connections.get("causal_connections", []):
                    connected_event.connections.setdefault("causal_connections", []).append(event.event_id)

    def _events_are_temporally_connected(self, event1: NarrativeEvent, event2: NarrativeEvent) -> bool:
        if (abs(event1.timeline_position["sequence_number"] - 
                event2.timeline_position["sequence_number"]) == 1):
            return True
            
        time1 = event1.timeline_position.get("absolute_time")
        time2 = event2.timeline_position.get("absolute_time")
        
        if time1 and time2:
            # Check if events are temporally adjacent within a reasonable threshold
            return abs((time1 - time2).total_seconds()) < 3600  # 1 hour threshold
            
        return False

    def _events_are_parallel(self, event1: NarrativeEvent, event2: NarrativeEvent) -> bool:
        time1 = event1.timeline_position.get("absolute_time")
        time2 = event2.timeline_position.get("absolute_time")
        
        if time1 and time2 and time1 == time2:
            return True
            
        return event1.timeline_position.get("relative_time") == event2.timeline_position.get("relative_time")

    def _events_are_causally_connected(self, event1: NarrativeEvent, event2: NarrativeEvent) -> bool:
        # Check if event2's plot points or story elements depend on event1's outcomes
        event1_outcomes = set(event1.story_elements.get("plot_points", []))
        event2_prerequisites = set(event2.content.get("prerequisites", []))
        
        return bool(event1_outcomes.intersection(event2_prerequisites))
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from difflib import SequenceMatcher

class ContinuityItem(BaseModel):
    """Model for a continuity element being tracked."""
    element_type: str = Field(description="Type of continuity element (e.g., timeline, character, object, plot)")
    name: str = Field(description="Name or identifier of the element")
    first_mention: Dict[str, Any] = Field(description="Details of first mention including chapter and context")
    occurrences: List[Dict[str, Any]] = Field(description="List of subsequent occurrences")
    attributes: Dict[str, Any] = Field(description="Tracked attributes of the element")
    status: str = Field(description="Current status of the element (active/resolved/inconsistent)")

class ContinuityReport(BaseModel):
    """Model for the continuity check report."""
    timeline_issues: List[Dict[str, Any]] = Field(description="Issues with chronological continuity")
    character_issues: List[Dict[str, Any]] = Field(description="Issues with character continuity")
    plot_issues: List[Dict[str, Any]] = Field(description="Issues with plot continuity")
    setting_issues: List[Dict[str, Any]] = Field(description="Issues with setting continuity")
    object_issues: List[Dict[str, Any]] = Field(description="Issues with object continuity")
    recommendations: List[Dict[str, Any]] = Field(description="Recommendations for fixing continuity issues")

class ContinuityCheckerAgent(BaseAgent):
    """Agent responsible for checking narrative continuity throughout the manuscript."""
    
    def __init__(self, tools_service: ToolsService):
        super().__init__(tools_service)
        self.logger = logging.getLogger(__name__)
        self.tools_service = tools_service
        
        try:
            self.llm = Ollama(
                base_url=tools_service.config.get('ollama_base_url'),
                model="mixtral"
            )
        except Exception as e:
            self.logger.error(f"Error initializing Ollama: {str(e)}")
            raise

        self.state = {
            "current_manuscript": None,
            "tracked_elements": {},
            "last_check": None
        }

    async def check_manuscript_continuity(
        self, 
        manuscript: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> ContinuityReport:
        """Main method to check manuscript continuity."""
        try:
            self.state["current_manuscript"] = manuscript.get("id")
            self.state["last_check"] = datetime.utcnow()

            # Initialize report
            report = ContinuityReport(
                timeline_issues=[],
                character_issues=[],
                plot_issues=[],
                setting_issues=[],
                object_issues=[],
                recommendations=[]
            )

            # Build tracking database
            await self._build_tracking_database(manuscript)

            # Run continuity checks
            timeline_issues = await self._check_timeline_continuity(manuscript, story_bible)
            character_issues = await self._check_character_continuity(manuscript, story_bible)
            plot_issues = await self._check_plot_continuity(manuscript, story_bible)
            setting_issues = await self._check_setting_continuity(manuscript, story_bible)
            object_issues = await self._check_object_continuity(manuscript, story_bible)

            # Update report
            report.timeline_issues.extend(timeline_issues)
            report.character_issues.extend(character_issues)
            report.plot_issues.extend(plot_issues)
            report.setting_issues.extend(setting_issues)
            report.object_issues.extend(object_issues)

            # Generate recommendations
            report.recommendations = await self._generate_recommendations(report)

            return report

        except Exception as e:
            self.logger.error(f"Error in manuscript continuity check: {str(e)}")
            raise

    async def _build_tracking_database(self, manuscript: Dict[str, Any]) -> None:
        """Build initial database of tracked elements."""
        self.state["tracked_elements"] = {
            "timeline": [],
            "characters": {},
            "settings": {},
            "objects": {},
            "plot_points": []
        }

        for chapter_num, chapter in enumerate(manuscript.get("chapters", []), 1):
            await self._process_chapter_elements(chapter, chapter_num)

    async def _process_chapter_elements(
        self, 
        chapter: Dict[str, Any],
        chapter_num: int
    ) -> None:
        """Process and track elements from a single chapter."""
        content = chapter.get("content", "")
        
        # Extract and track timeline elements
        timeline_elements = await self._extract_temporal_markers(content)
        for element in timeline_elements:
            self.state["tracked_elements"]["timeline"].append({
                "chapter": chapter_num,
                "element": element
            })

        # Extract and track character mentions
        character_mentions = await self._extract_character_mentions(content)
        for character, details in character_mentions.items():
            if character not in self.state["tracked_elements"]["characters"]:
                self.state["tracked_elements"]["characters"][character] = []
            self.state["tracked_elements"]["characters"][character].append({
                "chapter": chapter_num,
                "details": details
            })

        # Extract and track settings
        settings = await self._extract_settings(content)
        for setting in settings:
            if setting["name"] not in self.state["tracked_elements"]["settings"]:
                self.state["tracked_elements"]["settings"][setting["name"]] = []
            self.state["tracked_elements"]["settings"][setting["name"]].append({
                "chapter": chapter_num,
                "details": setting
            })

    async def _extract_temporal_markers(self, content: str) -> List[Dict[str, Any]]:
        """Extract temporal markers from text using LLM."""
        response = await self.llm.agenerate([
            f"Extract all temporal markers (dates, times, seasons, etc.) from the following text: {content}"
        ])
        return self._process_temporal_markers(response.generations[0].text)

    async def _extract_character_mentions(self, content: str) -> Dict[str, Any]:
        """Extract character mentions and their context from text."""
        response = await self.llm.agenerate([
            f"Extract all character mentions and their context from the following text: {content}"
        ])
        # Process the response into structured data
        try:
            mentions = {}
            lines = response.generations[0].text.strip().split('\n')
            current_character = None
            current_details = {}
            
            for line in lines:
                if line.startswith('Character:'):
                    if current_character:
                        mentions[current_character] = current_details
                    current_character = line.split(':', 1)[1].strip()
                    current_details = {}
                elif ':' in line and current_character:
                    key, value = line.split(':', 1)
                    current_details[key.strip()] = value.strip()
                    
            if current_character:
                mentions[current_character] = current_details
                
            return mentions
        except Exception as e:
            self.logger.error(f"Error processing character mentions: {str(e)}")
            return {}

    async def _extract_settings(self, content: str) -> List[Dict[str, Any]]:
        """Extract settings and their descriptions from text."""
        response = await self.llm.agenerate([
            f"Extract all settings and their descriptions from the following text: {content}"
        ])
        try:
            settings = []
            lines = response.generations[0].text.strip().split('\n')
            current_setting = {}
            
            for line in lines:
                if line.startswith('Setting:'):
                    if current_setting:
                        settings.append(current_setting)
                    current_setting = {"name": line.split(':', 1)[1].strip()}
                elif ':' in line and current_setting:
                    key, value = line.split(':', 1)
                    current_setting[key.strip()] = value.strip()
                    
            if current_setting:
                settings.append(current_setting)
                
            return settings
        except Exception as e:
            self.logger.error(f"Error processing settings: {str(e)}")
            return []

    def _process_temporal_markers(self, llm_response: str) -> List[Dict[str, Any]]:
        """Process LLM response into structured temporal markers."""
        markers = []
        try:
            lines = llm_response.strip().split('\n')
            
            current_marker = {}
            for line in lines:
                line = line.strip()
                if not line:
                    if current_marker:
                        markers.append(current_marker.copy())
                        current_marker = {}
                    continue
                    
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip('- ').lower()
                    value = value.strip()
                    
                    if key == 'time marker':
                        current_marker['marker'] = value
                    elif key == 'type':
                        current_marker['type'] = value.lower()
                    elif key == 'context':
                        current_marker['context'] = value
                        
            if current_marker:
                markers.append(current_marker)
                
            # Validate and structure markers
            validated_markers = []
            for marker in markers:
                if 'marker' in marker and 'type' in marker:
                    validated_marker = {
                        'marker': marker['marker'],
                        'type': marker['type'],
                        'context': marker.get('context', ''),
                        'normalized_value': self._normalize_temporal_marker(marker['marker']),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    validated_markers.append(validated_marker)
            
            return validated_markers
            
        except Exception as e:
            self.logger.error(f"Error processing temporal markers: {str(e)}")
            return []

    def _normalize_temporal_marker(self, marker: str) -> Dict[str, Any]:
        """Convert temporal marker to normalized format."""
        normalized = {
            'type': None,
            'value': None,
            'unit': None,
            'reference_point': None
        }
        
        try:
            # Check for absolute dates/times
            for format in [
                '%Y-%m-%d',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y-%m-%d %H:%M:%S',
                '%I:%M %p',
                '%H:%M'
            ]:
                try:
                    parsed_time = datetime.strptime(marker, format)
                    normalized.update({
                        'type': 'absolute',
                        'value': parsed_time.isoformat(),
                        'unit': 'datetime'
                    })
                    return normalized
                except ValueError:
                    continue
            
            # Check for relative time markers
            import re
            relative_patterns = {
                r'(\d+)\s*(day|week|month|year)s?\s*(ago|later|after)': {
                    'unit': lambda x: x[1],
                    'value': lambda x: int(x[0]),
                    'reference': lambda x: x[2]
                },
                r'(yesterday|tomorrow|today)': {
                    'unit': lambda x: 'day',
                    'value': lambda x: -1 if x[0] == 'yesterday' else 1 if x[0] == 'tomorrow' else 0,
                    'reference': lambda x: 'relative'
                },
                r'(last|next)\s*(week|month|year)': {
                    'unit': lambda x: x[1],
                    'value': lambda x: -1 if x[0] == 'last' else 1,
                    'reference': lambda x: 'relative'
                }
            }
            
            for pattern, extractors in relative_patterns.items():
                match = re.search(pattern, marker.lower())
                if match:
                    normalized.update({
                        'type': 'relative',
                        'unit': extractors['unit'](match.groups()),
                        'value': extractors['value'](match.groups()),
                        'reference_point': extractors['reference'](match.groups())
                    })
                    return normalized
            
            # Check for seasonal/time-of-day markers
            seasonal_patterns = {
                'season': r'(spring|summer|autumn|fall|winter)',
                'time_of_day': r'(morning|afternoon|evening|night|dawn|dusk|midnight|noon)'
            }
            
            for marker_type, pattern in seasonal_patterns.items():
                match = re.search(pattern, marker.lower())
                if match:
                    normalized.update({
                        'type': 'cyclical',
                        'unit': marker_type,
                        'value': match.group(1)
                    })
                    return normalized
                    
            # If no patterns match, store as unstructured
            normalized.update({
                'type': 'unstructured',
                'value': marker,
                'unit': 'unknown'
            })
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing temporal marker: {str(e)}")
            return normalized

    async def _check_timeline_continuity(
        self,
        manuscript: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check chronological continuity."""
        issues = []
        timeline = self.state["tracked_elements"]["timeline"]

        for i in range(1, len(timeline)):
            current = timeline[i]
            previous = timeline[i-1]
            
            if not self._is_temporally_consistent(previous, current):
                issues.append({
                    "type": "timeline_inconsistency",
                    "chapter_range": [previous["chapter"], current["chapter"]],
                    "description": "Temporal inconsistency detected",
                    "details": {
                        "previous_event": previous["element"],
                        "current_event": current["element"]
                    }
                })

        return issues

    def _is_temporally_consistent(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any]
    ) -> bool:
        """Check if two temporal elements are consistent."""
        try:
            prev_marker = previous['element']
            curr_marker = current['element']
            
            # If either marker is unstructured, we can't verify consistency
            if prev_marker['normalized_value']['type'] == 'unstructured' or \
               curr_marker['normalized_value']['type'] == 'unstructured':
                return True
                
            # Check absolute times
            if prev_marker['normalized_value']['type'] == 'absolute' and \
               curr_marker['normalized_value']['type'] == 'absolute':
                prev_time = datetime.fromisoformat(prev_marker['normalized_value']['value'])
                curr_time = datetime.fromisoformat(curr_marker['normalized_value']['value'])
                return prev_time <= curr_time
                
            # Check relative times
            if prev_marker['normalized_value']['type'] == 'relative' and \
               curr_marker['normalized_value']['type'] == 'relative':
                if prev_marker['normalized_value']['reference_point'] == \
                   curr_marker['normalized_value']['reference_point']:
                    prev_value = prev_marker['normalized_value']['value']
                    curr_value = curr_marker['normalized_value']['value']
                    return prev_value <= curr_value
                    
            # Check cyclical markers
            if prev_marker['normalized_value']['type'] == 'cyclical' and \
               curr_marker['normalized_value']['type'] == 'cyclical':
                if prev_marker['normalized_value']['unit'] == \
                   curr_marker['normalized_value']['unit']:
                    # Define order for cyclical events
                    cyclical_order = {
                        'time_of_day': [
                            'dawn', 'morning', 'noon', 'afternoon',
                            'evening', 'dusk', 'night', 'midnight'
                        ],
                        'season': [
                            'spring', 'summer', 'autumn', 'fall', 'winter'
                        ]
                    }
                    
                    unit = prev_marker['normalized_value']['unit']
                    if unit in cyclical_order:
                        order = cyclical_order[unit]
                        prev_idx = order.index(prev_marker['normalized_value']['value'])
                        curr_idx = order.index(curr_marker['normalized_value']['value'])
                        return prev_idx <= curr_idx
                        
            # If we can't definitively say it's inconsistent, return True
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking temporal consistency: {str(e)}")
            return True

    async def _check_character_continuity(
        self,
        manuscript: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check character continuity throughout the manuscript."""
        issues = []
        characters = self.state["tracked_elements"]["characters"]
        
        for character_name, appearances in characters.items():
            character_bible = story_bible.get("characters", {}).get(character_name, {})
            
            for i in range(1, len(appearances)):
                prev_appearance = appearances[i-1]
                curr_appearance = appearances[i]
                
                inconsistencies = await self._check_character_consistency(
                    character_bible,
                    prev_appearance["details"],
                    curr_appearance["details"]
                )
                
                if inconsistencies:
                    issues.append({
                        "type": "character_inconsistency",
                        "character": character_name,
                        "chapter_range": [
                            prev_appearance["chapter"],
                            curr_appearance["chapter"]
                        ],
                        "inconsistencies": inconsistencies
                    })
                    
        return issues

    async def _check_character_consistency(
        self,
        character_bible: Dict[str, Any],
        previous_details: Dict[str, Any],
        current_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency of character details."""
        inconsistencies = []
        
        # Check physical descriptions
        physical_inconsistencies = await self._check_physical_consistency(
            character_bible.get('physical_description', {}),
            previous_details.get('physical_description', {}),
            current_details.get('physical_description', {})
        )
        if physical_inconsistencies:
            inconsistencies.extend(physical_inconsistencies)
        
        # Check personality traits
        personality_inconsistencies = await self._check_personality_consistency(
            character_bible.get('personality', {}),
            previous_details.get('personality', {}),
            current_details.get('personality', {})
        )
        if personality_inconsistencies:
            inconsistencies.extend(personality_inconsistencies)
        
        # Check knowledge and abilities
        knowledge_inconsistencies = await self._check_knowledge_consistency(
            character_bible.get('knowledge_abilities', {}),
            previous_details.get('demonstrated_knowledge', {}),
            current_details.get('demonstrated_knowledge', {})
        )
        if knowledge_inconsistencies:
            inconsistencies.extend(knowledge_inconsistencies)
        
        # Check relationships
        relationship_inconsistencies = await self._check_relationship_consistency(
            character_bible.get('relationships', {}),
            previous_details.get('interactions', {}),
            current_details.get('interactions', {})
        )
        if relationship_inconsistencies:
            inconsistencies.extend(relationship_inconsistencies)
            
        return inconsistencies

    async def _check_physical_consistency(
        self,
        bible_description: Dict[str, Any],
        previous_description: Dict[str, Any],
        current_description: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency of physical descriptions."""
        inconsistencies = []
        
        physical_attributes = [
            'height', 'build', 'hair', 'eyes', 'skin', 
            'distinguishing_features', 'age', 'clothing'
        ]
        
        for attr in physical_attributes:
            bible_value = bible_description.get(attr)
            prev_value = previous_description.get(attr)
            curr_value = current_description.get(attr)
            
            if bible_value and curr_value and not self._is_description_consistent(bible_value, curr_value):
                inconsistencies.append({
                    'type': 'physical_inconsistency',
                    'attribute': attr,
                    'bible_value': bible_value,
                    'current_value': curr_value,
                    'severity': 'high'
                })
            
            if prev_value and curr_value and not self._is_description_consistent(prev_value, curr_value):
                inconsistencies.append({
                    'type': 'physical_inconsistency',
                    'attribute': attr,
                    'previous_value': prev_value,
                    'current_value': curr_value,
                    'severity': 'medium'
                })
                
        return inconsistencies

    async def _check_personality_consistency(
        self,
        bible_personality: Dict[str, Any],
        previous_personality: Dict[str, Any],
        current_personality: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency of personality traits and behaviors."""
        inconsistencies = []
        
        personality_aspects = [
            'traits', 'motivations', 'fears', 'desires',
            'habits', 'speech_patterns', 'behavioral_patterns'
        ]
        
        for aspect in personality_aspects:
            bible_value = bible_personality.get(aspect, [])
            prev_value = previous_personality.get(aspect, [])
            curr_value = current_personality.get(aspect, [])
            
            contradictions = self._find_personality_contradictions(bible_value, curr_value)
            if contradictions:
                inconsistencies.append({
                    'type': 'personality_inconsistency',
                    'aspect': aspect,
                    'contradictions': contradictions,
                    'severity': 'high'
                })
            
            shifts = self._find_personality_shifts(prev_value, curr_value)
            if shifts:
                inconsistencies.append({
                    'type': 'personality_shift',
                    'aspect': aspect,
                    'shifts': shifts,
                    'severity': 'medium'
                })
                
        return inconsistencies

    def _find_personality_contradictions(
        self,
        bible_traits: List[str],
        current_traits: List[str]
    ) -> List[Dict[str, Any]]:
        """Find contradictions between established and current personality traits."""
        contradictions = []
        
        opposing_traits = {
            'outgoing': ['shy', 'introverted', 'reserved'],
            'honest': ['dishonest', 'deceptive', 'manipulative'],
            'brave': ['cowardly', 'fearful', 'timid'],
            'kind': ['cruel', 'mean', 'harsh'],
            'patient': ['impatient', 'hasty', 'impulsive'],
            'loyal': ['disloyal', 'treacherous', 'unfaithful'],
            'optimistic': ['pessimistic', 'cynical', 'negative'],
            'confident': ['insecure', 'self-doubting', 'timid']
        }
        
        for bible_trait in bible_traits:
            for curr_trait in current_traits:
                if bible_trait in opposing_traits:
                    if curr_trait in opposing_traits[bible_trait]:
                        contradictions.append({
                            'bible_trait': bible_trait,
                            'current_trait': curr_trait,
                            'type': 'opposite_traits'
                        })
                
        return contradictions

    def _find_personality_shifts(
        self,
        previous_traits: List[str],
        current_traits: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify significant shifts in personality traits."""
        shifts = []
        
        prev_set = set(previous_traits)
        curr_set = set(current_traits)
        
        new_traits = curr_set - prev_set
        removed_traits = prev_set - curr_set
        
        if new_traits:
            shifts.append({
                'type': 'new_traits',
                'traits': list(new_traits)
            })
            
        if removed_traits:
            shifts.append({
                'type': 'removed_traits',
                'traits': list(removed_traits)
            })
            
        return shifts

    async def _check_knowledge_consistency(
        self,
        bible_knowledge: Dict[str, Any],
        previous_knowledge: Dict[str, Any],
        current_knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency of character knowledge and abilities."""
        inconsistencies = []
        
        knowledge_categories = [
            'skills', 'education', 'experiences',
            'expertise', 'languages', 'abilities'
        ]
        
        for category in knowledge_categories:
            bible_value = bible_knowledge.get(category, {})
            prev_value = previous_knowledge.get(category, {})
            curr_value = current_knowledge.get(category, {})
            
            knowledge_contradictions = self._find_knowledge_contradictions(
                bible_value,
                curr_value
            )
            if knowledge_contradictions:
                inconsistencies.append({
                    'type': 'knowledge_inconsistency',
                    'category': category,
                    'contradictions': knowledge_contradictions,
                    'severity': 'high'
                })
            
            unexplained_knowledge = self._find_unexplained_knowledge(
                prev_value,
                curr_value
            )
            if unexplained_knowledge:
                inconsistencies.append({
                    'type': 'unexplained_knowledge',
                    'category': category,
                    'new_knowledge': unexplained_knowledge,
                    'severity': 'medium'
                })
                
        return inconsistencies

    def _find_knowledge_contradictions(
        self,
        bible_knowledge: Dict[str, Any],
        current_knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find contradictions in character knowledge and abilities."""
        contradictions = []
        
        for skill, level in current_knowledge.items():
            bible_level = bible_knowledge.get(skill)
            if bible_level:
                if isinstance(level, (int, float)) and isinstance(bible_level, (int, float)):
                    if level > bible_level:
                        contradictions.append({
                            'skill': skill,
                            'bible_level': bible_level,
                            'current_level': level,
                            'type': 'unexpected_skill_improvement'
                        })
                elif level != bible_level:
                    contradictions.append({
                        'skill': skill,
                        'bible_level': bible_level,
                        'current_level': level,
                        'type': 'skill_mismatch'
                    })
                    
        return contradictions

    def _find_unexplained_knowledge(
        self,
        previous_knowledge: Dict[str, Any],
        current_knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find unexplained gains in knowledge or abilities."""
        unexplained = []
        
        for skill, level in current_knowledge.items():
            prev_level = previous_knowledge.get(skill)
            if not prev_level:
                unexplained.append({
                    'skill': skill,
                    'level': level,
                    'type': 'new_skill'
                })
            elif isinstance(level, (int, float)) and isinstance(prev_level, (int, float)):
                if level > prev_level + 1:  # Allowing for gradual improvement
                    unexplained.append({
                        'skill': skill,
                        'previous_level': prev_level,
                        'current_level': level,
                        'type': 'rapid_improvement'
                    })
                    
        return unexplained

    async def _check_plot_continuity(
        self,
        manuscript: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check plot continuity."""
        issues = []
        plot_points = self.state["tracked_elements"]["plot_points"]
        plot_outline = story_bible.get("plot_outline", {})

        for i in range(1, len(plot_points)):
            current = plot_points[i]
            previous = plot_points[i-1]
            
            if not await self._verify_plot_progression(previous, current, plot_outline):
                issues.append({
                    "type": "plot_inconsistency",
                    "chapter_range": [previous["chapter"], current["chapter"]],
                    "description": "Plot progression issue detected",
                    "details": {
                        "previous_point": previous,
                        "current_point": current
                    }
                })

        return issues

    async def _verify_plot_progression(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any],
        plot_outline: Dict[str, Any]
    ) -> bool:
        """Verify logical progression of plot points."""
        try:
            plot_sequence = plot_outline.get('sequence', [])
            
            prev_point_type = previous.get('type')
            curr_point_type = current.get('type')
            
            if prev_point_type in plot_sequence and curr_point_type in plot_sequence:
                prev_idx = plot_sequence.index(prev_point_type)
                curr_idx = plot_sequence.index(curr_point_type)
                
                if curr_idx < prev_idx:
                    return False
                    
            dependencies = plot_outline.get('dependencies', {})
            if curr_point_type in dependencies:
                required_points = dependencies[curr_point_type]
                if prev_point_type not in required_points:
                    return False
                    
            if 'character_arcs' in plot_outline:
                for char_arc in plot_outline['character_arcs']:
                    if char_arc['arc_points'].get(curr_point_type):
                        required_previous = char_arc['arc_points'].get(prev_point_type)
                        if not required_previous:
                            return False
                            
            if 'theme_development' in plot_outline:
                themes = plot_outline['theme_development']
                for theme in themes:
                    if curr_point_type in theme['progression']:
                        curr_stage_idx = theme['progression'].index(curr_point_type)
                        if prev_point_type in theme['progression']:
                            prev_stage_idx = theme['progression'].index(prev_point_type)
                            if curr_stage_idx < prev_stage_idx:
                                return False
                                
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying plot progression: {str(e)}")
            return True

def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
    """Calculate semantic similarity between two text descriptions."""
    try:
        # Using SequenceMatcher for basic string similarity
        # In a production environment, this could be replaced with
        # a more sophisticated embedding-based similarity measure
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    except Exception as e:
        self.logger.error(f"Error calculating semantic similarity: {str(e)}")
        return 0.0

    def _is_description_consistent(self, value1: Any, value2: Any) -> bool:
        """Check if two descriptions are consistent."""
        try:
            if isinstance(value1, str) and isinstance(value2, str):
                similarity = self._calculate_semantic_similarity(value1, value2)
                return similarity >= 0.8
            elif isinstance(value1, (list, set)) and isinstance(value2, (list, set)):
                return bool(set(value1).intersection(set(value2)))
            else:
                return value1 == value2
        except Exception as e:
            self.logger.error(f"Error checking description consistency: {str(e)}")
            return True

    async def _generate_recommendations(self, report: ContinuityReport) -> List[Dict[str, Any]]:
        """Generate recommendations for fixing continuity issues."""
        recommendations = []

        try:
            # Process timeline issues
            for issue in report.timeline_issues:
                rec = await self._generate_timeline_recommendation(issue)
                if rec:
                    recommendations.append(rec)

            # Process character issues
            for issue in report.character_issues:
                rec = await self._generate_character_recommendation(issue)
                if rec:
                    recommendations.append(rec)

            # Process plot issues
            for issue in report.plot_issues:
                rec = await self._generate_plot_recommendation(issue)
                if rec:
                    recommendations.append(rec)

            # Process setting issues
            for issue in report.setting_issues:
                rec = await self._generate_setting_recommendation(issue)
                if rec:
                    recommendations.append(rec)

            # Process object issues
            for issue in report.object_issues:
                rec = await self._generate_object_recommendation(issue)
                if rec:
                    recommendations.append(rec)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    async def _generate_timeline_recommendation(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate recommendation for timeline issue."""
        try:
            return {
                "type": "timeline_recommendation",
                "issue_type": issue["type"],
                "chapter_range": issue["chapter_range"],
                "recommendation": f"Review temporal sequence in chapters {issue['chapter_range'][0]}-{issue['chapter_range'][1]}. "
                                f"Ensure events follow logical chronological order and resolve any contradictions in timing.",
                "priority": "high" if issue.get("severity") == "high" else "medium",
                "suggested_actions": [
                    "Create a timeline diagram",
                    "Mark all temporal references",
                    "Verify sequence consistency",
                    "Adjust time markers if needed"
                ]
            }
        except Exception as e:
            self.logger.error(f"Error generating timeline recommendation: {str(e)}")
            return None

    async def _generate_character_recommendation(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate recommendation for character issue."""
        try:
            actions = []
            if "physical_inconsistency" in str(issue.get("type", "")):
                actions = [
                    "Review character physical description in story bible",
                    "Track all physical attribute mentions",
                    "Create consistent description template",
                    "Update inconsistent descriptions"
                ]
            elif "personality_inconsistency" in str(issue.get("type", "")):
                actions = [
                    "Review character personality profile",
                    "Check for character development arc",
                    "Ensure personality changes are properly motivated",
                    "Document character evolution"
                ]

            return {
                "type": "character_recommendation",
                "character": issue.get("character", "Unknown"),
                "issue_type": issue["type"],
                "chapter_range": issue["chapter_range"],
                "recommendation": f"Review character consistency for {issue.get('character', 'character')} "
                                f"in chapters {issue['chapter_range'][0]}-{issue['chapter_range'][1]}.",
                "priority": "high" if issue.get("severity") == "high" else "medium",
                "suggested_actions": actions
            }
        except Exception as e:
            self.logger.error(f"Error generating character recommendation: {str(e)}")
            return None

    async def _generate_plot_recommendation(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate recommendation for plot issue."""
        try:
            return {
                "type": "plot_recommendation",
                "issue_type": issue["type"],
                "chapter_range": issue["chapter_range"],
                "recommendation": f"Review plot progression in chapters {issue['chapter_range'][0]}-{issue['chapter_range'][1]}. "
                                f"Ensure events follow logical sequence and maintain cause-effect relationships.",
                "priority": "high",
                "suggested_actions": [
                    "Map plot points sequence",
                    "Verify cause-effect relationships",
                    "Check plot dependencies",
                    "Review character motivations",
                    "Ensure proper setup for key events"
                ]
            }
        except Exception as e:
            self.logger.error(f"Error generating plot recommendation: {str(e)}")
            return None

    async def _generate_setting_recommendation(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate recommendation for setting issue."""
        try:
            return {
                "type": "setting_recommendation",
                "issue_type": issue["type"],
                "chapter_range": issue["chapter_range"],
                "recommendation": f"Review setting consistency in chapters {issue['chapter_range'][0]}-{issue['chapter_range'][1]}. "
                                f"Ensure location details and environmental elements remain consistent.",
                "priority": "medium",
                "suggested_actions": [
                    "Create setting reference sheet",
                    "Map location relationships",
                    "Track environmental changes",
                    "Verify travel times and distances"
                ]
            }
        except Exception as e:
            self.logger.error(f"Error generating setting recommendation: {str(e)}")
            return None

    async def _generate_object_recommendation(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate recommendation for object issue."""
        try:
            return {
                "type": "object_recommendation",
                "issue_type": issue["type"],
                "chapter_range": issue["chapter_range"],
                "recommendation": f"Review object consistency in chapters {issue['chapter_range'][0]}-{issue['chapter_range'][1]}. "
                                f"Track object properties and location throughout the narrative.",
                "priority": "medium",
                "suggested_actions": [
                    "Create object tracking sheet",
                    "Verify object properties",
                    "Track object locations",
                    "Check object interactions"
                ]
            }
        except Exception as e:
            self.logger.error(f"Error generating object recommendation: {str(e)}")
            return None

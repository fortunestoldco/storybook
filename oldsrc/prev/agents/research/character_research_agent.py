from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService

class CharacterResearchAgent(BaseAgent):
    """Agent responsible for researching character archetypes, psychology, and historical personas."""
    
    def __init__(self, tools_service: ToolsService, **kwargs):
        super().__init__(**kwargs)
        self.tools_service = tools_service
        self.research_tools = tools_service.get_research_tools()
        self.logger = logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.state = {
            "memory": {},
            "last_update": None,
            "research_cache": {}
        }

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle character research tasks."""
        try:
            story_bible = task.get("story_bible", {})
            characters = story_bible.get("characters", [])
            
            research_report = {
                "novel_id": story_bible.get("novel_id"),
                "research_type": "character",
                "character_analyses": [],
                "archetype_findings": [],
                "historical_parallels": [],
                "psychological_insights": [],
                "recommendations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Research each character
            for character in characters:
                character_analysis = await self._analyze_character(character, story_bible)
                research_report["character_analyses"].append(character_analysis)
                
                # Find historical parallels
                historical_parallels = await self._find_historical_parallels(character)
                research_report["historical_parallels"].extend(historical_parallels)
                
                # Analyze psychological aspects
                psych_insights = await self._analyze_psychological_aspects(character)
                research_report["psychological_insights"].extend(psych_insights)
            
            # Identify archetypes
            archetypes = await self._identify_archetypes(characters)
            research_report["archetype_findings"] = archetypes
            
            # Generate recommendations
            research_report["recommendations"] = self._generate_recommendations(
                research_report,
                story_bible
            )
            
            # Validate report
            if not self._validate_report(research_report):
                raise ValueError("Character research report validation failed")
            
            # Update state
            self.state["last_update"] = datetime.utcnow()
            self.state["memory"]["last_report"] = research_report
            
            return research_report
            
        except Exception as e:
            self.logger.error(f"Error in character research: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_character(self, character: Dict[str, Any], story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an individual character."""
        try:
            analysis = {
                "character_id": character.get("id"),
                "name": character.get("name"),
                "role": character.get("role"),
                "personality_traits": await self._analyze_personality(character),
                "motivation_analysis": await self._analyze_motivation(character),
                "conflict_patterns": await self._analyze_conflicts(character, story_bible),
                "relationship_dynamics": await self._analyze_relationships(character, story_bible),
                "character_arc": await self._analyze_character_arc(character, story_bible),
                "background_authenticity": await self._verify_background_authenticity(character),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate analysis confidence
            analysis["confidence_score"] = self._calculate_analysis_confidence(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing character: {str(e)}")
            return {
                "character_id": character.get("id"),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_personality(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character personality traits and patterns."""
        try:
            personality = {
                "traits": [],
                "mbti_type": None,
                "enneagram_type": None,
                "big_five_scores": {},
                "behavioral_patterns": [],
                "consistency_score": 0.0,
                "development_potential": []
            }
            
            # Extract explicit traits
            traits = character.get("traits", [])
            personality["traits"] = self._validate_traits(traits)
            
            # Analyze MBTI type
            personality["mbti_type"] = await self._determine_mbti_type(character)
            
            # Analyze Enneagram type
            personality["enneagram_type"] = await self._determine_enneagram_type(character)
            
            # Calculate Big Five scores
            personality["big_five_scores"] = await self._calculate_big_five_scores(character)
            
            # Identify behavioral patterns
            personality["behavioral_patterns"] = await self._identify_behavioral_patterns(character)
            
            # Calculate trait consistency
            personality["consistency_score"] = self._calculate_trait_consistency(personality)
            
            # Identify development opportunities
            personality["development_potential"] = self._identify_development_potential(personality)
            
            return personality
            
        except Exception as e:
            self.logger.error(f"Error analyzing personality: {str(e)}")
            return {
                "traits": [],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_motivation(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character motivations and driving forces."""
        try:
            motivation = {
                "primary_motivation": None,
                "secondary_motivations": [],
                "goals": {
                    "short_term": [],
                    "long_term": []
                },
                "fears": [],
                "desires": [],
                "internal_conflicts": [],
                "motivation_strength": 0.0,
                "motivation_consistency": 0.0
            }
            
            # Extract explicit motivations
            motivation["primary_motivation"] = character.get("motivation")
            motivation["secondary_motivations"] = character.get("secondary_motivations", [])
            
            # Analyze goals
            goals = character.get("goals", {})
            motivation["goals"]["short_term"] = goals.get("short_term", [])
            motivation["goals"]["long_term"] = goals.get("long_term", [])
            
            # Extract psychological factors
            motivation["fears"] = character.get("fears", [])
            motivation["desires"] = character.get("desires", [])
            motivation["internal_conflicts"] = self._identify_internal_conflicts(character)
            
            # Calculate motivation metrics
            motivation["motivation_strength"] = self._calculate_motivation_strength(motivation)
            motivation["motivation_consistency"] = self._calculate_motivation_consistency(motivation)
            
            return motivation
            
        except Exception as e:
            self.logger.error(f"Error analyzing motivation: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_conflicts(self, character: Dict[str, Any], story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character conflicts and obstacles."""
        try:
            conflicts = {
                "internal_conflicts": [],
                "external_conflicts": [],
                "relationship_conflicts": [],
                "conflict_patterns": [],
                "resolution_patterns": [],
                "conflict_intensity": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze internal conflicts
            conflicts["internal_conflicts"] = self._analyze_internal_conflicts(character)
            
            # Analyze external conflicts
            conflicts["external_conflicts"] = self._analyze_external_conflicts(
                character,
                story_bible
            )
            
            # Analyze relationship conflicts
            conflicts["relationship_conflicts"] = self._analyze_relationship_conflicts(
                character,
                story_bible
            )
            
            # Identify patterns
            conflicts["conflict_patterns"] = self._identify_conflict_patterns(conflicts)
            conflicts["resolution_patterns"] = self._identify_resolution_patterns(
                character,
                conflicts
            )
            
            # Calculate intensity
            conflicts["conflict_intensity"] = self._calculate_conflict_intensity(conflicts)
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Error analyzing conflicts: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_relationships(self, character: Dict[str, Any], story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character relationships and dynamics."""
        try:
            relationships = {
                "key_relationships": [],
                "relationship_network": {},
                "alliance_patterns": [],
                "opposition_patterns": [],
                "power_dynamics": {},
                "relationship_development": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze key relationships
            relationships["key_relationships"] = self._analyze_key_relationships(
                character,
                story_bible
            )
            
            # Build relationship network
            relationships["relationship_network"] = self._build_relationship_network(
                character,
                story_bible
            )
            
            # Identify patterns
            relationships["alliance_patterns"] = self._identify_alliance_patterns(
                relationships["relationship_network"]
            )
            relationships["opposition_patterns"] = self._identify_opposition_patterns(
                relationships["relationship_network"]
            )
            
            # Analyze power dynamics
            relationships["power_dynamics"] = self._analyze_power_dynamics(
                relationships["relationship_network"]
            )
            
            # Track relationship development
            relationships["relationship_development"] = self._track_relationship_development(
                character,
                story_bible
            )
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error analyzing relationships: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_character_arc(self, character: Dict[str, Any], story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character development and story arc."""
        try:
            arc = {
                "arc_type": None,
                "development_stages": [],
                "key_moments": [],
                "growth_areas": [],
                "transformation_points": [],
                "arc_completion": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Determine arc type
            arc["arc_type"] = self._determine_arc_type(character)
            
            # Identify development stages
            arc["development_stages"] = self._identify_development_stages(
                character,
                story_bible
            )
            
            # Identify key moments
            arc["key_moments"] = self._identify_key_moments(
                character,
                story_bible
            )
            
            # Analyze growth
            arc["growth_areas"] = self._analyze_growth_areas(character)
            
            # Identify transformation points
            arc["transformation_points"] = self._identify_transformation_points(
                character,
                story_bible
            )
            
            # Calculate completion
            arc["arc_completion"] = self._calculate_arc_completion(arc)
            
            return arc
            
        except Exception as e:
            self.logger.error(f"Error analyzing character arc: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _verify_background_authenticity(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the historical authenticity of character background."""
        try:
            authenticity = {
                "historical_accuracy": 0.0,
                "anachronisms": [],
                "verified_elements": [],
                "suggested_corrections": [],
                "research_sources": [],
                "confidence_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check historical accuracy
            background = character.get("background", {})
            authenticity["historical_accuracy"] = await self._check_historical_accuracy(background)
            
            # Identify anachronisms
            authenticity["anachronisms"] = await self._identify_anachronisms(background)
            
            # Verify elements
            authenticity["verified_elements"] = await self._verify_background_elements(background)
            
            # Generate corrections
            authenticity["suggested_corrections"] = self._generate_corrections(
                background,
                authenticity["anachronisms"]
            )
            
            # Collect sources
            authenticity["research_sources"] = await self._collect_research_sources(background)
            
            # Calculate confidence
            authenticity["confidence_score"] = self._calculate_verification_confidence(authenticity)
            
            return authenticity
            
        except Exception as e:
            self.logger.error(f"Error verifying background: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _find_historical_parallels(self, character: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find historical figures or archetypes that parallel the character."""
        try:
            parallels = []
            
            # Get character traits and role
            traits = character.get("traits", [])
            role = character.get("role", "")
            
            # Search historical database
            doc_results = await self.tools_service.get_document_retriever()._arun(
                f"historical figures with traits: {', '.join(traits)} role: {role}"
            )
            
            for result in doc_results:
                parallel = {
                    "historical_figure": result["name"],
                    "similarity_score": self._calculate_similarity_score(character, result),
                    "matching_traits": self._find_matching_traits(character, result),
                    "contextual_differences": self._identify_contextual_differences(character, result),
                    "relevant_lessons": self._extract_relevant_lessons(result),
                    "confidence_score": self._calculate_parallel_confidence(result),
                    "timestamp": datetime.utcnow().isoformat()
                }
                parallels.append(parallel)
            
            # Sort by similarity score
            parallels.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return parallels[:5]  # Return top 5 parallels
            
        except Exception as e:
            self.logger.error(f"Error finding historical parallels: {str(e)}")
            return []

    async def _analyze_psychological_aspects(self, character: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze psychological aspects of the character."""
        try:
            insights = []
            
            # Analyze personality type
            personality_insight = await self._analyze_personality_type(character)
            insights.append(personality_insight)
            
            # Analyze behavioral patterns
            behavior_insight = await self._analyze_behavioral_patterns(character)
            insights.append(behavior_insight)
            
            # Analyze emotional patterns
            emotional_insight = await self._analyze_emotional_patterns(character)
            insights.append(emotional_insight)
            
            # Analyze cognitive patterns
            cognitive_insight = await self._analyze_cognitive_patterns(character)
            insights.append(cognitive_insight)
            
            # Analyze defense mechanisms
            defense_insight = await self._analyze_defense_mechanisms(character)
            insights.append(defense_insight)
            
            # Filter valid insights and sort by confidence
            valid_insights = [
                insight for insight in insights 
                if insight and insight.get("confidence_score", 0) > 0.5
            ]
            valid_insights.sort(key=lambda x: x["confidence_score"], reverse=True)
            
            return valid_insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing psychological aspects: {str(e)}")
            return []

    async def _analyze_personality_type(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character's personality type using psychological frameworks."""
        try:
            personality_type = {
                "insight_type": "personality",
                "mbti_analysis": {
                    "type": None,
                    "explanation": [],
                    "confidence": 0.0
                },
                "big_five_analysis": {
                    "scores": {},
                    "interpretation": [],
                    "confidence": 0.0
                },
                "enneagram_analysis": {
                    "type": None,
                    "wing": None,
                    "explanation": [],
                    "confidence": 0.0
                },
                "key_traits": [],
                "behavioral_implications": [],
                "development_suggestions": [],
                "confidence_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze MBTI
            mbti_result = await self._determine_mbti_type(character)
            personality_type["mbti_analysis"] = mbti_result
            
            # Analyze Big Five
            big_five_result = await self._analyze_big_five(character)
            personality_type["big_five_analysis"] = big_five_result
            
            # Analyze Enneagram
            enneagram_result = await self._determine_enneagram_type(character)
            personality_type["enneagram_analysis"] = enneagram_result
            
            # Extract key traits
            personality_type["key_traits"] = self._extract_key_traits(
                mbti_result,
                big_five_result,
                enneagram_result
            )
            
            # Generate behavioral implications
            personality_type["behavioral_implications"] = self._generate_behavioral_implications(
                personality_type["key_traits"]
            )
            
            # Generate development suggestions
            personality_type["development_suggestions"] = self._generate_development_suggestions(
                personality_type
            )
            
            # Calculate overall confidence
            personality_type["confidence_score"] = (
                mbti_result["confidence"] +
                big_five_result["confidence"] +
                enneagram_result["confidence"]
            ) / 3
            
            return personality_type
            
        except Exception as e:
            self.logger.error(f"Error analyzing personality type: {str(e)}")
            return {
                "insight_type": "personality",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_behavioral_patterns(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character's behavioral patterns and tendencies."""
        try:
            behavior = {
                "insight_type": "behavioral",
                "recurring_patterns": [],
                "situational_responses": {},
                "trigger_patterns": [],
                "coping_mechanisms": [],
                "behavioral_strengths": [],
                "behavioral_challenges": [],
                "growth_opportunities": [],
                "confidence_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Identify recurring patterns
            behavior["recurring_patterns"] = self._identify_recurring_patterns(character)
            
            # Analyze situational responses
            behavior["situational_responses"] = self._analyze_situational_responses(character)
            
            # Identify triggers
            behavior["trigger_patterns"] = self._identify_triggers(character)
            
            # Analyze coping mechanisms
            behavior["coping_mechanisms"] = self._analyze_coping_mechanisms(character)
            
            # Identify strengths and challenges
            behavior["behavioral_strengths"] = self._identify_behavioral_strengths(character)
            behavior["behavioral_challenges"] = self._identify_behavioral_challenges(character)
            
            # Generate growth opportunities
            behavior["growth_opportunities"] = self._generate_growth_opportunities(
                behavior["behavioral_challenges"]
            )
            
            # Calculate confidence score
            behavior["confidence_score"] = self._calculate_behavioral_confidence(behavior)
            
            return behavior
            
        except Exception as e:
            self.logger.error(f"Error analyzing behavioral patterns: {str(e)}")
            return {
                "insight_type": "behavioral",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_emotional_patterns(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character's emotional patterns and emotional intelligence."""
        try:
            emotional = {
                "insight_type": "emotional",
                "emotional_range": [],
                "emotional_triggers": {},
                "emotional_responses": [],
                "emotional_intelligence": {
                    "self_awareness": 0.0,
                    "self_regulation": 0.0,
                    "motivation": 0.0,
                    "empathy": 0.0,
                    "social_skills": 0.0
                },
                "emotional_growth": [],
                "confidence_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze emotional range
            emotional["emotional_range"] = self._analyze_emotional_range(character)
            
            # Identify emotional triggers
            emotional["emotional_triggers"] = self._identify_emotional_triggers(character)
            
            # Analyze emotional responses
            emotional["emotional_responses"] = self._analyze_emotional_responses(character)
            
            # Assess emotional intelligence
            emotional["emotional_intelligence"] = self._assess_emotional_intelligence(character)
            
            # Identify growth areas
            emotional["emotional_growth"] = self._identify_emotional_growth_areas(
                emotional["emotional_intelligence"]
            )
            
            # Calculate confidence score
            emotional["confidence_score"] = self._calculate_emotional_confidence(emotional)
            
            return emotional
            
        except Exception as e:
            self.logger.error(f"Error analyzing emotional patterns: {str(e)}")
            return {
                "insight_type": "emotional",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_cognitive_patterns(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character's cognitive patterns and decision-making processes."""
        try:
            cognitive = {
                "insight_type": "cognitive",
                "thinking_style": None,
                "decision_patterns": [],
                "cognitive_biases": [],
                "problem_solving_approach": [],
                "learning_style": None,
                "mental_models": [],
                "cognitive_strengths": [],
                "cognitive_challenges": [],
                "development_areas": [],
                "confidence_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze thinking style
            cognitive["thinking_style"] = self._analyze_thinking_style(character)
            
            # Identify decision patterns
            cognitive["decision_patterns"] = self._identify_decision_patterns(character)
            
            # Identify cognitive biases
            cognitive["cognitive_biases"] = self._identify_cognitive_biases(character)
            
            # Analyze problem-solving approach
            cognitive["problem_solving_approach"] = self._analyze_problem_solving(character)
            
            # Determine learning style
            cognitive["learning_style"] = self._determine_learning_style(character)
            
            # Identify mental models
            cognitive["mental_models"] = self._identify_mental_models(character)
            
            # Analyze strengths and challenges
            cognitive["cognitive_strengths"] = self._identify_cognitive_strengths(character)
            cognitive["cognitive_challenges"] = self._identify_cognitive_challenges(character)
            
            # Generate development areas
            cognitive["development_areas"] = self._generate_cognitive_development_areas(
                cognitive["cognitive_challenges"]
            )
            
            # Calculate confidence score
            cognitive["confidence_score"] = self._calculate_cognitive_confidence(cognitive)
            
            return cognitive
            
        except Exception as e:
            self.logger.error(f"Error analyzing cognitive patterns: {str(e)}")
            return {
                "insight_type": "cognitive",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_defense_mechanisms(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze character's psychological defense mechanisms."""
        try:
            defense = {
                "insight_type": "defense_mechanisms",
                "primary_defenses": [],
                "secondary_defenses": [],
                "trigger_situations": {},
                "adaptive_value": {},
                "development_level": 0.0,
                "recommendations": [],
                "confidence_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Identify primary defense mechanisms
            defense["primary_defenses"] = self._identify_primary_defenses(character)
            
            # Identify secondary defense mechanisms
            defense["secondary_defenses"] = self._identify_secondary_defenses(character)
            
            # Analyze trigger situations
            defense["trigger_situations"] = self._analyze_defense_triggers(character)
            
            # Assess adaptive value
            defense["adaptive_value"] = self._assess_defense_adaptation(
                defense["primary_defenses"],
                defense["secondary_defenses"]
            )
            
            # Calculate development level
            defense["development_level"] = self._calculate_defense_development(defense)
            
            # Generate recommendations
            defense["recommendations"] = self._generate_defense_recommendations(defense)
            
            # Calculate confidence score
            defense["confidence_score"] = self._calculate_defense_confidence(defense)
            
            return defense
            
        except Exception as e:
            self.logger.error(f"Error analyzing defense mechanisms: {str(e)}")
            return {
                "insight_type": "defense_mechanisms",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _identify_archetypes(self, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify character archetypes and their variations."""
        try:
            archetypes = []
            
            for character in characters:
                archetype = {
                    "character_id": character.get("id"),
                    "primary_archetype": None,
                    "secondary_archetypes": [],
                    "archetype_variations": [],
                    "alignment_score": 0.0,
                    "unique_elements": [],
                    "development_suggestions": [],
                    "confidence_score": 0.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Identify primary archetype
                archetype["primary_archetype"] = await self._identify_primary_archetype(character)
                
                # Identify secondary archetypes
                archetype["secondary_archetypes"] = await self._identify_secondary_archetypes(
                    character,
                    archetype["primary_archetype"]
                )
                
                # Analyze variations
                archetype["archetype_variations"] = self._analyze_archetype_variations(
                    character,
                    archetype["primary_archetype"]
                )
                
                # Calculate alignment
                archetype["alignment_score"] = self._calculate_archetype_alignment(
                    character,
                    archetype
                )
                
                # Identify unique elements
                archetype["unique_elements"] = self._identify_unique_elements(
                    character,
                    archetype
                )
                
                # Generate development suggestions
                archetype["development_suggestions"] = self._generate_archetype_suggestions(
                    character,
                    archetype
                )
                
                # Calculate confidence score
                archetype["confidence_score"] = self._calculate_archetype_confidence(archetype)
                
                archetypes.append(archetype)
            
            return archetypes
            
        except Exception as e:
            self.logger.error(f"Error identifying archetypes: {str(e)}")
            return []

    def _generate_recommendations(self, report: Dict[str, Any], story_bible: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on character research findings."""
        try:
            recommendations = []
            
            # Process character analyses
            for analysis in report["character_analyses"]:
                character_recs = self._generate_character_recommendations(analysis, story_bible)
                recommendations.extend(character_recs)
            
            # Process archetype findings
            archetype_recs = self._generate_archetype_recommendations(
                report["archetype_findings"],
                story_bible
            )
            recommendations.extend(archetype_recs)
            
            # Process historical parallels
            historical_recs = self._generate_historical_recommendations(
                report["historical_parallels"],
                story_bible
            )
            recommendations.extend(historical_recs)
            
            # Process psychological insights
            psych_recs = self._generate_psychological_recommendations(
                report["psychological_insights"],
                story_bible
            )
            recommendations.extend(psych_recs)
            
            # Sort and deduplicate recommendations
            recommendations.sort(key=lambda x: x.get("priority", 0), reverse=True)
            recommendations = self._deduplicate_recommendations(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def _generate_character_recommendations(self, analysis: Dict[str, Any], story_bible: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on character analysis."""
        try:
            recommendations = []
            
            # Personality recommendations
            if "personality_traits" in analysis:
                personality_recs = self._generate_personality_recommendations(
                    analysis["personality_traits"],
                    story_bible
                )
                recommendations.extend(personality_recs)
            
            # Motivation recommendations
            if "motivation_analysis" in analysis:
                motivation_recs = self._generate_motivation_recommendations(
                    analysis["motivation_analysis"],
                    story_bible
                )
                recommendations.extend(motivation_recs)
            
            # Conflict recommendations
            if "conflict_patterns" in analysis:
                conflict_recs = self._generate_conflict_recommendations(
                    analysis["conflict_patterns"],
                    story_bible
                )
                recommendations.extend(conflict_recs)
            
            # Relationship recommendations
            if "relationship_dynamics" in analysis:
                relationship_recs = self._generate_relationship_recommendations(
                    analysis["relationship_dynamics"],
                    story_bible
                )
                recommendations.extend(relationship_recs)
            
            # Character arc recommendations
            if "character_arc" in analysis:
                arc_recs = self._generate_arc_recommendations(
                    analysis["character_arc"],
                    story_bible
                )
                recommendations.extend(arc_recs)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating character recommendations: {str(e)}")
            return []

    def _generate_personality_recommendations(self, personality_traits: Dict[str, Any], story_bible: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on personality analysis."""
        try:
            recommendations = []
            
            # Trait development recommendations
            for trait in personality_traits.get("traits", []):
                rec = {
                    "category": "personality_development",
                    "trait": trait,
                    "recommendation": f"Develop {trait} through character interactions and challenges",
                    "implementation_ideas": [
                        f"Create scenes that showcase {trait}",
                        f"Show how {trait} affects relationships",
                        f"Demonstrate growth or conflict related to {trait}"
                    ],
                    "priority": self._calculate_trait_priority(trait, story_bible),
                    "confidence_score": self._calculate_trait_confidence(trait),
                    "timestamp": datetime.utcnow().isoformat()
                }
                recommendations.append(rec)
            
            # Consistency recommendations
            consistency_score = personality_traits.get("consistency_score", 0.0)
            if consistency_score < 0.7:
                rec = {
                    "category": "personality_consistency",
                    "recommendation": "Improve personality trait consistency",
                    "implementation_ideas": [
                        "Review trait manifestations across scenes",
                        "Create trait expression guidelines",
                        "Document trait interactions"
                    ],
                    "priority": 4,
                    "confidence_score": consistency_score,
                    "timestamp": datetime.utcnow().isoformat()
                }
                recommendations.append(rec)
            
            # Development potential recommendations
            for potential in personality_traits.get("development_potential", []):
                rec = {
                    "category": "personality_growth",
                    "focus_area": potential,
                    "recommendation": f"Explore character growth in {potential}",
                    "implementation_ideas": [
                        f"Create challenges related to {potential}",
                        f"Show gradual development in {potential}",
                        "Include reflection moments"
                    ],
                    "priority": 3,
                    "confidence_score": 0.8,
                    "timestamp": datetime.utcnow().isoformat()
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating personality recommendations: {str(e)}")
            return []

    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations while preserving highest priority items."""
        try:
            unique_recs = {}
            for rec in recommendations:
                key = f"{rec['category']}_{rec.get('focus_area', '')}_{rec['recommendation']}"
                if key not in unique_recs or rec.get('priority', 0) > unique_recs[key].get('priority', 0):
                    unique_recs[key] = rec
            
            return list(unique_recs.values())
            
        except Exception as e:
            self.logger.error(f"Error deduplicating recommendations: {str(e)}")
            return recommendations

    def _calculate_trait_priority(self, trait: str, story_bible: Dict[str, Any]) -> int:
        """Calculate priority score for a personality trait."""
        try:
            priority = 2  # Base priority
            
            # Check if trait is related to main themes
            if "themes" in story_bible:
                for theme in story_bible["themes"]:
                    if trait.lower() in theme.lower():
                        priority += 1
            
            # Check if trait is related to plot
            if "plot" in story_bible:
                plot_summary = story_bible["plot"].get("summary", "")
                if trait.lower() in plot_summary.lower():
                    priority += 1
            
            # Check if trait is related to character arc
            if "character_arcs" in story_bible:
                for arc in story_bible["character_arcs"]:
                    if trait.lower() in arc.get("description", "").lower():
                        priority += 1
            
            return min(priority, 5)  # Cap at 5
            
        except Exception as e:
            self.logger.error(f"Error calculating trait priority: {str(e)}")
            return 2

    def _calculate_trait_confidence(self, trait: str) -> float:
        """Calculate confidence score for a personality trait."""
        try:
            confidence = 0.7  # Base confidence
            
            # Adjust based on trait specificity
            if len(trait.split()) > 1:
                confidence += 0.1  # More specific traits get higher confidence
            
            # Adjust based on trait complexity
            if any(word in trait.lower() for word in ["sometimes", "occasionally", "tends to"]):
                confidence -= 0.1  # Qualified traits get lower confidence
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trait confidence: {str(e)}")
            return 0.7

    async def cleanup(self) -> None:
        """Cleanup after character research completion."""
        try:
            # Clear temporary state
            self.state["memory"] = {}
            
            # Archive old cache entries
            current_time = datetime.utcnow()
            self.state["research_cache"] = {
                topic: data 
                for topic, data in self.state["research_cache"].items()
                if (current_time - datetime.fromisoformat(data["timestamp"])).days < 30
            }
            
            # Log cleanup
            self.logger.info("Character research agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
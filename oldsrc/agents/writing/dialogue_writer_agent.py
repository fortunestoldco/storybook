from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import json

class DialogueWriterAgent(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.character_voices = {}
        self.system_prompts = self._load_system_prompts()
        self.dialogue_chains = self._initialize_dialogue_chains()

    def _load_system_prompts(self) -> Dict[str, Any]:
        """Load system prompts from configuration file."""
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    def _initialize_dialogue_chains(self) -> Dict[str, LLMChain]:
        """Initialize specialized chains for dialogue generation."""
        chains = {}
        
        # Character voice chain
        voice_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("character_voice_instruction", "")),
            ("human", "{character_info}"),
            ("ai", "{voice_profile}")
        ])
        chains["character_voice"] = LLMChain(
            llm=self.llm,
            prompt=voice_prompt,
            verbose=True
        )
        
        # Dialogue generation chain
        dialogue_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("dialogue_generation_instruction", "")),
            ("human", "{dialogue_context}"),
            ("ai", "{dialogue_exchange}")
        ])
        chains["dialogue_generation"] = LLMChain(
            llm=self.llm,
            prompt=dialogue_prompt,
            verbose=True
        )
        
        return chains

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Write dialogue for a chapter based on dialogue markers."""
        try:
            story_bible = task.get("story_bible", {})
            character_spec = task.get("character_spec", {})
            chapter_content = task.get("chapter_content", {})
            
            # Initialize character voices if not already done
            if not self.character_voices:
                self.character_voices = await self._initialize_character_voices(character_spec)

            chapter_with_dialogue = {
                "novel_id": story_bible.get("novel_id"),
                "chapter_number": chapter_content.get("chapter_number"),
                "title": chapter_content.get("title"),
                "scenes": await self._process_scenes_dialogue(
                    chapter_content.get("scenes", []),
                    character_spec
                ),
                "status": "dialogue_complete",
                "last_updated": datetime.utcnow().isoformat()
            }

            # Store updated chapter in MongoDB
            await self.mongodb_service.update_chapter(
                story_bible["novel_id"],
                chapter_content["chapter_number"],
                chapter_with_dialogue
            )

            return chapter_with_dialogue
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "task": "dialogue_writing",
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.mongodb_service.store_error(error_data)
            raise

    async def _initialize_character_voices(self, character_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Initialize unique voice patterns for each character."""
        voices = {}
        
        # Process main characters
        for character in character_spec.get("main_characters", []):
            voice_profile = await self.dialogue_chains["character_voice"].arun(
                character_info=character
            )
            voices[character["name"]] = self._create_character_voice(character, voice_profile)
            
        # Process supporting characters
        for character in character_spec.get("supporting_characters", []):
            voice_profile = await self.dialogue_chains["character_voice"].arun(
                character_info=character
            )
            voices[character["name"]] = self._create_character_voice(character, voice_profile)
            
        return voices

    def _create_character_voice(self, character: Dict[str, Any], voice_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unique voice profile for a character."""
        return {
            "vocabulary_level": self._determine_vocabulary_level(character),
            "speech_patterns": self._determine_speech_patterns(character, voice_profile),
            "dialectal_features": self._determine_dialectal_features(character),
            "emotional_expressions": self._determine_emotional_expressions(character),
            "catchphrases": self._generate_catchphrases(character),
            "communication_style": self._determine_communication_style(character)
        }

    def _determine_vocabulary_level(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the vocabulary level and complexity for a character."""
        background = character.get("background", {})
        personality = character.get("personality", {})
        
        return {
            "complexity": self._calculate_vocabulary_complexity(background),
            "education_level": background.get("education", "average"),
            "professional_jargon": self._get_professional_vocabulary(background.get("profession", "")),
            "cultural_influences": background.get("cultural_influences", []),
            "adaptability": personality.get("adaptability", "moderate")
        }

    def _determine_speech_patterns(self, character: Dict[str, Any], voice_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Determine unique speech patterns for a character."""
        personality = character.get("personality", {})
        
        return {
            "sentence_length": voice_profile.get("preferred_sentence_length", "medium"),
            "rhythm_patterns": voice_profile.get("speech_rhythm", []),
            "verbal_tics": voice_profile.get("verbal_tics", []),
            "pause_patterns": self._analyze_pause_patterns(personality),
            "emphasis_patterns": self._analyze_emphasis_patterns(personality)
        }

    def _determine_dialectal_features(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Determine dialectal features of a character's speech."""
        background = character.get("background", {})
        
        return {
            "accent": background.get("accent", "neutral"),
            "dialect_markers": self._get_dialect_markers(background),
            "regional_expressions": self._get_regional_expressions(background),
            "cultural_markers": self._get_cultural_speech_markers(background)
        }

    def _determine_emotional_expressions(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how a character expresses emotions through speech."""
        personality = character.get("personality", {})
        
        return {
            "emotional_vocabulary": self._get_emotional_vocabulary(personality),
            "emotional_tells": self._identify_emotional_tells(personality),
            "suppression_patterns": self._identify_suppression_patterns(personality),
            "emotional_escalation": self._map_emotional_escalation(personality)
        }

    async def _process_scenes_dialogue(
        self,
        scenes: List[Dict[str, Any]],
        character_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process and write dialogue for all scenes in the chapter."""
        processed_scenes = []
        
        for scene in scenes:
            processed_scene = scene.copy()
            dialogue_markers = scene.get("dialogue_markers", [])
            
            if dialogue_markers:
                processed_scene["dialogue"] = await self._generate_scene_dialogue(
                    dialogue_markers,
                    character_spec,
                    scene.get("mood", "neutral"),
                    scene.get("tension_level", "moderate")
                )
            
            processed_scenes.append(processed_scene)
            
        return processed_scenes

    async def _generate_scene_dialogue(
        self,
        dialogue_markers: List[Dict[str, Any]],
        character_spec: Dict[str, Any],
        mood: str,
        tension_level: str
    ) -> List[Dict[str, Any]]:
        """Generate dialogue for a scene based on markers."""
        dialogue_exchanges = []
        
        for marker in dialogue_markers:
            exchange = await self._generate_dialogue_exchange(
                marker,
                character_spec,
                mood,
                tension_level
            )
            dialogue_exchanges.append(exchange)
            
        return dialogue_exchanges

    async def _generate_dialogue_exchange(
        self,
        marker: Dict[str, Any],
        character_spec: Dict[str, Any],
        mood: str,
        tension_level: str
    ) -> Dict[str, Any]:
        """Generate a specific dialogue exchange."""
        try:
            dialogue_context = {
                "speakers": marker.get("speakers", []),
                "context": marker.get("context", ""),
                "purpose": marker.get("purpose", ""),
                "mood": mood,
                "tension_level": tension_level,
                "character_voices": {
                    speaker: self.character_voices.get(speaker, {})
                    for speaker in marker.get("speakers", [])
                }
            }
            
            exchange = await self.dialogue_chains["dialogue_generation"].arun(
                dialogue_context=dialogue_context
            )
            
            return {
                "speakers": marker["speakers"],
                "lines": exchange.get("lines", []),
                "subtext": exchange.get("subtext", ""),
                "emotional_progression": exchange.get("emotional_progression", {}),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "marker": marker,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def cleanup(self) -> None:
        """Cleanup resources after task completion."""
        try:
            # Clear character voices cache
            self.character_voices = {}
            
            # Clear dialogue chains
            self.dialogue_chains = {}
            
            # Log completion
            await self.mongodb_service.store_agent_status({
                "agent_name": self.name,
                "status": "cleanup_complete",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            await self.mongodb_service.store_error({
                "agent_name": self.name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

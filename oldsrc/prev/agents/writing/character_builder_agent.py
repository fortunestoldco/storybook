from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import json

class CharacterBuilderAgent(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.system_prompts = self._load_system_prompts()
        self.character_chains = self._initialize_character_chains()
        self.relationship_matrix = {}

    def _load_system_prompts(self) -> Dict[str, Any]:
        """Load system prompts from configuration file."""
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    def _initialize_character_chains(self) -> Dict[str, LLMChain]:
        """Initialize specialized chains for character development."""
        chains = {}
        
        # Character creation chain
        character_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("character_creation_instruction", "")),
            ("human", "{character_context}"),
            ("ai", "{character_profile}")
        ])
        chains["character_creation"] = LLMChain(
            llm=self.llm,
            prompt=character_prompt,
            verbose=True
        )
        
        # Relationship development chain
        relationship_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("relationship_development_instruction", "")),
            ("human", "{relationship_context}"),
            ("ai", "{relationship_dynamics}")
        ])
        chains["relationship_development"] = LLMChain(
            llm=self.llm,
            prompt=relationship_prompt,
            verbose=True
        )
        
        # Character arc chain
        arc_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("character_arc_instruction", "")),
            ("human", "{arc_context}"),
            ("ai", "{character_arc}")
        ])
        chains["character_arc"] = LLMChain(
            llm=self.llm,
            prompt=arc_prompt,
            verbose=True
        )
        
        return chains

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Develop deep and nuanced characters based on the story bible."""
        try:
            story_bible = task.get("story_bible", {})
            world_spec = task.get("world_spec", {})

            character_profiles = {
                "novel_id": story_bible.get("novel_id"),
                "main_characters": await self._develop_main_characters(story_bible, world_spec),
                "supporting_characters": await self._develop_supporting_characters(story_bible, world_spec),
                "minor_characters": await self._develop_minor_characters(story_bible, world_spec),
                "relationship_network": await self._develop_relationship_network(),
                "character_arcs": await self._develop_character_arcs(story_bible),
                "status": "character_development_complete",
                "created_at": datetime.utcnow().isoformat()
            }

            # Store character profiles in MongoDB
            await self.mongodb_service.store_character_profiles(
                story_bible["novel_id"],
                character_profiles
            )

            return character_profiles
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "task": "character_development",
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.mongodb_service.store_error(error_data)
            raise

    async def _develop_main_characters(
        self,
        story_bible: Dict[str, Any],
        world_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Develop detailed profiles for main characters."""
        main_characters = []
        
        for character_brief in story_bible.get("main_characters", []):
            character_context = {
                "character_brief": character_brief,
                "world_context": world_spec,
                "story_themes": story_bible.get("themes", []),
                "character_type": "main"
            }
            
            profile = await self._create_character_profile(character_context)
            main_characters.append(profile)
            
            # Add to relationship matrix
            self.relationship_matrix[profile["name"]] = {}
        
        return main_characters

    async def _develop_supporting_characters(
        self,
        story_bible: Dict[str, Any],
        world_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Develop detailed profiles for supporting characters."""
        supporting_characters = []
        
        for character_brief in story_bible.get("supporting_characters", []):
            character_context = {
                "character_brief": character_brief,
                "world_context": world_spec,
                "story_themes": story_bible.get("themes", []),
                "character_type": "supporting"
            }
            
            profile = await self._create_character_profile(character_context)
            supporting_characters.append(profile)
            
            # Add to relationship matrix
            self.relationship_matrix[profile["name"]] = {}
        
        return supporting_characters

    async def _develop_minor_characters(
        self,
        story_bible: Dict[str, Any],
        world_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Develop detailed profiles for minor characters."""
        minor_characters = []
        
        for character_brief in story_bible.get("minor_characters", []):
            character_context = {
                "character_brief": character_brief,
                "world_context": world_spec,
                "story_themes": story_bible.get("themes", []),
                "character_type": "minor"
            }
            
            profile = await self._create_character_profile(character_context)
            minor_characters.append(profile)
            
            # Add to relationship matrix if they have significant relationships
            if profile.get("significant_relationships", False):
                self.relationship_matrix[profile["name"]] = {}
        
        return minor_characters

    async def _create_character_profile(self, character_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed character profile."""
        try:
            profile_response = await self.character_chains["character_creation"].arun(
                character_context=character_context
            )
            
            profile = {
                "name": profile_response.get("name", ""),
                "background": self._create_background(profile_response),
                "personality": self._create_personality(profile_response),
                "motivations": self._create_motivations(profile_response),
                "physical_description": self._create_physical_description(profile_response),
                "voice": self._create_voice(profile_response),
                "character_type": character_context["character_type"],
                "created_at": datetime.utcnow().isoformat()
            }
            
            return profile
            
        except Exception as e:
            return {"error": str(e)}

    def _create_background(self, profile_response: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed background for a character."""
        return {
            "origin": profile_response.get("origin", ""),
            "family_history": profile_response.get("family_history", {}),
            "education": profile_response.get("education", ""),
            "occupation": profile_response.get("occupation", ""),
            "key_life_events": profile_response.get("key_life_events", []),
            "cultural_background": profile_response.get("cultural_background", {}),
            "formative_experiences": profile_response.get("formative_experiences", [])
        }

    def _create_personality(self, profile_response: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed personality profile."""
        return {
            "traits": profile_response.get("personality_traits", []),
            "quirks": profile_response.get("quirks", []),
            "values": profile_response.get("values", []),
            "fears": profile_response.get("fears", []),
            "desires": profile_response.get("desires", []),
            "psychological_profile": profile_response.get("psychological_profile", {}),
            "cognitive_patterns": profile_response.get("cognitive_patterns", {})
        }

    def _create_motivations(self, profile_response: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed character motivations."""
        return {
            "primary_goal": profile_response.get("primary_goal", ""),
            "secondary_goals": profile_response.get("secondary_goals", []),
            "internal_motivations": profile_response.get("internal_motivations", []),
            "external_motivations": profile_response.get("external_motivations", []),
            "conflicts": profile_response.get("internal_conflicts", [])
        }

    def _create_physical_description(self, profile_response: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed physical description."""
        return {
            "appearance": profile_response.get("appearance", {}),
            "mannerisms": profile_response.get("mannerisms", []),
            "physical_traits": profile_response.get("physical_traits", []),
            "distinguishing_features": profile_response.get("distinguishing_features", [])
        }

    def _create_voice(self, profile_response: Dict[str, Any]) -> Dict[str, Any]:
        """Create character voice profile."""
        return {
            "speech_pattern": profile_response.get("speech_pattern", ""),
            "vocabulary": profile_response.get("vocabulary_level", ""),
            "accent": profile_response.get("accent", ""),
            "verbal_tics": profile_response.get("verbal_tics", []),
            "common_phrases": profile_response.get("common_phrases", [])
        }

    async def _develop_relationship_network(self) -> Dict[str, Dict[str, Any]]:
        """Develop the network of relationships between characters."""
        relationship_network = {}
        
        # Develop relationships between all characters in the matrix
        for char1 in self.relationship_matrix:
            relationship_network[char1] = {}
            for char2 in self.relationship_matrix:
                if char1 != char2 and char2 not in relationship_network.get(char1, {}):
                    relationship_context = {
                        "character1": char1,
                        "character2": char2,
                        "existing_relationships": relationship_network
                    }
                    
                    relationship = await self.character_chains["relationship_development"].arun(
                        relationship_context=relationship_context
                    )
                    
                    relationship_network[char1][char2] = relationship
                    if char2 not in relationship_network:
                        relationship_network[char2] = {}
                    relationship_network[char2][char1] = self._invert_relationship(relationship)
        
        return relationship_network

    def _invert_relationship(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Invert a relationship description for the other character's perspective."""
        return {
            "type": relationship.get("type", ""),
            "dynamics": relationship.get("dynamics", ""),
            "history": relationship.get("history", ""),
            "feelings": relationship.get("reciprocal_feelings", relationship.get("feelings", "")),
            "reciprocal_feelings": relationship.get("feelings", ""),
            "conflicts": relationship.get("conflicts", []),
            "development_potential": relationship.get("development_potential", "")
        }

    async def _develop_character_arcs(self, story_bible: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Develop character arcs for all major characters."""
        character_arcs = {}
        
        for character_name in self.relationship_matrix:
            arc_context = {
                "character_name": character_name,
                "story_themes": story_bible.get("themes", []),
                "plot_points": story_bible.get("plot_points", {}),
                "character_relationships": self.relationship_matrix[character_name]
            }
            
            arc = await self.character_chains["character_arc"].arun(
                arc_context=arc_context
            )
            
            character_arcs[character_name] = {
                "arc_type": arc.get("arc_type", ""),
                "development_stages": arc.get("development_stages", []),
                "key_moments": arc.get("key_moments", []),
                "thematic_significance": arc.get("thematic_significance", ""),
                "resolution": arc.get("resolution", "")
            }
        
        return character_arcs

    async def cleanup(self) -> None:
        """Cleanup resources after task completion."""
        try:
            # Clear relationship matrix
            self.relationship_matrix = {}
            
            # Clear character chains
            self.character_chains = {}
            
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
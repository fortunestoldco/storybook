from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import json

class StoryWriterAgent(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.current_chapter = 1
        self.chapter_outline = {}
        self.system_prompts = self._load_system_prompts()
        self.writing_chains = self._initialize_writing_chains()

    def _load_system_prompts(self) -> Dict[str, Any]:
        """Load system prompts from configuration file."""
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    def _initialize_writing_chains(self) -> Dict[str, LLMChain]:
        """Initialize specialized chains for story writing."""
        chains = {}
        
        # Chapter outline chain
        outline_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("chapter_outline_instruction", "")),
            ("human", "{chapter_context}"),
            ("ai", "{chapter_outline}")
        ])
        chains["chapter_outline"] = LLMChain(
            llm=self.llm,
            prompt=outline_prompt,
            verbose=True
        )
        
        # Scene writing chain
        scene_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("scene_writing_instruction", "")),
            ("human", "{scene_context}"),
            ("ai", "{scene_content}")
        ])
        chains["scene_writing"] = LLMChain(
            llm=self.llm,
            prompt=scene_prompt,
            verbose=True
        )
        
        # Description chain
        description_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("description_instruction", "")),
            ("human", "{description_context}"),
            ("ai", "{description_content}")
        ])
        chains["description"] = LLMChain(
            llm=self.llm,
            prompt=description_prompt,
            verbose=True
        )
        
        return chains

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chapter writing tasks."""
        try:
            story_bible = task.get("story_bible", {})
            world_spec = task.get("world_spec", {})
            character_spec = task.get("character_spec", {})
            chapter_number = task.get("chapter_number", self.current_chapter)

            chapter_content = {
                "novel_id": story_bible.get("novel_id"),
                "chapter_number": chapter_number,
                "title": await self._generate_chapter_title(chapter_number, story_bible),
                "outline": await self._create_chapter_outline(
                    chapter_number,
                    story_bible,
                    character_spec
                ),
                "scenes": await self._write_chapter_scenes(
                    chapter_number,
                    story_bible,
                    world_spec,
                    character_spec
                ),
                "dialogue_placeholders": [],
                "status": "draft_complete",
                "created_at": datetime.utcnow().isoformat(),
                "word_count": 0
            }

            # Calculate word count
            chapter_content["word_count"] = self._calculate_word_count(chapter_content["scenes"])

            # Store chapter in MongoDB
            await self.mongodb_service.store_chapter(
                story_bible["novel_id"],
                chapter_number,
                chapter_content
            )

            return chapter_content
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "task": "chapter_writing",
                "chapter_number": chapter_number,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.mongodb_service.store_error(error_data)
            raise

    async def _generate_chapter_title(
        self,
        chapter_number: int,
        story_bible: Dict[str, Any]
    ) -> str:
        """Generate an appropriate title for the chapter."""
        try:
            chapter_context = {
                "chapter_number": chapter_number,
                "story_themes": story_bible.get("themes", []),
                "plot_points": story_bible.get("plot_points", {}).get(str(chapter_number), []),
                "chapter_focus": story_bible.get("chapter_focus", {}).get(str(chapter_number), "")
            }
            
            title_response = await self.writing_chains["chapter_outline"].arun(
                chapter_context=chapter_context
            )
            
            return title_response.get("title", f"Chapter {chapter_number}")
            
        except Exception as e:
            return f"Chapter {chapter_number}"

    async def _create_chapter_outline(
        self,
        chapter_number: int,
        story_bible: Dict[str, Any],
        character_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a detailed outline for the chapter."""
        try:
            chapter_context = {
                "chapter_number": chapter_number,
                "plot_points": story_bible.get("plot_points", {}).get(str(chapter_number), []),
                "characters": character_spec,
                "themes": story_bible.get("themes", []),
                "previous_chapters": await self._get_previous_chapters_summary(
                    story_bible["novel_id"],
                    chapter_number
                )
            }
            
            outline_response = await self.writing_chains["chapter_outline"].arun(
                chapter_context=chapter_context
            )
            
            self.chapter_outline = {
                "key_events": outline_response.get("key_events", []),
                "character_arcs": outline_response.get("character_arcs", {}),
                "scene_sequence": outline_response.get("scene_sequence", []),
                "thematic_elements": outline_response.get("thematic_elements", []),
                "pacing_notes": outline_response.get("pacing_notes", {})
            }
            
            return self.chapter_outline
            
        except Exception as e:
            return {"error": str(e)}

    async def _write_chapter_scenes(
        self,
        chapter_number: int,
        story_bible: Dict[str, Any],
        world_spec: Dict[str, Any],
        character_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Write the scenes for the chapter."""
        scenes = []
        
        for scene_outline in self.chapter_outline["scene_sequence"]:
            scene_context = {
                "scene_outline": scene_outline,
                "world_spec": world_spec,
                "character_spec": character_spec,
                "story_bible": story_bible,
                "previous_scenes": scenes,
                "chapter_number": chapter_number
            }
            
            scene = await self._write_scene(scene_context)
            scenes.append(scene)
            
            # Update progress in MongoDB
            await self._update_writing_progress(
                story_bible["novel_id"],
                chapter_number,
                len(scenes),
                len(self.chapter_outline["scene_sequence"])
            )
        
        return scenes

    async def _write_scene(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Write a single scene."""
        try:
            scene_response = await self.writing_chains["scene_writing"].arun(
                scene_context=scene_context
            )
            
            scene = {
                "scene_number": len(scene_context["previous_scenes"]) + 1,
                "setting": await self._describe_scene_setting(
                    scene_response["setting_context"],
                    scene_context["world_spec"]
                ),
                "characters_present": scene_response.get("characters_present", []),
                "narrative": scene_response.get("narrative", ""),
                "dialogue_markers": self._place_dialogue_markers(scene_response),
                "pacing": scene_response.get("pacing", "moderate"),
                "mood": scene_response.get("mood", "neutral"),
                "created_at": datetime.utcnow().isoformat()
            }
            
            return scene
            
        except Exception as e:
            return {"error": str(e)}

    async def _describe_scene_setting(
        self,
        setting_context: Dict[str, Any],
        world_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed scene setting description."""
        try:
            description_context = {
                "setting": setting_context,
                "world_elements": world_spec,
                "mood": setting_context.get("mood", "neutral"),
                "time_of_day": setting_context.get("time_of_day", ""),
                "weather": setting_context.get("weather", ""),
                "sensory_details": True
            }
            
            description = await self.writing_chains["description"].arun(
                description_context=description_context
            )
            
            return {
                "location": setting_context.get("location", ""),
                "description": description.get("description", ""),
                "sensory_details": description.get("sensory_details", {}),
                "atmosphere": description.get("atmosphere", "")
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _place_dialogue_markers(self, scene_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Place markers for dialogue to be filled by DialogueWriterAgent."""
        dialogue_markers = []
        
        for interaction in scene_response.get("character_interactions", []):
            marker = {
                "speakers": interaction.get("speakers", []),
                "context": interaction.get("context", ""),
                "emotional_state": interaction.get("emotional_state", {}),
                "purpose": interaction.get("purpose", ""),
                "position": interaction.get("position", "")
            }
            dialogue_markers.append(marker)
        
        return dialogue_markers

    async def _update_writing_progress(
        self,
        novel_id: str,
        chapter_number: int,
        scenes_complete: int,
        total_scenes: int
    ) -> None:
        """Update the writing progress in MongoDB."""
        progress = {
            "novel_id": novel_id,
            "chapter_number": chapter_number,
            "scenes_complete": scenes_complete,
            "total_scenes": total_scenes,
            "percentage_complete": (scenes_complete / total_scenes) * 100 if total_scenes > 0 else 0,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await self.mongodb_service.update_writing_progress(progress)

    def _calculate_word_count(self, scenes: List[Dict[str, Any]]) -> int:
        """Calculate the total word count for the chapter."""
        word_count = 0
        
        for scene in scenes:
            if "narrative" in scene:
                word_count += len(scene["narrative"].split())
            if "setting" in scene and "description" in scene["setting"]:
                word_count += len(scene["setting"]["description"].split())
        
        return word_count

    async def cleanup(self) -> None:
        """Cleanup resources after task completion."""
        try:
            # Clear chapter outline
            self.chapter_outline = {}
            
            # Clear writing chains
            self.writing_chains = {}
            
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

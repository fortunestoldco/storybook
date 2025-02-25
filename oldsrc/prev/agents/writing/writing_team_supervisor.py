from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
from langchain.chat_models import (
    ChatOpenAI,
    ChatAnthropic,
    ChatHuggingFace,
    ChatGoogleGenerativeAI,
    ChatOllama
)
from agents.writing.world_builder_agent import WorldBuilderAgent
from agents.writing.character_builder_agent import CharacterBuilderAgent
from agents.writing.story_writer_agent import StoryWriterAgent
from agents.writing.dialogue_writer_agent import DialogueWriterAgent
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import json

class WritingTeamSupervisor(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.current_task = None
        self.team_status = {
            "world_builder": "idle",
            "character_builder": "idle",
            "story_writer": "idle",
            "dialogue_writer": "idle"
        }
        self.system_prompts = self._load_system_prompts()
        self.supervision_chains = self._initialize_supervision_chains()
        self.story_writer_agents = self._initialize_story_writer_agents()
        self.dialogue_writer_agents = self._initialize_dialogue_writer_agents()
        self.default_story_writer = self.story_writer_agents["ChatAnthropic"]
        self.default_dialogue_writer = self.dialogue_writer_agents["ChatAnthropic"]

    def _load_system_prompts(self) -> Dict[str, Any]:
        """Load system prompts from configuration file."""
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    def _initialize_supervision_chains(self) -> Dict[str, LLMChain]:
        """Initialize specialized chains for team supervision."""
        chains = {}
        
        # Quality assessment chain
        quality_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("quality_assessment_instruction", "")),
            ("human", "{content}"),
            ("ai", "{assessment}")
        ])
        chains["quality_assessment"] = LLMChain(
            llm=self.llm,
            prompt=quality_prompt,
            verbose=True
        )
        
        # Coordination chain
        coordination_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("team_coordination_instruction", "")),
            ("human", "{task_status}"),
            ("ai", "{coordination_plan}")
        ])
        chains["coordination"] = LLMChain(
            llm=self.llm,
            prompt=coordination_prompt,
            verbose=True
        )
        
        return chains

    def _initialize_story_writer_agents(self) -> Dict[str, StoryWriterAgent]:
        """Initialize different LLM-based story writer agents."""
        return {
            "ChatOpenAI": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatOpenAI(),
                name="OpenAI_Writer"
            ),
            "ChatAnthropic": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatAnthropic(),
                name="Anthropic_Writer"
            ),
            "ChatHuggingFace": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatHuggingFace(),
                name="HF_Writer"
            ),
            "ChatGoogleGenerativeAI": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatGoogleGenerativeAI(),
                name="Google_Writer"
            ),
            "ChatOllama": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatOllama(),
                name="Ollama_Writer"
            )
        }

    def _initialize_dialogue_writer_agents(self) -> Dict[str, DialogueWriterAgent]:
        """Initialize different LLM-based dialogue writer agents."""
        return {
            "ChatOpenAI": DialogueWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatOpenAI(),
                name="OpenAI_Dialogue"
            ),
            "ChatAnthropic": DialogueWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatAnthropic(),
                name="Anthropic_Dialogue"
            ),
            "ChatHuggingFace": DialogueWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatHuggingFace(),
                name="HF_Dialogue"
            ),
            "ChatGoogleGenerativeAI": DialogueWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatGoogleGenerativeAI(),
                name="Google_Dialogue"
            ),
            "ChatOllama": DialogueWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatOllama(),
                name="Ollama_Dialogue"
            )
        }

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle writing team coordination tasks."""
        try:
            task_type = task.get("type")
            
            if task_type == "initialize_project":
                return await self._initialize_project(task)
            elif task_type == "manage_writing_phase":
                return await self._manage_writing_phase(task)
            elif task_type == "review_chapter":
                return await self._review_chapter(task)
            elif task_type == "multi_writer_review":
                return await self._handle_multi_writer_review(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            error_data = {
                "error": str(e),
                "task": task_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.mongodb_service.store_error(error_data)
            raise

    async def _initialize_project(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the writing project with the story bible."""
        try:
            story_bible = task.get("story_bible", {})
            novel_id = story_bible.get("novel_id")

            # Update team status
            self.team_status["world_builder"] = "working"
            
            # Create project structure
            project_structure = {
                "novel_id": novel_id,
                "status": "initializing",
                "world_building_complete": False,
                "character_development_complete": False,
                "current_chapter": 0,
                "total_chapters": story_bible.get("chapter_count", 0),
                "chapters_complete": [],
                "chapters_in_progress": [],
                "feedback_rounds": [],
                "created_at": datetime.utcnow().isoformat()
            }

            # Initialize world building
            world_builder = WorldBuilderAgent(
                mongodb_service=self.mongodb_service,
                llm=self.llm,
                name="World_Builder"
            )
            await world_builder.handle_task({
                "type": "world_building",
                "story_bible": story_bible
            })

            # Store project structure
            await self.mongodb_service.store_project(project_structure)

            return {
                "status": "initialized",
                "next_step": "world_building",
                "project_structure": project_structure
            }
            
        except Exception as e:
            self.team_status["world_builder"] = "error"
            raise

    async def _manage_writing_phase(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manage the writing phase of the project."""
        phase = task.get("phase")
        novel_id = task.get("novel_id")
        
        if phase == "world_building":
            return await self._manage_world_building(novel_id)
        elif phase == "character_development":
            return await self._manage_character_development(novel_id)
        elif phase == "chapter_writing":
            return await self._manage_chapter_writing(task)
        else:
            raise ValueError(f"Unknown writing phase: {phase}")

    async def _manage_world_building(self, novel_id: str) -> Dict[str, Any]:
        """Manage the world building phase."""
        try:
            # Get world building specifications
            world_spec = await self.mongodb_service.get_world_spec(novel_id)
            
            if world_spec.get("status") == "complete":
                self.team_status["world_builder"] = "idle"
                self.team_status["character_builder"] = "working"
                
                return {
                    "status": "complete",
                    "next_phase": "character_development",
                    "world_spec": world_spec
                }
            
            return {
                "status": "in_progress",
                "current_phase": "world_building",
                "world_spec": world_spec
            }
            
        except Exception as e:
            self.team_status["world_builder"] = "error"
            raise

    async def _manage_character_development(self, novel_id: str) -> Dict[str, Any]:
        """Manage the character development phase."""
        try:
            # Get character specifications
            character_spec = await self.mongodb_service.get_character_spec(novel_id)
            
            if character_spec.get("status") == "complete":
                self.team_status["character_builder"] = "idle"
                self.team_status["story_writer"] = "working"
                self.team_status["dialogue_writer"] = "working"
                
                return {
                    "status": "complete",
                    "next_phase": "chapter_writing",
                    "character_spec": character_spec
                }
            
            return {
                "status": "in_progress",
                "current_phase": "character_development",
                "character_spec": character_spec
            }
            
        except Exception as e:
            self.team_status["character_builder"] = "error"
            raise

    async def _manage_chapter_writing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manage the chapter writing process."""
        try:
            novel_id = task["novel_id"]
            chapter_number = task["chapter_number"]
            
            # Get chapter status
            chapter_status = await self._get_chapter_status(novel_id, chapter_number)
            
            if chapter_status["status"] == "complete":
                # Move to next chapter
                next_chapter = chapter_number + 1
                project = await self.mongodb_service.get_project(novel_id)
                
                if next_chapter > project["total_chapters"]:
                    return {
                        "status": "novel_complete",
                        "final_chapter": chapter_number
                    }
                
                return {
                    "status": "next_chapter",
                    "next_chapter_number": next_chapter,
                    "previous_chapter": chapter_status
                }
            
            return chapter_status
            
        except Exception as e:
            self.team_status["story_writer"] = "error"
            self.team_status["dialogue_writer"] = "error"
            raise

    async def _review_chapter(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review a completed chapter for quality."""
        try:
            novel_id = task["novel_id"]
            chapter_number = task["chapter_number"]
            chapter = await self.mongodb_service.get_chapter(novel_id, chapter_number)
            
            # Perform quality assessment
            assessment = await self.supervision_chains["quality_assessment"].arun(
                content=chapter
            )
            
            if assessment.get("quality_score", 0) < 0.8:
                return await self._handle_chapter_revision(task, assessment)
            
            return {
                "status": "approved",
                "assessment": assessment,
                "chapter_number": chapter_number
            }
            
        except Exception as e:
            raise

    async def _handle_multi_writer_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multiple writers reviewing and improving content."""
        try:
            novel_id = task["novel_id"]
            chapter_number = task["chapter_number"]
            content = task["content"]
            
            reviews = {}
            improvements = {}
            
            # Get reviews from different writer agents
            for agent_name, agent in self.story_writer_agents.items():
                if agent != self.default_story_writer:
                    review = await agent.review_content(content)
                    reviews[agent_name] = review
                    
                    if review.get("needs_improvement", False):
                        improvement = await agent.improve_content(content, review["suggestions"])
                        improvements[agent_name] = improvement
            
            # Analyze reviews and improvements
            analysis = await self._analyze_multi_writer_feedback(reviews, improvements)
            
            # Store review results
            await self.mongodb_service.store_review_results(
                novel_id,
                chapter_number,
                {
                    "reviews": reviews,
                    "improvements": improvements,
                    "analysis": analysis,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return {
                "status": "review_complete",
                "analysis": analysis,
                "recommended_changes": analysis.get("recommended_changes", [])
            }
            
        except Exception as e:
            raise

    async def cleanup(self) -> None:
        """Cleanup resources after task completion."""
        try:
            # Reset team status
            self.team_status = {
                "world_builder": "idle",
                "character_builder": "idle",
                "story_writer": "idle",
                "dialogue_writer": "idle"
            }
            
            # Clear current task
            self.current_task = None
            
            # Clear supervision chains
            self.supervision_chains = {}
            
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

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
from config.llm_config import LLMRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class AuthorFeedback(BaseModel):
    """Model for structured author feedback."""
    content_quality: int = Field(description="Rating from 1-10 for content quality")
    pacing: int = Field(description="Rating from 1-10 for story pacing")
    character_development: int = Field(description="Rating from 1-10 for character development")
    plot_coherence: int = Field(description="Rating from 1-10 for plot coherence")
    suggestions: List[str] = Field(description="List of specific improvement suggestions")
    overall_rating: int = Field(description="Overall rating from 1-10")

class AuthorRelationsAgent(BaseAgent):
    def __init__(self, tools_service, mongodb_service: MongoDBService):
        super().__init__(tools_service)
        self.logger = logging.getLogger(__name__)
        self.mongodb_service = mongodb_service
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route tasks to appropriate handlers."""
        try:
            task_type = task.get("type")
            if task_type == "brainstorm":
                return await self._handle_brainstorm(task)
            elif task_type == "feedback":
                return await self._handle_feedback(task)
            elif task_type == "import_brainstorm":
                return await self._handle_import_brainstorm(task)
            elif task_type == "existing_brainstorm":
                return await self._handle_existing_brainstorm(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _handle_brainstorm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a brainstorming session with the user."""
        try:
            novel_id = task["novel_id"]
            initial_prompt = task["initial_prompt"]
            
            session_data = {
                "novel_id": novel_id,
                "session_type": "brainstorm",
                "notes": [],
                "conclusions": {}
            }
            
            brainstorm_response = await self.llm_router.process_with_streaming(
                task="brainstorm",
                prompt=initial_prompt
            )
            session_data["notes"].append(brainstorm_response)
            
            session_data["conclusions"] = await self._generate_conclusions(session_data["notes"])
            
            return session_data
        except Exception as e:
            self.logger.error(f"Error in brainstorm session: {str(e)}")
            raise

    async def _handle_feedback(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user feedback on a draft."""
        try:
            novel_id = task["novel_id"]
            draft_number = task["draft_number"]
            feedback_text = task["feedback"]
            
            parser = PydanticOutputParser(pydantic_object=AuthorFeedback)
            prompt = (
                "Analyze the following author feedback and provide structured ratings:\n\n"
                f"{feedback_text}\n\n"
                f"{parser.get_format_instructions()}"
            )
            
            feedback_analysis = await self.llm_router.process_with_streaming(
                task="feedback_analysis",
                prompt=prompt,
                parser=parser
            )
            
            feedback_report = {
                "novel_id": novel_id,
                "draft_number": draft_number,
                "raw_feedback": feedback_text,
                "analyzed_feedback": feedback_analysis.dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            await self.mongodb_service.store_feedback(novel_id, feedback_report)
            return feedback_report
        except Exception as e:
            self.logger.error(f"Error handling feedback: {str(e)}")
            raise

    async def _handle_import_brainstorm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a brainstorming session for the imported manuscript."""
        try:
            novel_id = task["novel_id"]
            manuscript = task["manuscript"]
            initial_prompt = task["initial_prompt"]
            
            session_data = {
                "novel_id": novel_id,
                "session_type": "import_brainstorm",
                "manuscript": manuscript,
                "notes": [],
                "conclusions": {}
            }
            
            # Analyze imported manuscript
            analysis_prompt = (
                "Analyze the following manuscript and provide insights for improvement:\n\n"
                f"{manuscript}\n\n"
                f"Initial considerations: {initial_prompt}"
            )
            
            analysis_response = await self.llm_router.process_with_streaming(
                task="manuscript_analysis",
                prompt=analysis_prompt
            )
            session_data["notes"].append(analysis_response)
            
            session_data["conclusions"] = await self._generate_conclusions(session_data["notes"])
            
            return session_data
        except Exception as e:
            self.logger.error(f"Error in import brainstorm: {str(e)}")
            raise

    async def _handle_existing_brainstorm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a brainstorming session for an existing project."""
        try:
            novel_id = task["novel_id"]
            initial_prompt = task["initial_prompt"]
            
            # Get existing project data
            project_data = await self.mongodb_service.get_project(novel_id)
            
            session_data = {
                "novel_id": novel_id,
                "session_type": "existing_brainstorm",
                "notes": [],
                "conclusions": {},
                "project_context": project_data
            }
            
            context_prompt = (
                "Review the existing project and provide insights for development:\n\n"
                f"Project Data: {project_data}\n\n"
                f"Focus Areas: {initial_prompt}"
            )
            
            brainstorm_response = await self.llm_router.process_with_streaming(
                task="project_analysis",
                prompt=context_prompt
            )
            session_data["notes"].append(brainstorm_response)
            
            session_data["conclusions"] = await self._generate_conclusions(session_data["notes"])
            
            return session_data
        except Exception as e:
            self.logger.error(f"Error in existing project brainstorm: {str(e)}")
            raise

<<<<<<< HEAD
    async def _generate_conclusions(self, notes: List[str]) -> Dict[str, Any]:
        """Generate structured conclusions from brainstorming notes."""
        try:
            notes_text = "\n".join(notes)
            prompt = (
                "Analyze these brainstorming notes and generate structured conclusions:\n\n"
                f"{notes_text}\n\n"
                "Include: key themes, character insights, plot suggestions, and next steps"
            )
            
            conclusions = await self.llm_router.process_with_streaming(
                task="analysis",
                prompt=prompt
            )
            
            return conclusions
        except Exception as e:
            self.logger.error(f"Error generating conclusions: {str(e)}")
            raise
=======
    def _generate_conclusions(self, notes: List[str]) -> Dict[str, Any]:
        """Generate conclusions based on brainstorming notes."""
        conclusions = {}
        for note in notes:
            # Process each note and generate conclusions
            # Example logic: extract key points and summarize
            key_points = self._extract_key_points(note)
            for point in key_points:
                if point not in conclusions:
                    conclusions[point] = 0
                conclusions[point] += 1
        return conclusions

    def _extract_key_points(self, note: str) -> List[str]:
        """Extract key points from a note."""
        # Example logic: split note into sentences and extract key points
        sentences = note.split(".")
        key_points = [sentence.strip() for sentence in sentences if sentence]
        return key_points

    def _analyze_feedback(self, feedback: str) -> Dict[str, Any]:
        """Analyze user feedback."""
        analysis = {}
        # Example logic: categorize feedback into positive, negative, and suggestions
        feedback_lines = feedback.split("\n")
        analysis["positive"] = [line for line in feedback_lines if "good" in line.lower()]
        analysis["negative"] = [line for line in feedback_lines if "bad" in line.lower()]
        analysis["suggestions"] = [line for line in feedback_lines if "suggest" in line.lower()]
        return analysis

    def _generate_recommendations(self, feedback: str) -> List[str]:
        """Generate recommendations based on feedback."""
        recommendations = []
        # Example logic: generate recommendations based on feedback analysis
        analysis = self._analyze_feedback(feedback)
        if analysis["positive"]:
            recommendations.append("Continue with the positive aspects mentioned.")
        if analysis["negative"]:
            recommendations.append("Address the negative aspects mentioned.")
        if analysis["suggestions"]:
            recommendations.append("Consider the suggestions provided.")
        return recommendations

    async def reflect(self, result: Dict[str, Any]) -> bool:
        """Reflect on the quality and completeness of the task result."""
        # Implement reflection logic here
        # Check if all required information has been gathered
        # Verify the quality of the interaction
        required_sections = ["notes", "conclusions"]
        if not all(section in result for section in required_sections):
            return False
        return True

    async def cleanup(self) -> None:
        """Cleanup after task completion."""
        self.chat_history.clear()

>>>>>>> 55b233f059c21547cec6e1c16bc34951a6ecdc92
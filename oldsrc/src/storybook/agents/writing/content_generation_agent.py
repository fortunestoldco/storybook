# content_generation_agent.py

from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ContentGenerationAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            themes = task["themes"]
            character_profiles = task["character_profiles"]
            prompt = ChatPromptTemplate.from_template("Generate a chapter outline for a fantasy novel based on the following themes and character profiles:\n\n{themes}\n\n{character_profiles}")
            chapter_outline = await self.llm_router.process_with_streaming(task="generate_chapter_outline", prompt=prompt)
            return {"chapter_outline": chapter_outline}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

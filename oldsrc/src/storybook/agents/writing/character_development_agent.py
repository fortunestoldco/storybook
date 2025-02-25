# character_development_agent.py

from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class CharacterDevelopmentAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            character_profile = task["character_profile"]
            prompt = ChatPromptTemplate.from_template("Design a character arc for the following character profile:\n\n{character_profile}")
            character_arc = await self.llm_router.process_with_streaming(task="design_character_arc", prompt=prompt)
            return {"character_arc": character_arc}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

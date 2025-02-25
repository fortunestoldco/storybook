# world_building_agent.py

from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class WorldBuildingAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            world_description = task["world_description"]
            prompt = ChatPromptTemplate.from_template("Generate a detailed world description based on the following themes and setting:\n\n{world_description}")
            world_details = await self.llm_router.process_with_streaming(task="generate_world_details", prompt=prompt)
            return {"world_details": world_details}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

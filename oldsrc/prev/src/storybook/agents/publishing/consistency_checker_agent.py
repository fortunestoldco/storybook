from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ConsistencyCheckerAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            manuscript = task["manuscript"]
            prompt = ChatPromptTemplate.from_template("Check the following manuscript for consistency in plot, character traits, and narrative style:\n\n{manuscript}")
            consistency_report = await self.llm_router.process_with_streaming(task="check_consistency", prompt=prompt)
            return {"consistency_report": consistency_report}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

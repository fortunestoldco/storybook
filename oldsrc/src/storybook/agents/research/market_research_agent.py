# market_research_agent.py

from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class MarketResearchAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            market_data = task["market_data"]
            prompt = ChatPromptTemplate.from_template("Analyze the following market data and identify trending genres and themes:\n\n{market_data}")
            analysis = await self.llm_router.process_with_streaming(task="analyze_market_data", prompt=prompt)
            return {"analysis": analysis}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

# consumer_insights_agent.py

from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ConsumerInsightsAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            survey_responses = task["survey_responses"]
            prompt = ChatPromptTemplate.from_template("Analyze the following survey responses and identify key consumer insights:\n\n{survey_responses}")
            insights = await self.llm_router.process_with_streaming(task="analyze_survey_responses", prompt=prompt)
            return {"insights": insights}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

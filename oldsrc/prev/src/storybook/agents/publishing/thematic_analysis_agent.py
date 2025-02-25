from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ThematicAnalysisAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type == "analyze_theme":
                return await self._analyze_theme(task)
            elif task_type == "improve_theme":
                return await self._improve_theme(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _analyze_theme(self, task: Dict[str, Any]) -> Dict[str, Any]:
        manuscript = task["manuscript"]
        prompt = ChatPromptTemplate.from_template("Identify the key themes in the following manuscript:\n\n{manuscript}")
        thematic_analysis = await self.llm_router.process_with_streaming(task="analyze_theme", prompt=prompt)
        return {"thematic_analysis": thematic_analysis}

    async def _improve_theme(self, task: Dict[str, Any]) -> Dict[str, Any]:
        identified_themes = task["identified_themes"]
        prompt = ChatPromptTemplate.from_template("Suggest improvements to ensure the following themes are consistently conveyed throughout the manuscript:\n\n{identified_themes}")
        theme_improvement = await self.llm_router.process_with_streaming(task="improve_theme", prompt=prompt)
        return {"theme_improvement": theme_improvement}

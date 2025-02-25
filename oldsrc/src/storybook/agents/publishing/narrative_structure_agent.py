from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class NarrativeStructureAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type == "analyze_structure":
                return await self._analyze_structure(task)
            elif task_type == "improve_structure":
                return await self._improve_structure(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _analyze_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        manuscript = task["manuscript"]
        prompt = ChatPromptTemplate.from_template("Analyze the narrative structure of the following manuscript:\n\n{manuscript}")
        structure_analysis = await self.llm_router.process_with_streaming(task="analyze_structure", prompt=prompt)
        return {"structure_analysis": structure_analysis}

    async def _improve_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        structure_analysis = task["structure_analysis"]
        prompt = ChatPromptTemplate.from_template("Suggest improvements to enhance the narrative structure of the following text:\n\n{structure_analysis}")
        structure_improvement = await self.llm_router.process_with_streaming(task="improve_structure", prompt=prompt)
        return {"structure_improvement": structure_improvement}

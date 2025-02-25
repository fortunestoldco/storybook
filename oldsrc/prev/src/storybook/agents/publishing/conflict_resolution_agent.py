from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ConflictResolutionAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type == "analyze_conflict":
                return await self._analyze_conflict(task)
            elif task_type == "resolve_conflict":
                return await self._resolve_conflict(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _analyze_conflict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        interactions = task["interactions"]
        prompt = ChatPromptTemplate.from_template("Analyze the following interactions between characters and identify any conflicts:\n\n{interactions}")
        conflict_analysis = await self.llm_router.process_with_streaming(task="analyze_conflict", prompt=prompt)
        return {"conflict_analysis": conflict_analysis}

    async def _resolve_conflict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        conflict_details = task["conflict_details"]
        prompt = ChatPromptTemplate.from_template("Based on the identified conflicts, suggest potential strategies to resolve the conflicts:\n\n{conflict_details}")
        resolution_strategy = await self.llm_router.process_with_streaming(task="resolve_conflict", prompt=prompt)
        return {"resolution_strategy": resolution_strategy}

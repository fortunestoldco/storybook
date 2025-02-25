from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class EditorAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            manuscript = task["manuscript"]
            prompt = ChatPromptTemplate.from_template("Perform a detailed edit on the following manuscript:\n\n{manuscript}")
            edited_text = await self.llm_router.process_with_streaming(task="edit_manuscript", prompt=prompt)
            return {"edited_text": edited_text}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

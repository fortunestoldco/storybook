from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.feedback import (
    FeedbackProcessingTool,
    FeedbackIntegrationTool
)
from storybook.agents.base_agent import BaseAgent

class HumanFeedbackManager(BaseAgent):
    """Manager responsible for processing and integrating human feedback."""
    
    def __init__(self):
        super().__init__(
            name="human_feedback_manager",
            tools=[]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process human feedback tasks."""
        task = state.current_input.get("task", {})
        
        if "integration" in task.get("type", "").lower():
            integration = await self.tools[1].invoke({
                "content": state.project.content,
                "feedback": task.get("feedback", {}),
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Feedback integration completed")],
                "feedback_updates": {"integration": integration}
            }
        
        processing = await self.tools[0].invoke({
            "content": state.project.content,
            "feedback": task.get("feedback", {})
        })
        return {
            "messages": [AIMessage(content="Feedback processing completed")],
            "feedback_updates": {"processing": processing}
        }

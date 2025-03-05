from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.feedback import FeedbackProcessingTool
from storybook.agents.base_agent import BaseAgent

class HumanFeedbackManager(BaseAgent):
    """Manager responsible for processing human feedback."""
    
    def __init__(self):
        super().__init__(
            name="human_feedback_manager",
            tools=[FeedbackProcessingTool()]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process human feedback."""
        feedback = await self.tools[0].arun(
            content=state.project.content,
            feedback=state.current_input.get("feedback", {})
        )
        return {
            "messages": [AIMessage(content="Feedback processed")],
            "feedback_updates": feedback
        }
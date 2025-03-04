from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.creativity import CreativeAssessmentTool
from storybook.tools.story import StoryElementsTool
from storybook.agents.base_agent import BaseAgent

class CreativeDirector(BaseAgent):
    """Creative Director agent responsible for artistic vision and story elements."""
    
    def __init__(self):
        super().__init__(
            name="creative_director",
            tools=[
                CreativeAssessmentTool(),
                StoryElementsTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process creative aspects of the project."""
        task = state.current_input.get("task", "")
        
        # Assess creative elements
        if "assess" in task.lower():
            assessment = await self.tools[0].arun(
                content=state.project.content
            )
            return {
                "messages": [
                    AIMessage(content
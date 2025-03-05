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
                content=state.project.content,
                style_preferences=state.project.style_preferences
            )
            return {
                "messages": [
                    AIMessage(content="Creative assessment completed")
                ],
                "creative_updates": {"assessment": assessment}
            }
        
        # Handle story elements
        if "story" in task.lower() or "element" in task.lower():
            elements = await self.tools[1].arun(
                content=state.project.content,
                outline=state.project.content.get("outline", {}),
                characters=state.project.content.get("characters", {}),
                plot=state.project.content.get("plot", {})
            )
            return {
                "messages": [
                    AIMessage(content="Story elements updated")
                ],
                "creative_updates": {"story_elements": elements}
            }
        
        # Default to overall creative vision update
        vision = await self.tools[0].arun(
            content=state.project.content,
            style_preferences=state.project.style_preferences,
            current_phase=state.phase
        )
        
        return {
            "messages": [
                AIMessage(content="Creative vision updated")
            ],
            "creative_updates": {"vision": vision}
        }
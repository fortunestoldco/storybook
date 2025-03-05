from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.structure import (
    StoryStructureTool,
    PacingAnalysisTool,
    ChapterOutlineTool
)
from storybook.agents.base_agent import BaseAgent

class StructureArchitect(BaseAgent):
    """Agent responsible for story structure and pacing."""
    
    def __init__(self):
        super().__init__(
            name="structure_architect",
            tools=[
                StoryStructureTool(),
                PacingAnalysisTool(),
                ChapterOutlineTool()
            ]
        )
    
    async def process(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Process story structure tasks."""
        task = state.current_input.get("task", "")
        
        if "outline" in task.lower():
            outline = await self.tools[2].arun(
                content=state.project.content,
                structure=state.project.content.get("structure", {})
            )
            return {
                "messages": [AIMessage(content="Chapter outline updated")],
                "structure_updates": {"outline": outline}
            }
        
        # Default to structure analysis
        structure = await self.tools[0].arun(
            content=state.project.content
        )
        return {
            "messages": [AIMessage(content="Structure analyzed")],
            "structure_updates": {"analysis": structure}
        }
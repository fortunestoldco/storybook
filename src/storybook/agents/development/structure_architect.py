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
    """Architect responsible for story structure and pacing."""
    
    def __init__(self):
        super().__init__(
            name="structure_architect",
            tools=[
                StoryStructureTool(),
                PacingAnalysisTool(),
                ChapterOutlineTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        task = state.current_input.get("task", {})
        
        if "pacing" in task.get("type", "").lower():
            pacing = await self.tools[1].invoke({
                "content": state.project.content,
                "scope": task.get("scope", "global")
            })
            return {
                "messages": [AIMessage(content="Pacing analysis completed")],
                "structure_updates": {"pacing": pacing}
            }
        
        if "outline" in task.get("type", "").lower():
            outline = await self.tools[2].invoke({
                "content": state.project.content,
                "chapter_count": task.get("chapter_count")
            })
            return {
                "messages": [AIMessage(content="Chapter outline completed")],
                "structure_updates": {"outline": outline}
            }
        
        structure = await self.tools[0].invoke({
            "content": state.project.content,
            "structure_type": task.get("structure_type", "three_act")
        })
        return {
            "messages": [AIMessage(content="Story structure analysis completed")],
            "structure_updates": {"structure": structure}
        }

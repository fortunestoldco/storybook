from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.plot import PlotStructureTool, ConflictDevelopmentTool
from storybook.agents.base_agent import BaseAgent

class PlotDevelopmentSpecialist(BaseAgent):
    """Specialist responsible for plot development and structure."""
    
    def __init__(self):
        super().__init__(
            name="plot_development_specialist",
            tools=[
                PlotStructureTool(),
                ConflictDevelopmentTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process plot development tasks."""
        task = state.current_input.get("task", {})
        
        if "conflict" in task.get("type", "").lower():
            conflict = await self.tools[1].invoke({
                "content": state.project.content,
                "conflict_type": task.get("conflict_type", "external")
            })
            return {
                "messages": [AIMessage(content="Conflict development completed")],
                "plot_updates": {"conflict": conflict}
            }
            
        structure = await self.tools[0].invoke({
            "content": state.project.content,
            "structure_type": task.get("structure_type", "three_act")
        })
        return {
            "messages": [AIMessage(content="Plot structure updated")],
            "plot_updates": {"structure": structure}
        }
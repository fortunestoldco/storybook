from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.plot import (
    PlotThreadTool,
    ConflictDesignTool,
    PlotCoherenceTool
)
from storybook.agents.base_agent import BaseAgent

class PlotDevelopmentSpecialist(BaseAgent):
    """Specialist for developing and maintaining plot elements."""
    
    def __init__(self):
        super().__init__(
            name="plot_development_specialist",
            tools=[
                PlotThreadTool(),
                ConflictDesignTool(),
                PlotCoherenceTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Develop and refine plot elements."""
        task = state.current_input.get("task", "")
        
        if "conflict" in task.lower():
            conflicts = await self.tools[1].arun(
                content=state.project.content,
                characters=state.project.content.get("characters", {})
            )
            return {
                "messages": [AIMessage(content="Conflict design updated")],
                "plot_updates": {"conflicts": conflicts}
            }
        
        if "coherence" in task.lower():
            coherence = await self.tools[2].arun(
                content=state.project.content,
                plot_threads=state.project.content.get("plot_threads", [])
            )
            return {
                "messages": [AIMessage(content="Plot coherence verified")],
                "plot_updates": {"coherence": coherence}
            }
        
        # Default to plot thread management
        threads = await self.tools[0].arun(
            content=state.project.content,
            outline=state.project.content.get("outline", {})
        )
        
        return {
            "messages": [AIMessage(content="Plot threads updated")],
            "plot_updates": {"threads": threads}
        }
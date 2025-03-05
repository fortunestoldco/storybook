from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.structural import (
    StoryFlowTool,
    PacingAnalysisTool,
    SceneBalanceTool
)
from storybook.agents.base_agent import BaseAgent

class StructuralEditor(BaseAgent):
    """Editor responsible for overall structural integrity and flow."""
    
    def __init__(self):
        super().__init__(
            name="structural_editor",
            tools=[
                StoryFlowTool(),
                PacingAnalysisTool(),
                SceneBalanceTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Analyze and refine story structure."""
        task = state.current_input.get("task", "")
        
        if "pacing" in task.lower():
            pacing_analysis = await self.tools[1].arun(
                content=state.project.content,
                target_pacing=state.project.style_preferences.get("pacing", {}),
                chapter_breakdown=state.project.content.get("chapters", {})
            )
            return {
                "messages": [AIMessage(content="Pacing analysis completed")],
                "structural_updates": {"pacing": pacing_analysis}
            }
        
        if "scene_balance" in task.lower():
            balance = await self.tools[2].arun(
                content=state.project.content,
                scenes=state.project.content.get("scenes", {}),
                target_metrics=state.project.style_preferences.get("scene_metrics", {})
            )
            return {
                "messages": [AIMessage(content="Scene balance optimized")],
                "structural_updates": {"scene_balance": balance}
            }
        
        # Default to story flow analysis
        flow = await self.tools[0].arun(
            content=state.project.content,
            plot_threads=state.project.content.get("plot_threads", []),
            structure=state.project.content.get("structure", {})
        )
        
        return {
            "messages": [AIMessage(content="Story flow analysis completed")],
            "structural_updates": {"flow": flow}
        }

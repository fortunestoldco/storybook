from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.marketing import (
    TitleGenerationTool,
    BlurbOptimizationTool,
    KeywordAnalysisTool
)
from storybook.agents.base_agent import BaseAgent

class TitleBlurbOptimizer(BaseAgent):
    """Optimizer for title and marketing blurb creation."""
    
    def __init__(self):
        super().__init__(
            name="title_blurb_optimizer",
            tools=[
                TitleGenerationTool(),
                BlurbOptimizationTool(),
                KeywordAnalysisTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Optimize title and marketing blurb."""
        task = state.current_input.get("task", "")
        
        if "blurb" in task.lower():
            blurb = await self.tools[1].arun(
                content=state.project.content,
                target_audience=state.project.target_audience,
                market_position=state.project.content.get("market_position", {})
            )
            return {
                "messages": [AIMessage(content="Marketing blurb optimized")],
                "marketing_updates": {"blurb": blurb}
            }
        
        if "keywords" in task.lower():
            keywords = await self.tools[2].arun(
                content=state.project.content,
                genre=state.project.genre,
                market_trends=state.project.content.get("market_analysis", {})
            )
            return {
                "messages": [AIMessage(content="Keywords analyzed")],
                "marketing_updates": {"keywords": keywords}
            }
        
        # Default to title generation
        title = await self.tools[0].arun(
            content=state.project.content,
            genre=state.project.genre,
            target_audience=state.project.target_audience
        )
        
        return {
            "messages": [AIMessage(content="Title options generated")],
            "marketing_updates": {"title": title}
        }

from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.market import (
    MarketAnalysisTool,
    PositioningStrategyTool,
    CompetitorAnalysisTool
)
from storybook.agents.base_agent import BaseAgent

class PositioningSpecialist(BaseAgent):
    """Specialist responsible for market positioning."""
    
    def __init__(self):
        super().__init__(
            name="positioning_specialist",
            tools=[
                MarketAnalysisTool(),
                PositioningStrategyTool(),
                CompetitorAnalysisTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Develop and refine market positioning."""
        task = state.current_input.get("task", "")
        
        if "strategy" in task.lower():
            strategy = await self.tools[1].arun(
                content=state.project.content,
                market_analysis=state.project.content.get("market_analysis", {}),
                target_audience=state.project.target_audience
            )
            return {
                "messages": [AIMessage(content="Positioning strategy developed")],
                "positioning_updates": {"strategy": strategy}
            }
        
        if "competitor" in task.lower():
            analysis = await self.tools[2].arun(
                genre=state.project.genre,
                target_market=state.project.target_audience
            )
            return {
                "messages": [AIMessage(content="Competitor analysis completed")],
                "positioning_updates": {"competitor_analysis": analysis}
            }
        
        # Default to market analysis
        market = await self.tools[0].arun(
            genre=state.project.genre,
            target_audience=state.project.target_audience,
            unique_elements=state.project.content.get("unique_elements", [])
        )
        
        return {
            "messages": [AIMessage(content="Market analysis completed")],
            "positioning_updates": {"market": market}
        }